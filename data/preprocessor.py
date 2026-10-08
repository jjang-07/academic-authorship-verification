"""Data loading, preprocessing, and pair generation for all supported corpora.

Outputs the universal TextPair dataclass consumed by every downstream V2 component.

Supported corpora
-----------------
- Reuters-50-50         V1 benchmark; continuity baseline.
- PAN 2024 AV           SOTA comparison; AI-generated text framing.
- PAN 2023 AV           Cross-year robustness check.
- High School Essays    Real-world unique contribution; collected manually.
- Blog Authorship       Schler et al. (2006); Huang et al. (2024) comparison target.

Design notes
------------
- No NLP model is loaded here.  Heavy processing (POS tagging, perplexity) belongs in
  features/handcrafted.py so that dataset loading stays fast and test-friendly.
- All corpus loaders return List[TextPair].  Downstream components only consume that type.
- Pair generation guarantees 50/50 class balance via under-sampling the majority class.
- Text cleaning is deterministic and reversible enough that features can still be extracted.
"""

from __future__ import annotations

import html
import itertools
import json
import os
import random
import re
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from sklearn.model_selection import train_test_split
from tqdm import tqdm


# ─────────────────────────────── Universal Data Contract ──────────────────────

@dataclass
class TextPair:
    """Universal data contract passed between every V2 pipeline stage.

    Fields
    ------
    text_a         : Verified author text (the reference document).
    text_b         : Unknown / test text (the document to verify).
    label          : 1 = same author, 0 = different author.
    author_id      : Canonical author identifier for stratified evaluation.
    topic_a        : Topic tag for text_a — used by topic-stratified evaluation.
    topic_b        : Topic tag for text_b.
    source_dataset : Which corpus this pair came from (for ablation tracking).
    """

    text_a: str
    text_b: str
    label: int
    author_id: str
    topic_a: str
    topic_b: str
    source_dataset: str


# ─────────────────────────────── Text Cleaning ───────────────────────────────

def normalize_text(text: str) -> str:
    """Lowercase and replace all URL variants with the placeholder token URL."""
    text = text.lower()
    text = re.sub(
        r"((www\.[^\s]+)|(https?://[^\s]+)|(http?://[^\s]+))",
        " URL ",
        text,
    )
    return text


def remove_punct_only_lines(text: str) -> str:
    """Drop lines that contain no alphanumeric characters (headers, separators, etc.)."""
    lines = text.split("\n")
    kept = [ln.strip() for ln in lines if re.search(r"[A-Za-z0-9]", ln)]
    return " ".join(kept)


def clean_text(text: str) -> str:
    """Full cleaning pipeline: HTML unescape → strip punct-only lines → normalize."""
    text = html.unescape(text)
    text = remove_punct_only_lines(text)
    text = normalize_text(text)
    return text.strip()


# ─────────────────────────────── Pair Generation ──────────────────────────────

def _balance_and_shuffle(
    same: List[tuple],
    diff: List[tuple],
    rng: random.Random,
) -> List[tuple]:
    """Under-sample the majority class so same/different pairs are 50/50."""
    min_n = min(len(same), len(diff))
    balanced = rng.sample(same, min_n) + rng.sample(diff, min_n)
    rng.shuffle(balanced)
    return balanced


def create_pairs_from_author_dict(
    texts_by_author: Dict[str, List[str]],
    source_dataset: str,
    topics_by_author: Optional[Dict[str, List[str]]] = None,
    seed: int = 42,
) -> List[TextPair]:
    """Generate balanced same/different-author TextPairs.

    Parameters
    ----------
    texts_by_author  : {author_id -> [text_0, text_1, ...]}
    source_dataset   : Label attached to every resulting TextPair.
    topics_by_author : Optional {author_id -> [topic_0, topic_1, ...]}
                       parallel to texts_by_author.  Topic strings are
                       propagated to TextPair.topic_a / topic_b for
                       topic-stratified evaluation.
    seed             : Controls shuffling and down-sampling randomness.

    Same-author pairs : all pairwise combinations within each author.
    Different-author  : one randomly chosen text per author-pair combination.
    Balance           : majority class is down-sampled to match minority.
    """
    rng = random.Random(seed)
    same: list = []
    diff: list = []

    # Same-author pairs — all C(n, 2) combinations per author
    for author, texts in texts_by_author.items():
        top = (topics_by_author or {}).get(author, [])
        for (i, t1), (j, t2) in itertools.combinations(enumerate(texts), 2):
            same.append((
                t1, t2, 1, author,
                top[i] if i < len(top) else "",
                top[j] if j < len(top) else "",
            ))

    # Different-author pairs — one text per author-pair
    authors = list(texts_by_author.keys())
    for i in range(len(authors)):
        for j in range(i + 1, len(authors)):
            a1, a2 = authors[i], authors[j]
            t_a = texts_by_author[a1]
            t_b = texts_by_author[a2]
            top_a = (topics_by_author or {}).get(a1, [])
            top_b = (topics_by_author or {}).get(a2, [])
            ia = rng.randrange(len(t_a))
            ib = rng.randrange(len(t_b))
            diff.append((
                t_a[ia], t_b[ib], 0, a1,
                top_a[ia] if ia < len(top_a) else "",
                top_b[ib] if ib < len(top_b) else "",
            ))

    balanced = _balance_and_shuffle(same, diff, rng)
    n_same = sum(1 for x in balanced if x[2] == 1)
    n_diff = sum(1 for x in balanced if x[2] == 0)
    print(
        f"  [{source_dataset}] {n_same} same-author + {n_diff} different-author pairs "
        f"({len(balanced)} total)."
    )

    return [
        TextPair(
            text_a=t1,
            text_b=t2,
            label=label,
            author_id=author_id,
            topic_a=topic_a,
            topic_b=topic_b,
            source_dataset=source_dataset,
        )
        for t1, t2, label, author_id, topic_a, topic_b in balanced
    ]


# ─────────────────────────────── Reuters-50-50 ────────────────────────────────

def _find_c50_root(base_dir: str) -> str:
    """Resolve the directory that directly contains C50train and C50test.

    Handles common layouts:
    - base_dir = data/raw/reuter+50+50  → C50train, C50test are immediate children
    - base_dir = data/raw                → C50train, C50test live in a subdir (e.g. reuter+50+50)
    - base_dir = data/raw/reuters        → path may not exist; search parent for reuter+50+50 etc.
    """
    base = Path(base_dir)
    # Case 1: base_dir directly contains C50train and C50test
    if (base / "C50train").is_dir() and (base / "C50test").is_dir():
        return str(base)
    # Case 2: base_dir does not exist — search its parent for a subdir with C50 layout
    if not base.exists():
        parent = base.parent
        for subdir in sorted(parent.iterdir()):
            if subdir.is_dir() and not subdir.name.startswith("."):
                if (subdir / "C50train").is_dir() and (subdir / "C50test").is_dir():
                    return str(subdir)
    # Case 3: base_dir exists but C50train/C50test are in a subdirectory
    for subdir in sorted(base.iterdir()):
        if subdir.is_dir() and not subdir.name.startswith("."):
            if (subdir / "C50train").is_dir() and (subdir / "C50test").is_dir():
                return str(subdir)
    # Fallback: return base_dir as-is (caller will get 0 texts if layout is wrong)
    return base_dir


def load_reuters(base_dir: str, seed: int = 42) -> List[TextPair]:
    """Load Reuters-50-50 from the standard C50train / C50test layout.

    Expected structure::

        <root>/
            C50train/
                AuthorName/
                    article1.txt
                    article2.txt
                    ...
            C50test/
                AuthorName/
                    ...

    ``base_dir`` may point directly at the root, or at a parent that contains a
    subdirectory with this layout (e.g. ``data/raw`` containing ``reuter+50+50``).
    Both splits are merged into a single author-text map so each author has up to
    100 texts (50 train + 50 test) before pair creation.
    """
    root = _find_c50_root(base_dir)
    texts_by_author: Dict[str, List[str]] = {}

    for split_name in ("C50train", "C50test"):
        split_dir = os.path.join(root, split_name)
        if not os.path.isdir(split_dir):
            continue
        for author in sorted(os.listdir(split_dir)):
            author_dir = os.path.join(split_dir, author)
            if not os.path.isdir(author_dir):
                continue
            for fname in sorted(os.listdir(author_dir)):
                if not fname.endswith(".txt"):
                    continue
                fpath = os.path.join(author_dir, fname)
                with open(fpath, "r", encoding="utf-8", errors="replace") as f:
                    texts_by_author.setdefault(author, []).append(clean_text(f.read()))

    total_texts = sum(len(v) for v in texts_by_author.values())
    print(
        f"Reuters-50-50: {total_texts} texts from {len(texts_by_author)} authors."
    )
    return create_pairs_from_author_dict(texts_by_author, "Reuters-50-50", seed=seed)


# ─────────────────────────────── High-School Essays ───────────────────────────

def load_student_essays(base_dir: str, seed: int = 42) -> List[TextPair]:
    """Load the manually collected high-school essay corpus.

    Expected structure::

        <base_dir>/
            student_001/
                history_1.txt
                science_2.txt
                ...
            student_002/
                ...

    Topic is inferred from the filename stem. Files named ``history_1.txt``
    produce the topic string ``"history"``; files without an underscore use
    the full stem as the topic.
    """
    texts_by_author: Dict[str, List[str]] = {}
    topics_by_author: Dict[str, List[str]] = {}

    for author in sorted(os.listdir(base_dir)):
        author_dir = os.path.join(base_dir, author)
        if not os.path.isdir(author_dir):
            continue
        for fname in sorted(os.listdir(author_dir)):
            if not fname.endswith(".txt"):
                continue
            stem = os.path.splitext(fname)[0]
            topic = stem.rsplit("_", 1)[0] if "_" in stem else stem
            fpath = os.path.join(author_dir, fname)
            with open(fpath, "r", encoding="utf-8", errors="replace") as f:
                raw = f.read()
            texts_by_author.setdefault(author, []).append(clean_text(raw))
            topics_by_author.setdefault(author, []).append(topic)

    total_texts = sum(len(v) for v in texts_by_author.values())
    print(
        f"HighSchoolEssay: {total_texts} texts from {len(texts_by_author)} authors."
    )
    return create_pairs_from_author_dict(
        texts_by_author,
        "HighSchoolEssay",
        topics_by_author=topics_by_author,
        seed=seed,
    )


# ─────────────────────────────── PAN AV Benchmarks ───────────────────────────

def _parse_pan_label(raw) -> Optional[int]:
    """Normalise the diverse PAN label formats to 0 / 1."""
    if raw is None:
        return None
    if isinstance(raw, str):
        return 1 if raw.strip().upper() in {"Y", "TRUE", "1", "SAME"} else 0
    return int(bool(raw))


def _load_pan_jsonl(
    pairs_path: str,
    labels_path: Optional[str],
    source_tag: str,
) -> List[TextPair]:
    """Generic JSONL reader shared by PAN 2023 and PAN 2024 loaders.

    Supported pair-file formats (one object per line)::

        {"id": "...", "pair": ["text_a", "text_b"]}
        {"id": "...", "text1": "...", "text2": "..."}   # alternative key names

    Supported label-file formats (one object per line, optional)::

        {"id": "...", "same": true}
        {"id": "...", "label": 1}
        {"id": "...", "label": "Y"}

    Labels can also be inline in the pairs file.  The resolution priority is:
    inline field > separate labels file.  Pairs with no resolvable label are
    skipped rather than assigned a default.
    """
    raw_pairs: list = []
    with open(pairs_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                raw_pairs.append(json.loads(line))

    # Build id → label map from the separate labels file when provided
    label_map: Dict[str, int] = {}
    if labels_path and os.path.exists(labels_path):
        with open(labels_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                pid = obj.get("id", "")
                raw = obj.get("label", obj.get("same", None))
                parsed = _parse_pan_label(raw)
                if parsed is not None:
                    label_map[pid] = parsed

    result: List[TextPair] = []
    skipped = 0

    for obj in raw_pairs:
        pid = obj.get("id", "")

        # Extract text fields — support both 'pair' list and separate keys
        if "pair" in obj:
            text_a, text_b = obj["pair"][0], obj["pair"][1]
        else:
            text_a = obj.get("text1", obj.get("text_a", ""))
            text_b = obj.get("text2", obj.get("text_b", ""))

        # Resolve label: inline overrides separate file
        raw_label = obj.get("label", obj.get("same", label_map.get(pid)))
        label = _parse_pan_label(raw_label)
        if label is None:
            skipped += 1
            continue

        result.append(TextPair(
            text_a=clean_text(text_a),
            text_b=clean_text(text_b),
            label=label,
            author_id=pid,
            topic_a=obj.get("topic_a", obj.get("topic", "")),
            topic_b=obj.get("topic_b", ""),
            source_dataset=source_tag,
        ))

    if skipped:
        print(f"  [{source_tag}] Warning: skipped {skipped} pairs with no ground-truth label.")
    print(f"{source_tag}: {len(result)} labeled pairs loaded.")
    return result


def load_pan2024(data_dir: str) -> List[TextPair]:
    """Load PAN 2024 Authorship Verification benchmark.

    Expects ``<data_dir>/pairs.jsonl`` and optionally ``<data_dir>/labels.jsonl``.
    """
    pairs_path = os.path.join(data_dir, "pairs.jsonl")
    labels_path = os.path.join(data_dir, "labels.jsonl")
    return _load_pan_jsonl(
        pairs_path,
        labels_path if os.path.exists(labels_path) else None,
        "PAN2024",
    )


def load_pan2023(data_dir: str) -> List[TextPair]:
    """Load PAN 2023 Authorship Verification benchmark.

    Same directory layout as PAN 2024: ``pairs.jsonl`` + ``labels.jsonl``.
    """
    pairs_path = os.path.join(data_dir, "pairs.jsonl")
    labels_path = os.path.join(data_dir, "labels.jsonl")
    return _load_pan_jsonl(
        pairs_path,
        labels_path if os.path.exists(labels_path) else None,
        "PAN2023",
    )


# ─────────────────────────────── Blog Authorship Corpus ───────────────────────

def load_blog_authorship(
    data_dir: str,
    min_posts: int = 2,
    seed: int = 42,
) -> List[TextPair]:
    """Load the Blog Authorship Corpus (Schler et al., 2006).

    Kaggle XML layout — one file per blogger::

        <blogger_id>.<gender>.<age>.<topic>.<zodiac_sign>.xml

    Each XML file::

        <Blog>
            <date>11,May,2004</date>
            <post>... blog text ...</post>
            <date>...</date>
            <post>...</post>
        </Blog>

    The ``<date>`` elements are siblings of ``<post>``; they are not nested
    inside posts, so we iterate direct children and collect only ``<post>``
    elements.

    Parameters
    ----------
    min_posts : Minimum number of posts required to include an author.
                Authors with fewer posts are excluded so pair generation is
                meaningful.
    """
    texts_by_author: Dict[str, List[str]] = {}
    topics_by_author: Dict[str, List[str]] = {}

    xml_files = sorted(f for f in os.listdir(data_dir) if f.endswith(".xml"))
    skipped_parse_errors = 0

    for fname in tqdm(xml_files, desc="Loading Blog Authorship Corpus"):
        # Filename: id.gender.age.topic.sign.xml
        name_parts = fname[:-4].split(".")
        author_id = name_parts[0]
        topic = name_parts[3] if len(name_parts) >= 4 else "unknown"

        fpath = os.path.join(data_dir, fname)
        try:
            tree = ET.parse(fpath)
            root = tree.getroot()
        except ET.ParseError:
            skipped_parse_errors += 1
            continue

        posts: List[str] = []
        for elem in root:
            if elem.tag.lower() == "post":
                text = (elem.text or "").strip()
                if text:
                    posts.append(clean_text(text))

        if len(posts) >= min_posts:
            texts_by_author[author_id] = posts
            topics_by_author[author_id] = [topic] * len(posts)

    if skipped_parse_errors:
        print(
            f"  [BlogAuthorship] Warning: skipped {skipped_parse_errors} XML files "
            f"due to parse errors."
        )
    total_posts = sum(len(v) for v in texts_by_author.values())
    print(
        f"BlogAuthorship: {total_posts} posts from {len(texts_by_author)} bloggers "
        f"(≥ {min_posts} posts each)."
    )
    return create_pairs_from_author_dict(
        texts_by_author,
        "BlogAuthorship",
        topics_by_author=topics_by_author,
        seed=seed,
    )


# ─────────────────────────────── Train / Val / Test Split ─────────────────────

def split_pairs(
    pairs: List[TextPair],
    train: float = 0.70,
    val: float = 0.15,
    test: float = 0.15,
    seed: int = 42,
) -> Tuple[List[TextPair], List[TextPair], List[TextPair]]:
    """Stratified train / val / test split that preserves label balance.

    Returns
    -------
    (train_pairs, val_pairs, test_pairs)
    """
    assert abs(train + val + test - 1.0) < 1e-9, "Split fractions must sum to 1.0."
    labels = [p.label for p in pairs]
    train_pairs, temp = train_test_split(
        pairs, train_size=train, stratify=labels, random_state=seed
    )
    temp_labels = [p.label for p in temp]
    val_frac = val / (val + test)
    val_pairs, test_pairs = train_test_split(
        temp, train_size=val_frac, stratify=temp_labels, random_state=seed
    )
    return list(train_pairs), list(val_pairs), list(test_pairs)


# ─────────────────────────────── Serialisation Helpers ────────────────────────

def save_pairs_jsonl(pairs: List[TextPair], output_path: str) -> None:
    """Persist a list of TextPair objects to a JSONL file.

    Uses the same field names as the TextPair dataclass so that
    load_pairs_jsonl() can reconstruct them with ``TextPair(**obj)``.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(asdict(p)) + "\n")
    print(f"Saved {len(pairs)} pairs → {output_path}")


def load_pairs_jsonl(path: str) -> List[TextPair]:
    """Restore TextPair objects from a JSONL file written by save_pairs_jsonl()."""
    pairs: List[TextPair] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                pairs.append(TextPair(**json.loads(line)))
    return pairs
