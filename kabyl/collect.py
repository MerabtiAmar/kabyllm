"""Corpus aggregation: pull every registered source, normalize, filter, deduplicate.

Produces a single JSONL corpus with per-document provenance, plus per-source and
per-stage statistics. The statistics are the point as much as the corpus is -- the v0
project could not say how much data it had, where it came from, or what fraction was
duplicated, and that is why it trained a 43M-parameter model on 1M tokens.
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .dedup import dedup
from .filters import FilterConfig, FilterStats, LanguageID, clean_document
from .normalize import words
from .sources import Kind, Source

log = logging.getLogger(__name__)


@dataclass
class Doc:
    text: str
    source: str
    n_words: int = 0

    def __post_init__(self) -> None:
        if not self.n_words:
            self.n_words = len(words(self.text))


@dataclass
class CollectReport:
    per_source: dict[str, dict] = field(default_factory=dict)
    total_docs: int = 0
    total_words: int = 0
    dedup_removed: int = 0

    def table(self) -> str:
        w = max([len(k) for k in self.per_source] + [8])
        head = f"{'source':<{w}}  {'kept':>9}  {'seen':>9}  {'words':>12}  top rejection"
        lines = [head, "-" * len(head)]
        for key, s in sorted(self.per_source.items(), key=lambda kv: -kv[1]["words"]):
            top = s["top_reject"] or "-"
            lines.append(
                f"{key:<{w}}  {s['kept']:>9,}  {s['seen']:>9,}  {s['words']:>12,}  {top}"
            )
        lines.append("-" * len(head))
        lines.append(
            f"{'TOTAL':<{w}}  {self.total_docs:>9,}  {'':>9}  {self.total_words:>12,}"
        )
        return "\n".join(lines)


def _text_column(ds, declared: str | None) -> str:
    """Find the column holding Kabyle text.

    Community datasets are inconsistent -- `text`, `kab`, `sentence`, `content`,
    `paragraph` all occur -- so the column is guessed when not declared.
    """
    if declared and declared in ds.column_names:
        return declared
    for cand in ("text", "kab", "kab_Latn", "sentence", "content", "paragraph", "body"):
        if cand in ds.column_names:
            return cand
    strings = [
        c for c in ds.column_names
        if getattr(ds.features[c], "dtype", None) == "string"
    ]
    if not strings:
        raise ValueError(f"no string column among {ds.column_names}")
    # Fall back to the widest column, which is almost always the document body.
    sample = ds.select(range(min(200, len(ds))))
    return max(strings, key=lambda c: sum(len(str(v or "")) for v in sample[c]))


def _chunk_local(text: str, max_words: int = 400) -> list[str]:
    """Split a local corpus file into documents.

    Local files come in two shapes and guessing wrong is costly: paragraph-structured
    prose (blank-line separated) and sentence-per-line dumps like `kab_Latn.txt`. The
    latter has no blank lines at all, so a naive `split("\\n\\n")` yields one 2.4 MB
    "document" that every length filter rejects.
    """
    paras = [c.strip() for c in text.split("\n\n") if c.strip()]
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if len(paras) < max(2, len(lines) // 20):
        return lines  # no paragraph structure; treat each line as a document

    out: list[str] = []
    for p in paras:
        w = p.split()
        if len(w) <= max_words:
            out.append(p)
        else:  # over-long paragraph: window it rather than drop it
            for i in range(0, len(w), max_words):
                out.append(" ".join(w[i:i + max_words]))
    return out


#: Below this share of Kabyle-specific letters, a column is not Kabyle. Real Kabyle
#: runs 0.05-0.15; French and English sit near zero.
MIN_KABYLE_DIACRITIC_RATE = 0.01


def _looks_kabyle(values: list[str], sample: int = 300) -> float:
    from .normalize import diacritic_rate

    text = " ".join(str(v) for v in values[:sample] if v)
    return diacritic_rate(text)


def load_source(src: Source, limit: int | None = None) -> list[str]:
    """Fetch Kabyle strings for one source. Returns [] and logs on failure.

    Individual sources are allowed to fail: several are community uploads that get
    renamed, gated, or restructured, and one dead repo must not abort a multi-hour
    corpus build.

    The selected column is verified to actually contain Kabyle before it is accepted.
    Most of these datasets are bilingual, and a heuristic that picks "the widest
    string column" reliably picks the *English* side -- which would poison the corpus
    silently rather than failing.
    """
    from datasets import concatenate_datasets, load_dataset

    try:
        if src.data_files:
            # Mismatched schemas across files; load each and keep only the text column.
            parts = []
            for fname in src.data_files:
                try:
                    d = load_dataset(src.repo, data_files=fname, split="train")
                    # Keep the conversations column intact when there is one; the
                    # per-turn handler below needs the structure, not flattened text.
                    col = (src.conversations_key
                           if src.conversations_key in d.column_names
                           else _text_column(d, src.text_col))
                    parts.append(d.select_columns([col]))
                except Exception as exc:
                    log.warning("  %s/%s skipped: %s", src.repo, fname, str(exc)[:90])
            if not parts:
                return []
            # Rename to a common name only for plain-text parts, so concatenation
            # works across files whose text column happens to be named differently.
            if not src.conversations_key:
                parts = [p.rename_column(p.column_names[0], "text") for p in parts]
            ds = concatenate_datasets(parts)
        else:
            ds = load_dataset(src.repo, src.config, split=src.split)
    except Exception as exc:
        log.warning("source %-16s UNAVAILABLE: %s", src.key, str(exc)[:160])
        return []

    if limit:
        ds = ds.select(range(min(limit, len(ds))))

    # ShareGPT-style conversations: flatten to turns, dropping any that are not
    # Kabyle. The dialog corpus is NLLB-translated from French and some assistant
    # turns were never translated at all, so filtering per turn (not per row) is the
    # only thing that keeps French out of a Kabyle corpus.
    if src.conversations_key and src.conversations_key in ds.column_names:
        from .normalize import diacritic_rate

        turns, dropped = [], 0
        for convo in ds[src.conversations_key]:
            for turn in convo or []:
                val = (turn or {}).get("value")
                if not isinstance(val, str) or not val.strip():
                    continue
                if diacritic_rate(val) < MIN_KABYLE_DIACRITIC_RATE:
                    dropped += 1
                    continue
                turns.append(val)
        log.info("source %-16s %d Kabyle turns (%d non-Kabyle turns dropped)",
                 src.key, len(turns), dropped)
        return turns

    # Kabyle inside a `translation` dict column (HF Translation feature).
    if src.translation_key and src.translation_key in ds.column_names:
        key = src.pair_cols[1] if src.pair_cols else "kab"
        values = [r.get(key) for r in ds[src.translation_key]]
        rate = _looks_kabyle([v for v in values if v])
        log.info("source %-16s %d rows, %s[%r]  diacritics %.1f%%",
                 src.key, len(ds), src.translation_key, key, rate * 100)
        if rate < MIN_KABYLE_DIACRITIC_RATE:
            log.warning("source %-16s REJECTED: %r does not look like Kabyle",
                        src.key, key)
            return []
        return [v for v in values if isinstance(v, str) and v.strip()]

    try:
        col = _text_column(ds, src.text_col)
    except ValueError as exc:
        log.warning("source %-16s no usable column: %s", src.key, exc)
        return []

    values = [t for t in ds[col] if isinstance(t, str) and t.strip()]
    rate = _looks_kabyle(values)
    log.info("source %-16s %d rows, column %r  diacritics %.1f%%",
             src.key, len(values), col, rate * 100)
    if rate < MIN_KABYLE_DIACRITIC_RATE:
        log.warning("source %-16s REJECTED: column %r is not Kabyle "
                    "(set text_col explicitly in sources.py)", src.key, col)
        return []
    return values


def collect(
    sources: list[Source],
    *,
    lexicon: set[str] | None = None,
    cfg: FilterConfig | None = None,
    use_lid: bool = True,
    limit_per_source: int | None = None,
    local_files: dict[str, str] | None = None,
    dedup_threshold: float = 0.8,
) -> tuple[list[Doc], CollectReport]:
    cfg = cfg or FilterConfig()
    rep = CollectReport()
    lid = LanguageID() if use_lid else None
    docs: list[Doc] = []

    raw_by_source: dict[str, list[str]] = {}
    for src in sources:
        if src.kind is Kind.PDF:
            continue  # handled by scripts/01_decode_pdfs.py
        raw = load_source(src, limit_per_source)
        if raw:
            raw_by_source[src.key] = raw

    for key, raw in (local_files or {}).items():
        p = Path(raw)
        if p.exists():
            chunks = _chunk_local(p.read_text(encoding="utf-8"))
            raw_by_source[key] = chunks
            log.info("source %-16s %d chunks (local %s)", key, len(chunks), p.name)
        else:
            log.warning("local file missing: %s", p)

    for key, raw in raw_by_source.items():
        stats = FilterStats()
        for t in raw:
            norm, verdict = clean_document(t, cfg, lexicon)
            if verdict and lid is not None:
                verdict = lid.verdict(norm)
            stats.record(verdict, len(words(norm)) if verdict else 0)
            if verdict:
                docs.append(Doc(norm, key))
        rep.per_source[key] = {
            "seen": stats.seen,
            "kept": stats.kept,
            "words": stats.kept_words,
            "top_reject": stats.rejected.most_common(1)[0][0] if stats.rejected else "",
            "rejections": dict(stats.rejected),
        }
        log.info("source %-16s %s", key, stats.summary().splitlines()[0])

    # Dedup across the merged pool, never per source: the same Tatoeba export is
    # redistributed under several repo names and would survive per-source dedup.
    log.info("deduplicating %d documents across all sources...", len(docs))
    keep, ddrep = dedup([d.text for d in docs], threshold=dedup_threshold)
    docs = [docs[i] for i in keep]
    rep.dedup_removed = ddrep.exact_removed + ddrep.near_removed

    rep.total_docs = len(docs)
    rep.total_words = sum(d.n_words for d in docs)
    return docs, rep


def write_splits(
    docs: list[Doc],
    out_dir: str | Path,
    *,
    val_frac: float = 0.01,
    test_frac: float = 0.01,
    seed: int = 1337,
    min_eval_per_source: int = 50,
) -> dict[str, int]:
    """Write train/val/test JSONL, stratified by source.

    Stratification matters: a single pooled validation split would be dominated by
    whichever source is largest, and a per-domain perplexity breakdown is the only
    way to notice that the model is fitting Tatoeba and ignoring literary prose.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    by_source: dict[str, list[Doc]] = {}
    for d in docs:
        by_source.setdefault(d.source, []).append(d)

    splits: dict[str, list[Doc]] = {"train": [], "val": [], "test": []}
    for key, group in by_source.items():
        rng.shuffle(group)
        n = len(group)
        n_val = max(min(min_eval_per_source, n // 10), int(n * val_frac))
        n_test = max(min(min_eval_per_source, n // 10), int(n * test_frac))
        if n_val + n_test >= n:
            splits["train"].extend(group)
            continue
        splits["val"].extend(group[:n_val])
        splits["test"].extend(group[n_val:n_val + n_test])
        splits["train"].extend(group[n_val + n_test:])

    counts: dict[str, int] = {}
    for name, group in splits.items():
        rng.shuffle(group)
        path = out / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for d in group:
                f.write(json.dumps(asdict(d), ensure_ascii=False) + "\n")
        counts[name] = len(group)
        log.info("wrote %s: %d docs, %d words", path.name,
                 len(group), sum(x.n_words for x in group))
    return counts
