"""Exact and near-duplicate removal.

Deduplication has to happen *after* merging every source, not per-source. The Kabyle
datasets overlap heavily -- Tatoeba is redistributed under at least four different
repo names, and the v0 project's own two files already shared 2,084 identical lines
between `kabyle_wiki_corpus_cleaned.txt` and `kab_Latn.txt`.

Duplicates are not harmless at this data scale. With ~35M tokens and a 1.7B model,
anything repeated is memorized, which is precisely the failure the v0 run exhibited
(train 1.33 vs val 6.21).

Two passes:
  * exact, on a normalized hash -- cheap, catches redistribution
  * near, via MinHash + LSH banding -- catches boilerplate and edited reposts

MinHash is implemented here rather than pulled from `datasketch` so the pipeline has
one less dependency to install on Kaggle, and so the banding parameters are visible
and tunable.
"""

from __future__ import annotations

import hashlib
import logging
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass

log = logging.getLogger(__name__)

_RE_NONWORD = re.compile(r"[\W_]+", re.UNICODE)

#: Mersenne prime field for the MinHash permutations. See `MinHasher` for why this
#: is 31 bits and not the more common 61.
_MERSENNE = (1 << 31) - 1


def canonical_key(text: str) -> str:
    """Aggressively flattened form for exact-duplicate detection.

    Case, punctuation, whitespace and diacritics are all discarded: two copies of a
    sentence that differ only in whether emphatics were marked are still one sentence.
    """
    flat = unicodedata.normalize("NFKD", text.lower())
    flat = "".join(c for c in flat if not unicodedata.combining(c))
    return _RE_NONWORD.sub(" ", flat).strip()


def exact_hash(text: str) -> str:
    return hashlib.blake2b(canonical_key(text).encode("utf-8"), digest_size=16).hexdigest()


def dedup_exact(docs: list[str]) -> tuple[list[int], int]:
    """Indices of the first occurrence of each distinct document."""
    seen: set[str] = set()
    keep: list[int] = []
    for i, d in enumerate(docs):
        h = exact_hash(d)
        if h not in seen:
            seen.add(h)
            keep.append(i)
    return keep, len(docs) - len(keep)


# --- MinHash near-duplicate detection ----------------------------------------

def shingles(text: str, k: int = 5) -> set[int]:
    """Hashed word k-shingles. Word-level (not character-level) because Kabyle
    morphology makes character shingles collide across unrelated words."""
    toks = canonical_key(text).split()
    if len(toks) < k:
        return {hash(" ".join(toks))} if toks else set()
    return {hash(" ".join(toks[i:i + k])) for i in range(len(toks) - k + 1)}


class MinHasher:
    """Permutation-based MinHash, vectorized over permutations with numpy.

    The pure-Python double loop costs ~2.6 ms per document, which is 90 minutes on a
    2M-document corpus. Vectorizing makes it roughly 30x faster.

    The field is 2^31-1 rather than the more usual 2^61-1 for a concrete reason:
    numpy has no uint128, so with a 61-bit modulus the product `a * s` reaches 2^122
    and silently wraps in uint64, producing a hash family that is no longer
    universal. At 31 bits both factors are < 2^31, so `a * s < 2^62` provably fits.
    The narrower field costs nothing here -- Jaccard is estimated from *agreement
    counts* across 128 slots, and accidental collisions at 2^31 among the few
    thousand shingles of a document are negligible.
    """

    def __init__(self, num_perm: int = 128, seed: int = 1337):
        import numpy as np

        rng = np.random.default_rng(seed)
        self.num_perm = num_perm
        self.a = rng.integers(1, _MERSENNE, size=num_perm, dtype=np.uint64)
        self.b = rng.integers(0, _MERSENNE, size=num_perm, dtype=np.uint64)

    def signature(self, text: str, k: int = 5):
        import numpy as np

        sh = shingles(text, k)
        if not sh:
            return np.zeros(self.num_perm, dtype=np.uint64)
        s = np.fromiter((x & _MERSENNE for x in sh), dtype=np.uint64, count=len(sh))
        M = np.uint64(_MERSENNE)
        # (num_perm, n_shingles) -> min over shingles
        vals = (self.a[:, None] * s[None, :] + self.b[:, None]) % M
        return vals.min(axis=1)


@dataclass
class DedupReport:
    total: int = 0
    exact_removed: int = 0
    near_removed: int = 0

    @property
    def kept(self) -> int:
        return self.total - self.exact_removed - self.near_removed

    def summary(self) -> str:
        if not self.total:
            return "nothing to deduplicate"
        pct = 100 * self.kept / self.total
        return (
            f"dedup: {self.total:,} -> {self.kept:,} ({pct:.1f}% kept)  "
            f"exact -{self.exact_removed:,}  near -{self.near_removed:,}"
        )


def dedup(
    docs: list[str],
    *,
    threshold: float = 0.8,
    num_perm: int = 128,
    bands: int = 16,
    shingle_k: int = 5,
    progress: bool = True,
) -> tuple[list[int], DedupReport]:
    """Return indices to keep, after exact then near-duplicate removal.

    `bands` x rows must equal `num_perm`. More bands means higher recall and more
    false candidates; 16 bands of 8 rows targets a Jaccard threshold near 0.8.
    """
    if num_perm % bands:
        raise ValueError(f"num_perm={num_perm} not divisible by bands={bands}")

    rep = DedupReport(total=len(docs))
    keep_idx, rep.exact_removed = dedup_exact(docs)

    hasher = MinHasher(num_perm=num_perm)
    rows = num_perm // bands

    it = keep_idx
    if progress:
        try:
            from tqdm import tqdm

            it = tqdm(keep_idx, desc="minhash", unit="doc")
        except ImportError:
            pass

    sigs: dict[int, object] = {}
    buckets: list[defaultdict[bytes, list[int]]] = [defaultdict(list) for _ in range(bands)]
    for i in it:
        sig = hasher.signature(docs[i], shingle_k)
        sigs[i] = sig
        raw = sig.tobytes()
        stride = rows * 8
        for b in range(bands):
            buckets[b][raw[b * stride:(b + 1) * stride]].append(i)

    # Union-find over candidate pairs that share any band.
    parent: dict[int, int] = {i: i for i in keep_idx}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[max(rx, ry)] = min(rx, ry)

    for band in buckets:
        for group in band.values():
            if len(group) < 2:
                continue
            head = group[0]
            for other in group[1:]:
                if _jaccard(sigs[head], sigs[other]) >= threshold:
                    union(head, other)

    survivors = sorted({find(i) for i in keep_idx})
    rep.near_removed = len(keep_idx) - len(survivors)
    log.info(rep.summary())
    return survivors, rep


def _jaccard(a, b) -> float:
    """Estimated Jaccard similarity: the fraction of agreeing MinHash slots."""
    return float((a == b).mean())
