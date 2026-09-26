"""Stage 2 - blocking / candidate generation (maximise RECALL).

Memory-safe streaming architecture:
  - One TF-IDF view at a time (fit → search → free)
  - Other-side sub-chunked into 200k-row batches (never holds full 10M-row matrix)
  - Peak blocking memory: ~1 GB regardless of dataset size
"""
import gc
import numpy as np
import psutil
from sklearn.feature_extraction.text import TfidfVectorizer

VIEWS = {
    "name_char": ("name_n", dict(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True, min_df=3, max_features=80000)),
    "name_word": ("name_n", dict(analyzer="word", ngram_range=(1, 2), sublinear_tf=True, min_df=3, max_features=80000,
                                 token_pattern=r"(?u)\b\w+\b")),
    "full_char": ("full_n", dict(analyzer="char_wb", ngram_range=(3, 4), sublinear_tf=True, min_df=4, max_features=80000)),
}
BLOCK_VIEWS = ["name_char", "name_word", "full_char"]

# Max rows to transform at once on the Other side.  200k × ~50 nnz × 8 bytes ≈ 80 MB.
B_CHUNK = 200_000
# Max S1 query rows per matmul.  1000 × 200k-col result is manageable.
Q_CHUNK = 1000


def _search_chunk(A_rows, row_ids, B_chunk, chunk_ids, k):
    """Compute top-k for a batch of S1 queries against a batch of Other records."""
    BT = B_chunk.T.tocsc()
    S = (A_rows @ BT).tocsr()
    pairs = []
    for r in range(S.shape[0]):
        lo, hi = S.indptr[r], S.indptr[r + 1]
        if hi == lo:
            continue
        d, ix = S.data[lo:hi], S.indices[lo:hi]
        if len(d) > k:
            top = np.argpartition(-d, k)[:k]
            ix = ix[top]
        for j in ix:
            pairs.append((int(row_ids[r]), int(chunk_ids[j])))
    return pairs


def generate_candidates_streaming(s1, oth, k=25):
    """Generate candidates one view at a time, sub-chunking the Other side.

    Memory budget breakdown (per view):
      A (S1 TF-IDF, 2.2M rows):  ~500 MB  (held for full view)
      B_chunk (200k rows):        ~80 MB   (created and freed per sub-chunk)
      BT (transpose):             ~80 MB   (created and freed per sub-chunk)
      S (query result):           ~20 MB   (created and freed per query chunk)
      Total peak:                 ~700 MB
    """
    n_oth = len(oth)
    c1 = s1["country_n"].values
    co = oth["country_n"].values
    src = oth["src"].values
    keys = set()

    # Pre-extract text columns as numpy arrays to avoid repeated DataFrame indexing
    oth_texts = {}
    for view_name in BLOCK_VIEWS:
        col = VIEWS[view_name][0]
        if col not in oth_texts:
            oth_texts[col] = oth[col].values
    s1_texts = {}
    for view_name in BLOCK_VIEWS:
        col = VIEWS[view_name][0]
        if col not in s1_texts:
            s1_texts[col] = s1[col].values

    for view_name in BLOCK_VIEWS:
        col, kw = VIEWS[view_name]
        avail = psutil.virtual_memory().available / (1024**3)
        print(f"      [View: {view_name}] Fitting TF-IDF (RAM: {avail:.1f} GB free)...", flush=True)

        v = TfidfVectorizer(dtype=np.float32, **kw)
        n_fit = min(len(oth), 400_000)
        v.fit(oth_texts[col][:n_fit])
        vocab_size = len(v.vocabulary_)
        print(f"      [View: {view_name}] Vocabulary: {vocab_size:,} terms", flush=True)

        print(f"      [View: {view_name}] Transforming S1 ({len(s1):,} rows)...", flush=True)
        A = v.transform(s1_texts[col])
        avail = psutil.virtual_memory().available / (1024**3)
        print(f"      [View: {view_name}] S1 matrix: {A.shape}, nnz={A.nnz:,} (RAM: {avail:.1f} GB free)", flush=True)

        for sname in ("S2", "S3"):
            idx_src = np.where(src == sname)[0]
            if len(idx_src) == 0:
                continue
            for c in np.unique(c1):
                ia = np.where(c1 == c)[0]
                ib_full = idx_src[co[idx_src] == c]
                if len(ib_full) == 0:
                    ib_full = idx_src          # fallback: unseen country
                if len(ia) == 0 or len(ib_full) == 0:
                    continue

                n_b_chunks = (len(ib_full) + B_CHUNK - 1) // B_CHUNK
                before = len(keys)
                print(f"         {sname}-{c}: S1={len(ia):,} x Other={len(ib_full):,} "
                      f"({n_b_chunks} sub-chunks of <={B_CHUNK:,})...", flush=True)

                # Sub-chunk the Other side: transform 200k rows, search, free
                for b_start in range(0, len(ib_full), B_CHUNK):
                    ib_sub = ib_full[b_start:b_start + B_CHUNK]
                    B_sub = v.transform(oth_texts[col][ib_sub])

                    # Query S1 in chunks of Q_CHUNK against this B sub-chunk
                    for q_start in range(0, len(ia), Q_CHUNK):
                        q_rows = ia[q_start:q_start + Q_CHUNK]
                        pairs = _search_chunk(A[q_rows], q_rows, B_sub, ib_sub, k)
                        for qi, qj in pairs:
                            keys.add(np.int64(qi) * n_oth + np.int64(qj))

                    del B_sub
                    gc.collect()

                added = len(keys) - before
                avail = psutil.virtual_memory().available / (1024**3)
                print(f"         {sname}-{c}: +{added:,} candidates (RAM: {avail:.1f} GB free)", flush=True)

        del A, v
        gc.collect()
        avail = psutil.virtual_memory().available / (1024**3)
        print(f"      [View: {view_name}] Done! Total candidates: {len(keys):,} (RAM: {avail:.1f} GB free)", flush=True)

    # Clean up pre-extracted text arrays
    del oth_texts, s1_texts
    gc.collect()

    arr = np.fromiter(keys, dtype=np.int64, count=len(keys))
    arr.sort()
    return arr // n_oth, arr % n_oth
