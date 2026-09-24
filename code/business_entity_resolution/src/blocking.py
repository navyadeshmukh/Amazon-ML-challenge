"""Stage 2 - blocking / candidate generation (maximise RECALL).

Union of top-K neighbours from several views, searched within the same country label
(open set) and separately for Source 2 and Source 3:
  sparse TF-IDF views: name char n-gram, name word, name+address char n-gram
  optional dense view : multilingual sentence embedding of name+address
"""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

VIEWS = {
    "name_char": ("name_n", dict(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True, min_df=1)),
    "name_word": ("name_n", dict(analyzer="word", ngram_range=(1, 2), sublinear_tf=True, min_df=1,
                                 token_pattern=r"(?u)\b\w+\b")),
    "full_char": ("full_n", dict(analyzer="char_wb", ngram_range=(3, 4), sublinear_tf=True, min_df=1)),
    "addr_char": ("addr_core", dict(analyzer="char_wb", ngram_range=(3, 4), sublinear_tf=True, min_df=1)),
    "addr_word": ("addr_core", dict(analyzer="word", sublinear_tf=True, min_df=1,
                                    token_pattern=r"(?u)\b\w+\b")),
}
BLOCK_VIEWS = ["name_char", "name_word", "full_char"]
DENSE_BLOCK_VIEWS = ["full", "name"]


def fit_encoders(dfs):
    """Fit TF-IDF on all records (unsupervised, no labels used)."""
    enc = {}
    for name, (col, kw) in VIEWS.items():
        v = TfidfVectorizer(dtype=np.float32, **kw)
        v.fit(np.concatenate([d[col].values for d in dfs]))
        enc[name] = v
    return enc


def encode(enc, df):
    return {name: enc[name].transform(df[VIEWS[name][0]].values) for name in VIEWS}


def _topk_sparse(A, B, ia, ib, k, chunk=1500):
    out_i, out_j = [], []
    BT = B[ib].T.tocsr()
    for st in range(0, len(ia), chunk):
        rows = ia[st:st + chunk]
        S = (A[rows] @ BT).tocsr()
        for r in range(S.shape[0]):
            lo, hi = S.indptr[r], S.indptr[r + 1]
            if hi == lo:
                continue
            d, ix = S.data[lo:hi], S.indices[lo:hi]
            if len(d) > k:
                ix = ix[np.argpartition(-d, k)[:k]]
            out_i.append(np.full(len(ix), rows[r]))
            out_j.append(ib[ix])
    return out_i, out_j


def _topk_dense(A, B, ia, ib, k, chunk=512):
    out_i, out_j = [], []
    Bs = B[ib]
    kk = min(k, len(ib))
    for st in range(0, len(ia), chunk):
        rows = ia[st:st + chunk]
        S = A[rows] @ Bs.T
        idx = np.argpartition(-S, kk - 1, axis=1)[:, :kk]
        out_i.append(np.repeat(rows, kk))
        out_j.append(ib[idx].ravel())
    return out_i, out_j


def generate_candidates(m1, mo, s1, oth, k=25, d1=None, do=None, kd=15):
    """Return int arrays (i, j): S1 row index, `oth` row index."""
    n_oth = len(oth)
    c1, co, src = s1["country_n"].values, oth["country_n"].values, oth["src"].values
    keys = set()

    def run(kind, view, A, B, kk):
        fn = _topk_sparse if kind == "sparse" else _topk_dense
        for sname in ("S2", "S3"):
            idx_src = np.where(src == sname)[0]
            if len(idx_src) == 0:
                continue
            for c in np.unique(c1):
                ia = np.where(c1 == c)[0]
                ib = idx_src[co[idx_src] == c]
                if len(ib) == 0:                       # unseen / mislabeled country -> search all
                    ib = idx_src
                oi, oj = fn(A, B, ia, ib, kk)
                if oi:
                    i, j = np.concatenate(oi), np.concatenate(oj)
                    keys.update((i.astype(np.int64) * n_oth + j).tolist())

    for view in BLOCK_VIEWS:
        run("sparse", view, m1[view], mo[view], k)
    if d1 is not None:
        for view in DENSE_BLOCK_VIEWS:
            run("dense", view, d1[view], do[view], kd)
    arr = np.fromiter(keys, dtype=np.int64, count=len(keys))
    arr.sort()
    return arr // n_oth, arr % n_oth
