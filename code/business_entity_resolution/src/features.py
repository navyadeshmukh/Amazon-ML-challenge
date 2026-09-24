"""Stage 3 - pair features (name, address, structural, embedding, competition)."""
import re
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler, Levenshtein

NUM_RE = re.compile(r"\d+")
STR_COLS = ["n_lev", "n_jw", "n_tsort", "n_tset", "n_partial", "n_compact_lev", "n_jac", "n_contain",
            "n_first_eq", "n_acronym", "n_lendiff", "a_lev", "a_tset", "a_partial", "a_jac", "a_contain",
            "a_full_lev", "a_num_jac", "a_num_any", "a_tail_jac", "pin_match", "pin_prefix", "a_missing"]


def _rowcos(A, B, i, j, chunk=200_000):
    out = np.empty(len(i), np.float32)
    for st in range(0, len(i), chunk):
        sl = slice(st, st + chunk)
        out[sl] = np.asarray(A[i[sl]].multiply(B[j[sl]]).sum(axis=1)).ravel()
    return out


def _rowdot(A, B, i, j, chunk=200_000):
    out = np.empty(len(i), np.float32)
    for st in range(0, len(i), chunk):
        sl = slice(st, st + chunk)
        out[sl] = (A[i[sl]] * B[j[sl]]).sum(axis=1)
    return out


def _jacc(a, b):
    return len(a & b) / len(a | b) if (a or b) else 0.0


def _acronym(a, b):
    ta, tb = a.split(), b.split()
    if len(ta) >= 2 and "".join(t[0] for t in ta) == b.replace(" ", ""):
        return 1
    if len(tb) >= 2 and "".join(t[0] for t in tb) == a.replace(" ", ""):
        return 1
    return 0


def _string_feats(n1, n2, nc1, nc2, a1, a2, af1, af2, p1, p2):
    rows = []
    for x, y, xc, yc, ax, ay, afx, afy, px, py in zip(n1, n2, nc1, nc2, a1, a2, af1, af2, p1, p2):
        tx, ty = set(x.split()), set(y.split())
        atx, aty = set(ax.split()), set(ay.split())
        nx, ny = set(NUM_RE.findall(ax)) - {px}, set(NUM_RE.findall(ay)) - {py}
        wx, wy = [t for t in ax.split() if not t.isdigit()], [t for t in ay.split() if not t.isdigit()]
        rows.append((
            Levenshtein.normalized_similarity(x, y), JaroWinkler.normalized_similarity(x, y),
            fuzz.token_sort_ratio(x, y) / 100, fuzz.token_set_ratio(x, y) / 100,
            fuzz.partial_ratio(x, y) / 100 if x and y else 0.0,
            Levenshtein.normalized_similarity(xc, yc), _jacc(tx, ty),
            len(tx & ty) / max(1, min(len(tx), len(ty))),
            int(bool(x) and x.split()[:1] == y.split()[:1]), _acronym(x, y),
            abs(len(x) - len(y)) / max(1, len(x), len(y)),
            Levenshtein.normalized_similarity(ax, ay), fuzz.token_set_ratio(ax, ay) / 100,
            fuzz.partial_ratio(ax, ay) / 100 if ax and ay else 0.0, _jacc(atx, aty),
            len(atx & aty) / max(1, min(len(atx), len(aty))),
            Levenshtein.normalized_similarity(afx, afy),
            _jacc(nx, ny) if (nx or ny) else -1.0, int(bool(nx & ny)),
            _jacc(set(wx[-2:]), set(wy[-2:])) if wx and wy else -1.0,
            int(px == py) if (px and py) else -1,
            int(bool(px and py and px[:3] == py[:3])), int(not ax or not ay)))
    return np.array(rows, dtype=np.float32).reshape(-1, len(STR_COLS))


def build_features(i, j, s1, oth, m1, mo, e1=None, eo=None, n_jobs=-1):
    f = pd.DataFrame({"i": i, "j": j})
    f["is_s3"] = (oth["src"].values[j] == "S3").astype(np.int8)
    for v in m1:
        f[f"cos_{v}"] = _rowcos(m1[v], mo[v], i, j)
    if e1 is not None:
        for v in e1:
            f[f"emb_{v}"] = _rowdot(e1[v], eo[v], i, j)

    arrs = [s1["name_n"].values[i], oth["name_n"].values[j],
            s1["name_compact"].values[i], oth["name_compact"].values[j],
            s1["addr_core"].values[i], oth["addr_core"].values[j],
            s1["addr_n"].values[i], oth["addr_n"].values[j],
            s1["pin"].values[i], oth["pin"].values[j]]
    n_chunks = max(1, min(64, len(i) // 20000 + 1))
    chunks = np.array_split(np.arange(len(i)), n_chunks)
    res = Parallel(n_jobs=n_jobs)(delayed(_string_feats)(*[a[c] for a in arrs]) for c in chunks)
    f[STR_COLS] = np.vstack(res) if res else np.empty((0, len(STR_COLS)), np.float32)
    f["country_match"] = (s1["country_n"].values[i] == oth["country_n"].values[j]).astype(np.int8)

    # ---- combined score + competition features (precision lever)
    f["combo"] = 0.65 * f["cos_name_char"] + 0.35 * f["cos_addr_char"]
    g = f.groupby("i")["combo"]
    f["rank_i"] = g.rank(ascending=False, method="min")
    f["gap_i"] = f["combo"] - g.transform("max")
    f["n_cands_i"] = g.transform("size")
    gj = f.groupby("j")["combo"]
    f["rank_j"] = gj.rank(ascending=False, method="min")
    f["gap_j"] = f["combo"] - gj.transform("max")
    f["n_cands_j"] = gj.transform("size")
    extra = ["n_jw", "a_lev"] + (["emb_full", "emb_name"] if e1 is not None else [])
    for c in extra:
        f[f"{c}_gap_i"] = f[c] - f.groupby("i")[c].transform("max")
        f[f"{c}_gap_j"] = f[c] - f.groupby("j")[c].transform("max")
    return f
