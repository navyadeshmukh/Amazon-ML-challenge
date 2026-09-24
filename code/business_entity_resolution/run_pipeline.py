"""End-to-end pipeline.

data -> normalise -> blocking (recall) -> features -> LightGBM (grouped 5-fold OOF)
     -> metrics (P / R / macro F0.5 / singletons) -> threshold+assignment tuning (precision)
     -> leave-one-country-out check -> test predictions -> matching_results.tsv + candidate_pairs.tsv

Examples
    python run_pipeline.py --quick                      # fast dev loop
    python run_pipeline.py                              # strong TF-IDF + string features
    python run_pipeline.py --embedder e5small           # + multilingual embeddings (recommended)
    python run_pipeline.py --embedder bgem3 --device cuda --loco
"""
import argparse
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from src.normalize import add_normalized
from src.blocking import fit_encoders, encode, generate_candidates
from src.features import build_features
from src.embeddings import load_embedder, embed_views

ROOT = Path(__file__).resolve().parents[3]          # .../Amazon-ML-challenge -> amazonML
SEED = 42


# ----------------------------------------------------------------------------- IO
def read(path):
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)


def load_split(d: Path, prefix: str):
    s1, s2, s3 = (add_normalized(read(d / f"{prefix}_source{k}.tsv")) for k in (1, 2, 3))
    s2["src"], s3["src"] = "S2", "S3"
    return s1.reset_index(drop=True), pd.concat([s2, s3], ignore_index=True)


def load_truth(path: Path):
    gt = read(path)
    return {a: {x.strip() for x in b.split(",") if x.strip()}
            for a, b in zip(gt["source1_entity_id"], gt["matched_entity_ids"])}


# ----------------------------------------------------------------------------- metrics
def f05(pred: set, true: set) -> float:
    """Official per-S1-entity F0.5 (empty/empty = 1.0, empty vs non-empty = 0.0)."""
    if not pred and not true:
        return 1.0
    if not pred or not true:
        return 0.0
    tp = len(pred & true)
    if tp == 0:
        return 0.0
    p, r = tp / len(pred), tp / len(true)
    return 1.25 * p * r / (0.25 * p + r)


def evaluate(pm, truth, ids):
    """macro F0.5 = the leaderboard metric. Also: pair-level precision/recall, singleton behaviour."""
    scores = [f05(pm.get(s, set()), truth.get(s, set())) for s in ids]
    tp = npred = ntrue = 0
    sing = sing_ok = 0
    for s in ids:
        p, t = pm.get(s, set()), truth.get(s, set())
        tp += len(p & t); npred += len(p); ntrue += len(t)
        if not t:
            sing += 1
            sing_ok += int(not p)
    ns = [sc for sc, s in zip(scores, ids) if truth.get(s)]
    return dict(macro_f05=float(np.mean(scores)),
                macro_f05_matched_only=float(np.mean(ns)) if ns else float("nan"),
                precision=tp / npred if npred else 1.0,
                recall=tp / ntrue if ntrue else 1.0,
                singletons=sing, singleton_acc=sing_ok / sing if sing else float("nan"),
                false_merged_singletons=sing - sing_ok)


def fmt(m):
    return (f"F0.5={m['macro_f05']:.4f} | P={m['precision']:.4f} R={m['recall']:.4f} | "
            f"F0.5(matched only)={m['macro_f05_matched_only']:.4f} | "
            f"singletons correct={m['singleton_acc']:.3f} ({m['false_merged_singletons']}/{m['singletons']} falsely merged)")


# ----------------------------------------------------------------------------- decoding
def decode(cand, prob, thr, exclusive):
    """exclusive=True: each S2/S3 record only goes to its highest-probability S1 (S1 is deduplicated)."""
    d = cand[["s1_id", "o_id"]].assign(p=prob)
    d = d[d["p"] >= thr]
    if exclusive and len(d):
        d = d.loc[d.groupby("o_id")["p"].idxmax()]
    out = {}
    for a, b in zip(d["s1_id"], d["o_id"]):
        out.setdefault(a, set()).add(b)
    return out


def tune(cand, prob, truth, ids, sweep_path=None):
    best, rows = (-1, None, None), []
    for excl in (False, True):
        for thr in np.arange(0.30, 0.99, 0.01):
            m = evaluate(decode(cand, prob, thr, excl), truth, ids)
            rows.append(dict(exclusive=excl, threshold=round(float(thr), 2), **m))
            if m["macro_f05"] > best[0]:
                best = (m["macro_f05"], float(thr), excl)
    if sweep_path is not None:
        pd.DataFrame(rows).to_csv(sweep_path, sep="\t", index=False)
    return best


# ----------------------------------------------------------------------------- pairs
def make_pairs(s1, oth, k, kd, embedder, n_jobs):
    enc = fit_encoders([s1, oth])
    m1, mo = encode(enc, s1), encode(enc, oth)
    e1 = eo = None
    if embedder is not None:
        t = time.time()
        e1, eo = embed_views(embedder, s1), embed_views(embedder, oth)
        print(f"      embeddings done ({time.time() - t:.0f}s)")
    i, j = generate_candidates(m1, mo, s1, oth, k=k, d1=e1, do=eo, kd=kd)
    f = build_features(i, j, s1, oth, m1, mo, e1, eo, n_jobs=n_jobs)
    f["s1_id"] = s1["entity_id"].values[i]
    f["o_id"] = oth["entity_id"].values[j]
    return f


def error_report(pm, truth, tr1, tro, cand, prob, path: Path, n=300):
    info = {}
    for df in (tr1, tro):
        for e, nm, ad, c in zip(df["entity_id"], df["business_name"], df["business_address"], df["country"]):
            info[e] = (nm, ad, c)
    pmap = dict(zip(zip(cand["s1_id"], cand["o_id"]), prob))
    rows = []
    for s in set(truth) | set(pm):
        pred, true = pm.get(s, set()), truth.get(s, set())
        for o in pred - true:
            rows.append(("FP", s, o, pmap.get((s, o), np.nan)))
        for o in true - pred:
            rows.append(("FN" if (s, o) in pmap else "FN_not_in_candidates", s, o, pmap.get((s, o), np.nan)))
    df = pd.DataFrame(rows, columns=["type", "s1_id", "o_id", "prob"])
    if len(df):
        df = df.sort_values("prob", ascending=False).groupby("type").head(n)
        for col, idx, pos in (("s1_name", "s1_id", 0), ("s1_address", "s1_id", 1),
                              ("o_name", "o_id", 0), ("o_address", "o_id", 1), ("country", "s1_id", 2)):
            df[col] = df[idx].map(lambda e, p=pos: info.get(e, ("", "", ""))[p])
    df.to_csv(path, sep="\t", index=False)
    return df["type"].value_counts().to_dict() if len(df) else {}


def write_outputs(cand, matches, s1_ids, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    cmap = cand.groupby("s1_id")["o_id"].apply(lambda s: ",".join(sorted(set(s)))).to_dict()
    pd.DataFrame({"source1_entity_id": s1_ids,
                  "candidate_entity_ids": [cmap.get(s, "") for s in s1_ids]}
                 ).to_csv(out_dir / "candidate_pairs.tsv", sep="\t", index=False)
    pd.DataFrame({"source1_entity_id": s1_ids,
                  "matched_entity_ids": [",".join(sorted(matches.get(s, set()))) for s in s1_ids]}
                 ).to_csv(out_dir / "matching_results.tsv", sep="\t", index=False)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=str(ROOT / "student_resource" / "dataset"))
    ap.add_argument("--out-root", default=str(ROOT / "Amazon-ML-challenge" / "output"))
    ap.add_argument("--report-dir", default=str(ROOT / "Amazon-ML-challenge" / "reports"))
    ap.add_argument("--k", type=int, default=25, help="top-K per sparse blocking view per source")
    ap.add_argument("--kd", type=int, default=15, help="top-K per dense blocking view per source")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--embedder", default="none", help="none | e5small | e5base | bgem3 | minilm | hash(test only)")
    ap.add_argument("--device", default=None, help="cpu | cuda (default: auto)")
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--loco", action="store_true", help="leave-one-country-out generalisation check")
    ap.add_argument("--quick", action="store_true", help="k=15, folds=3 (fast dev loop)")
    ap.add_argument("--min-thr-safety", type=float, default=0.0,
                    help="added to the tuned threshold at test time (extra precision for unseen France)")
    a = ap.parse_args()
    if a.quick:
        a.k, a.folds = 15, 3
    data, out, rep = Path(a.data_root), Path(a.out_root), Path(a.report_dir)
    rep.mkdir(parents=True, exist_ok=True)
    metrics, t0 = {"args": vars(a)}, time.time()
    embedder = load_embedder(a.embedder, a.device)

    # ---------------- TRAIN
    print("[1/7] load + normalise train")
    tr1, tro = load_split(data / "train", "train")
    truth = load_truth(data / "train" / "train_ground_truth.tsv")
    print(f"      S1={len(tr1)} S2+S3={len(tro)} countries={sorted(tr1.country_n.unique())}")

    print("[2/7] blocking + features (train)")
    tr = make_pairs(tr1, tro, a.k, a.kd, embedder, a.n_jobs)
    tr["y"] = [int(o in truth.get(s, ())) for s, o in zip(tr["s1_id"], tr["o_id"])]
    n_true = sum(len(v) for v in truth.values())
    cnt = pd.Series([o for v in truth.values() for o in v]).value_counts()
    rec = tr["y"].sum() / max(1, n_true)
    print(f"      truth links={n_true}, singleton S1={sum(1 for v in truth.values() if not v)}, "
          f"S2/S3 records linked to >1 S1: {(cnt > 1).sum()} (0 => exclusive assignment safe)")
    print(f"      BLOCKING: candidates={len(tr):,} | recall ceiling={rec:.4f} | avg cands/S1={len(tr)/len(tr1):.1f} | "
          f"reduction ratio={1 - len(tr)/max(1, len(tr1)*len(tro)):.5f} | pos rate={tr['y'].mean():.4f} ({time.time()-t0:.0f}s)")
    metrics.update(train_s1=len(tr1), train_other=len(tro), train_candidates=len(tr),
                   blocking_recall=float(rec), reduction_ratio=float(1 - len(tr) / max(1, len(tr1) * len(tro))))

    feat_cols = [c for c in tr.columns if c not in ("i", "j", "s1_id", "o_id", "y")]
    X, y, groups = tr[feat_cols].values, tr["y"].values, tr["s1_id"].values
    params = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_child_samples=20,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                  n_estimators=1500, random_state=SEED, verbose=-1, n_jobs=-1)

    print(f"[3/7] {a.folds}-fold LightGBM grouped by S1 entity")
    oof, models = np.zeros(len(tr)), []
    for f, (ta, va) in enumerate(GroupKFold(a.folds).split(X, y, groups)):
        m = lgb.LGBMClassifier(**params)
        m.fit(X[ta], y[ta], eval_set=[(X[va], y[va])], eval_metric="binary_logloss",
              callbacks=[lgb.early_stopping(100, verbose=False)])
        oof[va] = m.predict_proba(X[va])[:, 1]
        models.append(m)
        print(f"      fold {f}: best_iter={m.best_iteration_}")

    print("[4/7] validation on OOF (leaderboard metric = macro F0.5, singletons included)")
    ids = tr1["entity_id"].tolist()
    naive = evaluate(decode(tr, oof, 0.5, False), truth, ids)
    print("      naive thr=0.50        :", fmt(naive))
    score, thr, excl = tune(tr, oof, truth, ids, rep / "threshold_sweep.tsv")
    final = evaluate(decode(tr, oof, thr, excl), truth, ids)
    print(f"      TUNED thr={thr:.2f} excl={excl}:", fmt(final))
    per_c = {}
    pm = decode(tr, oof, thr, excl)
    for c in sorted(tr1.country_n.unique()):
        cid = tr1.loc[tr1.country_n == c, "entity_id"].tolist()
        per_c[c] = evaluate(pm, truth, cid)
        print(f"        {c:>8}: {fmt(per_c[c])}  (n={len(cid)})")
    metrics.update(naive_0p5=naive, tuned=final, threshold=thr, exclusive=excl, per_country=per_c,
                   error_counts=error_report(pm, truth, tr1, tro, tr, oof, rep / "oof_errors.tsv"))
    imp = pd.Series(np.mean([m.feature_importances_ for m in models], axis=0), index=feat_cols)
    print("      top features:", ", ".join(imp.sort_values(ascending=False).head(10).index))
    print("      errors:", metrics["error_counts"], "->", rep / "oof_errors.tsv", "| sweep ->", rep / "threshold_sweep.tsv")

    if a.loco and tr1.country_n.nunique() > 1:
        print("[5/7] leave-one-country-out (simulates unseen France): train on other countries, test on held-out")
        cty = tr1["country_n"].values[tr["i"].values]
        loco = {}
        for c in sorted(tr1.country_n.unique()):
            trn, tst = cty != c, cty == c
            p2 = dict(params, n_estimators=400)
            m = lgb.LGBMClassifier(**p2).fit(X[trn], y[trn])
            pr = m.predict_proba(X[tst])[:, 1]
            cid = tr1.loc[tr1.country_n == c, "entity_id"].tolist()
            sub = tr.loc[tst, ["s1_id", "o_id"]].reset_index(drop=True)
            r_fixed = evaluate(decode(sub, pr, thr, excl), truth, cid)
            print(f"        held-out {c:>8} @ global thr: {fmt(r_fixed)}")
            loco[c] = r_fixed
        metrics["loco"] = loco
    else:
        print("[5/7] loco skipped (use --loco)")

    # ---------------- TEST
    print("[6/7] test: normalise, block, featurise")
    te1, teo = load_split(data / "test", "test")
    print(f"      S1={len(te1)} S2+S3={len(teo)} countries={sorted(te1.country_n.unique())}")
    te = make_pairs(te1, teo, a.k, a.kd, embedder, a.n_jobs)
    print(f"      candidates={len(te):,} avg cands/S1={len(te)/len(te1):.1f}")

    print("[7/7] predict + write")
    p = np.mean([m.predict_proba(te[feat_cols].values)[:, 1] for m in models], axis=0)
    matches = decode(te, p, min(0.99, thr + a.min_thr_safety), excl)
    write_outputs(te, matches, te1["entity_id"].tolist(), out)
    n_match = sum(1 for v in matches.values() if v)
    metrics.update(test_s1=len(te1), test_candidates=len(te), test_s1_with_match=n_match,
                   test_countries=sorted(te1.country_n.unique()))
    (rep / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    print(f"      S1 with >=1 match: {n_match}/{len(te1)} -> {out}")
    print(f"done in {(time.time()-t0)/60:.1f} min | next: python utils/validate_submission.py --matching "
          f"{out}/matching_results.tsv --candidate {out}/candidate_pairs.tsv --test-dir {data}/test")


if __name__ == "__main__":
    main()
