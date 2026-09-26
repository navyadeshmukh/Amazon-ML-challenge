"""Modal Cloud Runner for Amazon ML Challenge 2026.

Team Slytherine — Entity Resolution Pipeline on Cloud NVIDIA A100 GPU

Usage:
    # 1. One-time data sync to Modal Volume:
    modal run modal_runner.py::sync_dataset

    # 2. Quick sanity run on A100 (~1-2 minutes, ~$0.07):
    modal run modal_runner.py --quick

    # 3. Full baseline run on A100 (~15-20 minutes, ~$0.55):
    modal run modal_runner.py --loco

    # 4. Full BGE-M3 Multilingual Embedding run on A100 (~25-35 minutes, ~$0.90):
    modal run modal_runner.py --embedder bgem3 --loco

    # 5. Budget A10G run to stretch credits further:
    modal run modal_runner.py --gpu a10g --embedder e5small --loco
"""

import json
import os
from pathlib import Path
import subprocess
import sys
import time
import modal

# -----------------------------------------------------------------------------
# Configuration & Paths
# -----------------------------------------------------------------------------
APP_NAME = "amazon-ml-pipeline"
VOLUME_NAME = "amazon-ml-data"

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

CHALLENGE_DIR = Path(__file__).resolve().parent
CODE_DIR = CHALLENGE_DIR / "code" / "business_entity_resolution"
ROOT_DIR = CHALLENGE_DIR.parent
DATA_DIR = ROOT_DIR / "student_resource" / "dataset"
OUT_DIR = CHALLENGE_DIR / "output"
REP_DIR = CHALLENGE_DIR / "reports"

# -----------------------------------------------------------------------------
# Remote Container Image (Debian Slim + PyTorch CUDA + SentenceTransformers)
# -----------------------------------------------------------------------------
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch>=2.3.0",
        "sentence-transformers>=5.0",
        "lightgbm==4.7.0",
        "rapidfuzz==3.14.6",
        "scikit-learn==1.8.0",
        "pandas==3.0.2",
        "numpy==2.4.4",
        "scipy==1.17.1",
        "joblib>=1.4",
        "tqdm",
    )
    .add_local_dir(
        CODE_DIR,
        remote_path="/root/Amazon-ML-challenge/code/business_entity_resolution",
    )
)


# -----------------------------------------------------------------------------
# Pipeline Execution Engine (Internal)
# -----------------------------------------------------------------------------
def _execute_pipeline_core(
    quick: bool,
    embedder: str,
    loco: bool,
    sample_s1: int,
    k: int,
    kd: int,
    folds: int,
    min_thr_safety: float,
):
    import json
    import os
    from pathlib import Path
    import subprocess
    import sys
    import time
    import torch

    print("=" * 72)
    print("☁️ RUNNING AMAZON ML PIPELINE IN MODAL CLOUD")
    print("=" * 72)
    print(f"Python: {sys.version}")
    print(f"PyTorch: {torch.__version__} | CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Device: {torch.cuda.get_device_name(0)}")
        props = torch.cuda.get_device_properties(0)
        print(f"Total VRAM: {props.total_memory / (1024**3):.2f} GB")
    print(f"CPU count: {os.cpu_count()}")
    print("=" * 72)

    vol_data = Path("/vol/dataset")
    train_s1 = vol_data / "train" / "train_source1.tsv"
    if not train_s1.exists():
        raise RuntimeError(
            f"Dataset not found at {train_s1}!\n"
            "Please run the dataset sync command first:\n"
            "    modal run modal_runner.py::sync_dataset"
        )

    out_dir = Path("/vol/output")
    rep_dir = Path("/vol/reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    rep_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "/root/Amazon-ML-challenge/code/business_entity_resolution/run_pipeline.py",
        "--data-root", str(vol_data),
        "--out-root", str(out_dir),
        "--report-dir", str(rep_dir),
        "--embedder", embedder,
        "--device", "cuda" if torch.cuda.is_available() else "cpu",
        "--k", str(k),
        "--kd", str(kd),
        "--folds", str(folds),
        "--sample-s1", str(sample_s1),
        "--min-thr-safety", str(min_thr_safety),
    ]
    if quick:
        cmd.append("--quick")
    if loco:
        cmd.append("--loco")

    print(f"\n🚀 Executing command: {' '.join(cmd)}\n")
    start_time = time.time()

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
    )

    for line in iter(proc.stdout.readline, ""):
        print(line, end="", flush=True)

    proc.stdout.close()
    return_code = proc.wait()
    if return_code != 0:
        raise RuntimeError(f"Pipeline exited with error code {return_code}")

    elapsed = time.time() - start_time
    print(f"\n✅ Remote execution completed in {elapsed/60:.2f} minutes!")

    # Commit written files to Modal persistent storage
    volume.commit()
    print("💾 Volume changes committed successfully.")

    # Return summary
    summary = {
        "elapsed_seconds": elapsed,
        "files": {},
    }
    for p in list(out_dir.glob("*")) + list(rep_dir.glob("*")):
        if p.is_file():
            summary["files"][str(p.relative_to(Path("/vol")))] = p.stat().st_size

    metrics_file = rep_dir / "metrics.json"
    if metrics_file.exists():
        try:
            summary["metrics"] = json.loads(metrics_file.read_text())
        except Exception as e:
            summary["metrics_error"] = str(e)

    return summary


# -----------------------------------------------------------------------------
# Modal Functions: NVIDIA A100 & NVIDIA A10G
# -----------------------------------------------------------------------------
@app.function(
    image=image,
    gpu="A100",  # NVIDIA A100 GPU (40GB VRAM)
    cpu=16.0,    # 16 vCPUs for parallel feature extraction
    memory=65536, # 64 GB RAM
    timeout=3600 * 4, # Up to 4 hours runtime
    volumes={"/vol": volume},
    env={
        "HF_HOME": "/vol/huggingface",
        "PYTHONUNBUFFERED": "1",
        "OMP_NUM_THREADS": "16",
    },
)
def run_on_a100(
    quick: bool = False,
    embedder: str = "bgem3",
    loco: bool = True,
    sample_s1: int = 0,
    k: int = 25,
    kd: int = 15,
    folds: int = 5,
    min_thr_safety: float = 0.0,
):
    return _execute_pipeline_core(
        quick=quick,
        embedder=embedder,
        loco=loco,
        sample_s1=sample_s1,
        k=k,
        kd=kd,
        folds=folds,
        min_thr_safety=min_thr_safety,
    )


@app.function(
    image=image,
    gpu="A10G",  # NVIDIA A10G GPU (24GB VRAM) — Budget Option (~$1.10/hr)
    cpu=16.0,
    memory=65536,
    timeout=3600 * 4,
    volumes={"/vol": volume},
    env={
        "HF_HOME": "/vol/huggingface",
        "PYTHONUNBUFFERED": "1",
        "OMP_NUM_THREADS": "16",
    },
)
def run_on_a10g(
    quick: bool = False,
    embedder: str = "bgem3",
    loco: bool = True,
    sample_s1: int = 0,
    k: int = 25,
    kd: int = 15,
    folds: int = 5,
    min_thr_safety: float = 0.0,
):
    return _execute_pipeline_core(
        quick=quick,
        embedder=embedder,
        loco=loco,
        sample_s1=sample_s1,
        k=k,
        kd=kd,
        folds=folds,
        min_thr_safety=min_thr_safety,
    )


# -----------------------------------------------------------------------------
# Local Entrypoint: Dataset Sync
# -----------------------------------------------------------------------------
@app.local_entrypoint()
def sync_dataset(force: bool = False):
    """One-time sync of local dataset to Modal Volume."""
    if not DATA_DIR.exists():
        print(f"❌ Error: Local dataset directory not found at: {DATA_DIR}")
        return

    train_dir = DATA_DIR / "train"
    test_dir = DATA_DIR / "test"
    if not train_dir.exists() or not test_dir.exists():
        print(f"❌ Error: Missing train/ or test/ folders inside {DATA_DIR}")
        return

    files = [p for p in DATA_DIR.rglob("*") if p.is_file() and not p.name.startswith(".")]
    total_gb = sum(p.stat().st_size for p in files) / (1024**3)

    print("=" * 72)
    print("📦 SYNCING DATASET TO MODAL VOLUME 'amazon-ml-data'")
    print(f"Local Path:  {DATA_DIR}")
    print(f"Remote Path: /dataset on Modal Volume")
    print(f"Total Files: {len(files)} ({total_gb:.2f} GB)")
    print("=" * 72)

    for f in sorted(files):
        rel = f.relative_to(DATA_DIR)
        print(f"  • {rel} ({f.stat().st_size / (1024**2):.1f} MB)")

    print("\n⏳ Uploading dataset to cloud volume (only needed once)...")
    t0 = time.time()
    with volume.batch_upload(force=force) as batch:
        batch.put_directory(DATA_DIR, "/dataset")

    elapsed = time.time() - t0
    print(f"\n🎉 Dataset sync complete in {elapsed:.1f}s ({elapsed/60:.2f} min)!")
    print("You can now launch the training pipeline on A100 with:")
    print("    modal run modal_runner.py --quick")
    print("    modal run modal_runner.py --embedder bgem3 --loco")


# -----------------------------------------------------------------------------
# Local Entrypoint: Main Pipeline Runner
# -----------------------------------------------------------------------------
@app.local_entrypoint()
def main(
    quick: bool = False,
    embedder: str = "none",  # none, e5small, bgem3, minilm
    loco: bool = False,
    sample_s1: int = 0,
    k: int = 25,
    kd: int = 15,
    folds: int = 5,
    min_thr_safety: float = 0.0,
    gpu: str = "A100",        # A100 or A10G
    download: bool = True,
    validate: bool = True,
):
    """Launch the pipeline on Modal cloud and download final results."""
    print("=" * 72)
    print("🚀 TEAM SLYTHERINE — MODAL CLOUD RUNNER")
    print(f"Target GPU: {gpu.upper()} | Embedder: {embedder} | Quick: {quick} | LOCO: {loco}")
    print("=" * 72)

    # 1. Remote call
    if gpu.upper() == "A10G":
        print("Launching on NVIDIA A10G (Budget option: ~$1.10/hr)...")
        res = run_on_a10g.remote(
            quick=quick,
            embedder=embedder,
            loco=loco,
            sample_s1=sample_s1,
            k=k,
            kd=kd,
            folds=folds,
            min_thr_safety=min_thr_safety,
        )
    else:
        print("Launching on NVIDIA A100 (High performance: ~$2.10/hr)...")
        res = run_on_a100.remote(
            quick=quick,
            embedder=embedder,
            loco=loco,
            sample_s1=sample_s1,
            k=k,
            kd=kd,
            folds=folds,
            min_thr_safety=min_thr_safety,
        )

    print("\n" + "=" * 72)
    print("📊 CLOUD EXECUTION SUMMARY")
    print("=" * 72)
    print(f"Elapsed Time: {res.get('elapsed_seconds', 0)/60:.2f} minutes")
    if "metrics" in res:
        metrics = res["metrics"]
        print(f"Blocking Recall Ceiling: {metrics.get('blocking_recall', 'N/A')}")
        print(f"Macro F0.5 (Tuned):       {metrics.get('oof_tuned_macro_f05', 'N/A')}")
        print(f"Optimal Threshold:       {metrics.get('best_threshold', 'N/A')}")

    # 2. Download results locally
    if download:
        print("\n📥 Downloading output artifacts from Modal Volume...")
        volume.reload()
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        REP_DIR.mkdir(parents=True, exist_ok=True)

        download_targets = [
            ("output/matching_results.tsv", OUT_DIR / "matching_results.tsv"),
            ("output/candidate_pairs.tsv", OUT_DIR / "candidate_pairs.tsv"),
            ("reports/metrics.json", REP_DIR / "metrics.json"),
            ("reports/threshold_sweep.tsv", REP_DIR / "threshold_sweep.tsv"),
            ("reports/oof_errors.tsv", REP_DIR / "oof_errors.tsv"),
        ]

        for remote_path, local_path in download_targets:
            try:
                content = b"".join(volume.read_file(remote_path))
                local_path.write_bytes(content)
                print(f"  ✓ Saved {remote_path} -> {local_path} ({len(content)/(1024**2):.2f} MB)")
            except Exception as e:
                print(f"  • {remote_path} not found in volume or skipped.")

    # 3. Local submission validation
    if validate:
        validator = ROOT_DIR / "student_resource" / "utils" / "validate_submission.py"
        test_dir = DATA_DIR / "test"
        matching_file = OUT_DIR / "matching_results.tsv"
        candidate_file = OUT_DIR / "candidate_pairs.tsv"

        if validator.exists() and matching_file.exists():
            print("\n🔍 Running local validation check on generated outputs...")
            cmd = [
                sys.executable,
                str(validator),
                "--matching", str(matching_file),
                "--candidate", str(candidate_file),
                "--test-dir", str(test_dir),
            ]
            check = subprocess.run(cmd)
            if check.returncode == 0:
                print("🎉 Output files are 100% VALID according to official scorer rules!")
            else:
                print("⚠️ Submission validator reported issues (check output above).")
