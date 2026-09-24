"""Build a submission zip: code/ + output/ + docs."""
import argparse
import zipfile
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--team", required=True, help="team name for the zip")
    ap.add_argument("--solution-root", default=str(Path(__file__).resolve().parents[3]),
                    help="root of the Amazon-ML-challenge folder")
    a = ap.parse_args()
    sol = Path(a.solution_root)
    out = sol / "output"
    assert (out / "matching_results.tsv").exists(), f"{out / 'matching_results.tsv'} not found"
    assert (out / "candidate_pairs.tsv").exists(), f"{out / 'candidate_pairs.tsv'} not found"
    zp = sol / f"{a.team}_submission.zip"
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(sol.rglob("*")):
            if p.is_file() and ".git" not in p.parts and ".venv" not in p.parts and "__pycache__" not in p.parts:
                if p.suffix in (".py", ".txt", ".md", ".tsv", ".json", ".ps1", ".sh", ".gitignore"):
                    zf.write(p, p.relative_to(sol))
    print(f"Created {zp} ({zp.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
