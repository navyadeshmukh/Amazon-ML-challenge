# Full workflow for Team Slytherin - run from Amazon-ML-challenge/
param([string]$Team = "Slytherin")
$ErrorActionPreference = "Stop"
$STUDENT = "..\student_resource"

Write-Host "=== Step 0: Install dependencies ===" -ForegroundColor Cyan
pip install -r code\requirements.txt
# Uncomment the next line for embedding upgrade:
# pip install -r code\requirements-embeddings.txt

Set-Location code\business_entity_resolution

Write-Host "`n=== Step 1: Quick sanity run (fast dev loop) ===" -ForegroundColor Cyan
python run_pipeline.py --quick

Write-Host "`n=== Step 2: Full baseline (TF-IDF + string features + LightGBM) ===" -ForegroundColor Cyan
python run_pipeline.py --loco

Write-Host "`n=== Step 3: Validate baseline submission ===" -ForegroundColor Cyan
python "$STUDENT\utils\validate_submission.py" `
    --matching "..\..\output\matching_results.tsv" `
    --candidate "..\..\output\candidate_pairs.tsv" `
    --test-dir "$STUDENT\dataset\test"

# Keep baseline as fallback
Copy-Item "..\..\output\matching_results.tsv" "..\..\output\baseline_matching_results.tsv"

Write-Host "`n=== Step 4: Embedding upgrade (e5small on CUDA) ===" -ForegroundColor Cyan
python run_pipeline.py --embedder e5small --device cuda --loco

Write-Host "`n=== Step 5: Validate embedding submission ===" -ForegroundColor Cyan
python "$STUDENT\utils\validate_submission.py" `
    --matching "..\..\output\matching_results.tsv" `
    --candidate "..\..\output\candidate_pairs.tsv" `
    --test-dir "$STUDENT\dataset\test"

Write-Host "`n=== Step 6: Package submission ===" -ForegroundColor Cyan
Remove-Item -ErrorAction SilentlyContinue "..\..\output\baseline_matching_results.tsv"
python src\make_submission_zip.py --team $Team

Set-Location ..\..
Write-Host "`nDone! Submission zip created." -ForegroundColor Green
