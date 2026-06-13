#!/usr/bin/env bash
#SBATCH --job-name=adp-swe-score
#SBATCH --partition=debug
#SBATCH --qos=debug_qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=/home/gneubig/exp/adp/evals/full-swebench/slurm-%x-%j.out
#SBATCH --error=/home/gneubig/exp/adp/evals/full-swebench/slurm-%x-%j.err

set -euo pipefail

PYTHON=/home/gneubig/work/openhands-benchmarks-venv/bin/python
BENCH_EVAL_DIR=/home/gneubig/workspace/project/b0ec6769629643e9b4eb723ca0e440cf/benchmarks-pr743
RUN_DIR=${RUN_DIR:?Set RUN_DIR to a completed SWE-bench run directory}
DATASET=${DATASET:-princeton-nlp/SWE-bench_Verified}
SPLIT=${SPLIT:-test}
TIMEOUT_SECONDS=${TIMEOUT_SECONDS:-1800}
SCORE_DIR=${SCORE_DIR:-${RUN_DIR}/apptainer_patch_eval}
SANDBOX_ROOT=${SANDBOX_ROOT:-/scratch/${USER}/swebench-eval-sandboxes-${SLURM_JOB_ID}}

export OPENHANDS_SWEBENCH_IMAGE_TEMPLATE=${OPENHANDS_SWEBENCH_IMAGE_TEMPLATE:-"ghcr.io/epoch-research/swe-bench.eval.x86_64.{instance_id}:latest"}
export APPTAINER_CACHEDIR=${APPTAINER_CACHEDIR:-/data/user_data/${USER}/openhands/apptainer-cache-swebench-eval-epoch}
export APPTAINER_TMPDIR=${APPTAINER_TMPDIR:-/scratch/${USER}/apptainer-eval-tmp-${SLURM_JOB_ID}}
export OPENHANDS_SUPPRESS_BANNER=1
export HF_HOME=/home/gneubig/exp/adp/cache/hf

mkdir -p "${SCORE_DIR}" "${SANDBOX_ROOT}" "${APPTAINER_CACHEDIR}" "${APPTAINER_TMPDIR}" "${HF_HOME}"

echo "job_id=${SLURM_JOB_ID}"
echo "node_list=${SLURM_JOB_NODELIST}"
echo "started_at=$(date -Is)"
echo "run_dir=${RUN_DIR}"
echo "score_dir=${SCORE_DIR}"
echo "sandbox_root=${SANDBOX_ROOT}"
echo "dataset=${DATASET}"
echo "split=${SPLIT}"
echo "timeout_seconds=${TIMEOUT_SECONDS}"
echo "swebench_image_template=${OPENHANDS_SWEBENCH_IMAGE_TEMPLATE}"
echo "apptainer_cachedir=${APPTAINER_CACHEDIR}"
echo "apptainer_tmpdir=${APPTAINER_TMPDIR}"

export BENCH_EVAL_DIR RUN_DIR DATASET SPLIT TIMEOUT_SECONDS SCORE_DIR SANDBOX_ROOT

"${PYTHON}" - <<'PY'
import hashlib
import json
import os
import re
import sys
import traceback
from pathlib import Path

bench_eval_dir = os.environ["BENCH_EVAL_DIR"]
for module_name in list(sys.modules):
    if module_name == "benchmarks" or module_name.startswith("benchmarks."):
        sys.modules.pop(module_name, None)
sys.path.insert(0, bench_eval_dir)

from datasets import load_dataset
from swebench.harness.constants import KEY_INSTANCE_ID, KEY_PREDICTION

from benchmarks.swebench import apptainer_eval, constants
from benchmarks.utils.patch_utils import remove_files_from_patch


def slugify(value: str, max_len: int = 90) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-")
    return value[:max_len] or "candidate"


def iter_rows(path: Path):
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            yield line_no, json.loads(line)


def candidate_sources(run_dir: Path) -> list[Path]:
    sources: list[Path] = []
    for name in ("output.jsonl", "output_errors.jsonl"):
        path = run_dir / name
        if path.exists() and path.stat().st_size > 0:
            sources.append(path)

    if not sources:
        for name in ("output.jsonl", "output_errors.jsonl"):
            sources.extend(
                sorted(
                    path
                    for path in run_dir.rglob(name)
                    if path.is_file() and path.stat().st_size > 0
                )
            )

    include_attempts = os.getenv("INCLUDE_ATTEMPTS", "0").lower() in {
        "1",
        "true",
        "yes",
    }
    if include_attempts or not sources:
        sources.extend(
            sorted(
                path
                for path in run_dir.rglob("output.critic_attempt_*.jsonl")
                if path.is_file() and path.stat().st_size > 0
            )
        )

    # Keep first occurrence if a caller intentionally includes attempt files
    # alongside final output files.
    deduped: list[Path] = []
    seen: set[Path] = set()
    for source in sources:
        resolved = source.resolve()
        if resolved not in seen:
            seen.add(resolved)
            deduped.append(source)
    return deduped


run_dir = Path(os.environ["RUN_DIR"]).resolve()
score_dir = Path(os.environ["SCORE_DIR"]).resolve()
sandbox_root = Path(os.environ["SANDBOX_ROOT"]).resolve()
dataset = os.environ["DATASET"]
split = os.environ["SPLIT"]
timeout_seconds = int(os.environ["TIMEOUT_SECONDS"])

score_dir.mkdir(parents=True, exist_ok=True)
(score_dir / "candidates").mkdir(exist_ok=True)

sources = candidate_sources(run_dir)
print("candidate_sources=" + json.dumps([str(source) for source in sources]), flush=True)

candidates: list[dict] = []
seen_candidate_keys: set[tuple[str, str]] = set()
for source in sources:
    for line_no, row in iter_rows(source):
        instance_id = row.get("instance_id")
        test_result = row.get("test_result") or {}
        patch = test_result.get("git_patch") or ""
        patch = remove_files_from_patch(patch, constants.SETUP_FILES_TO_REMOVE)
        if not instance_id or not patch.strip():
            continue

        patch_sha = hashlib.sha1(patch.encode("utf-8")).hexdigest()
        candidate_key = (instance_id, patch_sha)
        if candidate_key in seen_candidate_keys:
            continue
        seen_candidate_keys.add(candidate_key)

        candidate_id = slugify(
            f"{source.stem}-line{line_no}-{instance_id}-{patch_sha[:12]}"
        )
        candidates.append(
            {
                "candidate_id": candidate_id,
                "instance_id": instance_id,
                "source_file": str(source),
                "source_line": line_no,
                "source_kind": source.name,
                "patch_sha1": patch_sha,
                "patch_chars": len(patch),
                "error": row.get("error"),
                "git_patch_captured_on_error": bool(
                    test_result.get("git_patch_captured_on_error")
                ),
                "patch": patch,
            }
        )

candidates_path = score_dir / "patch_candidates.jsonl"
with candidates_path.open("w", encoding="utf-8") as f:
    for candidate in candidates:
        record = {key: value for key, value in candidate.items() if key != "patch"}
        f.write(json.dumps(record, sort_keys=True) + "\n")
print(f"patch_candidate_count={len(candidates)}", flush=True)
print(f"patch_candidates={candidates_path}", flush=True)

instances = {
    instance[KEY_INSTANCE_ID]: instance for instance in load_dataset(dataset, split=split)
}

results_path = score_dir / "patch_eval_results.jsonl"
summary_path = score_dir / "patch_eval_summary.json"
resolved = 0
applied = 0
failed = 0

with results_path.open("w", encoding="utf-8") as results_file:
    for index, candidate in enumerate(candidates, 1):
        candidate_id = candidate["candidate_id"]
        instance_id = candidate["instance_id"]
        candidate_score_dir = score_dir / "candidates" / candidate_id
        candidate_score_dir.mkdir(parents=True, exist_ok=True)
        patch_path = candidate_score_dir / "model.patch"
        patch_path.write_text(candidate["patch"], encoding="utf-8")

        result = {key: value for key, value in candidate.items() if key != "patch"}
        result.update(
            {
                "index": index,
                "total": len(candidates),
                "candidate_score_dir": str(candidate_score_dir),
                "patch_path": str(patch_path),
                "status": "started",
            }
        )
        print(
            f"[{index}/{len(candidates)}] scoring {candidate_id} "
            f"instance={instance_id}",
            flush=True,
        )

        try:
            instance = instances[instance_id]
            prediction = {
                KEY_INSTANCE_ID: instance_id,
                KEY_PREDICTION: candidate["patch"],
                "model_name_or_path": run_dir.name,
            }
            report = apptainer_eval.score_instance(
                instance=instance,
                prediction=prediction,
                score_dir=candidate_score_dir,
                sandbox_root=sandbox_root,
                timeout_seconds=timeout_seconds,
                apptainer_cache=Path(os.environ["APPTAINER_CACHEDIR"]),
            )
            instance_report = report.get(instance_id, {})
            was_resolved = bool(instance_report.get("resolved"))
            patch_applied = bool(instance_report.get("patch_successfully_applied"))
            resolved += int(was_resolved)
            applied += int(patch_applied)
            result.update(
                {
                    "status": "ok",
                    "resolved": was_resolved,
                    "patch_successfully_applied": patch_applied,
                    "report_path": str(candidate_score_dir / instance_id / "report.json"),
                    "test_output_path": str(
                        candidate_score_dir / instance_id / "test_output.txt"
                    ),
                    "apply_patch_log": str(
                        candidate_score_dir / instance_id / "apply_patch.log"
                    ),
                }
            )
        except Exception as exc:
            failed += 1
            result.update(
                {
                    "status": "error",
                    "resolved": False,
                    "patch_successfully_applied": False,
                    "exception": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )

        results_file.write(json.dumps(result, sort_keys=True) + "\n")
        results_file.flush()

summary = {
    "run_dir": str(run_dir),
    "score_dir": str(score_dir),
    "dataset": dataset,
    "split": split,
    "timeout_seconds": timeout_seconds,
    "swebench_image_template": os.getenv("OPENHANDS_SWEBENCH_IMAGE_TEMPLATE"),
    "total_patch_candidates": len(candidates),
    "resolved": resolved,
    "unresolved": len(candidates) - resolved,
    "patch_successfully_applied": applied,
    "score_errors": failed,
    "candidates_path": str(candidates_path),
    "results_path": str(results_path),
}
summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
print(json.dumps(summary, sort_keys=True), flush=True)
PY

echo "finished_at=$(date -Is)"
