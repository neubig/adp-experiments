#!/usr/bin/env python
import argparse
import json
import shutil
import subprocess
from pathlib import Path

from datasets import load_dataset
from swebench.harness.constants import (
    APPLY_PATCH_FAIL,
    KEY_INSTANCE_ID,
    KEY_MODEL,
    KEY_PREDICTION,
    TESTS_TIMEOUT,
)
from swebench.harness.grading import get_eval_report
from swebench.harness.test_spec.test_spec import make_test_spec


ROOT = Path("/home/gneubig/exp/adp/evals/2000/swe-bench-smoke")
RUN_DIR = (
    ROOT
    / "princeton-nlp__SWE-bench_Verified-test/openai/"
    / "local-qwen35-4b_sdk_c950fdb_maxiter_12_N_smoke5_native_tools_32k_tp2_repo_root_2048_max12"
)
PREDICTIONS = RUN_DIR / "output.swebench.jsonl"
SCORE_DIR = ROOT / "apptainer_official_score_32k_tp2"
SANDBOX_ROOT = Path("/scratch/gneubig/swebench-apptainer-score")
APPTAINER_CACHE = Path("/home/gneubig/work/apptainer-cache")
DATASET = "princeton-nlp/SWE-bench_Verified"
SPLIT = "test"
TIMEOUT_SECONDS = 1800


def parse_args():
    parser = argparse.ArgumentParser(
        description="Score SWE-bench predictions with Apptainer sandboxes."
    )
    parser.add_argument("--predictions", type=Path, default=PREDICTIONS)
    parser.add_argument("--score-dir", type=Path, default=SCORE_DIR)
    parser.add_argument("--sandbox-root", type=Path, default=SANDBOX_ROOT)
    parser.add_argument("--dataset", default=DATASET)
    parser.add_argument("--split", default=SPLIT)
    parser.add_argument("--timeout-seconds", type=int, default=TIMEOUT_SECONDS)
    parser.add_argument("--apptainer-cache", type=Path, default=APPTAINER_CACHE)
    return parser.parse_args()


def run(cmd, log_path=None, timeout=None):
    env = {
        **dict(PATH="/usr/local/bin:/usr/bin:/bin"),
        "APPTAINER_CACHEDIR": str(APPTAINER_CACHE),
    }
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout,
        env=env,
    )
    if log_path is not None:
        log_path.write_text(proc.stdout)
    return proc


def load_predictions():
    rows = []
    with PREDICTIONS.open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return {row[KEY_INSTANCE_ID]: row for row in rows}


def image_uri(instance):
    spec = make_test_spec(instance, namespace="swebench")
    return "docker://" + spec.instance_image_key


def ensure_sandbox(instance):
    instance_id = instance[KEY_INSTANCE_ID]
    sandbox = SANDBOX_ROOT / instance_id
    if sandbox.exists():
        return sandbox
    tmp = sandbox.with_suffix(".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.parent.mkdir(parents=True, exist_ok=True)
    proc = run(
        ["apptainer", "build", "--sandbox", str(tmp), image_uri(instance)],
        SCORE_DIR / f"{instance_id}.build.log",
        timeout=3600,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"apptainer build failed for {instance_id}")
    tmp.rename(sandbox)
    return sandbox


def apptainer_base_cmd(sandbox, work_dir):
    return [
        "apptainer",
        "exec",
        "--no-mount",
        "hostfs",
        "--bind",
        f"{sandbox / 'opt'}:/opt",
        "--bind",
        f"{work_dir}:/mnt",
        "--writable",
        str(sandbox),
        "bash",
        "-lc",
    ]


def verify_sandbox(instance, sandbox, work_dir):
    cmd = apptainer_base_cmd(sandbox, work_dir) + [
        "source /opt/miniconda3/bin/activate && "
        "conda activate testbed && "
        "cd /testbed && "
        "python --version && "
        "command -v python && "
        "git rev-parse HEAD"
    ]
    proc = run(cmd, work_dir / "sandbox_verify.log", timeout=120)
    if proc.returncode != 0:
        raise RuntimeError(f"sandbox verification failed for {instance[KEY_INSTANCE_ID]}")


def score_instance(instance, prediction):
    instance_id = instance[KEY_INSTANCE_ID]
    work_dir = SCORE_DIR / instance_id
    work_dir.mkdir(parents=True, exist_ok=True)

    report_path = work_dir / "report.json"
    if report_path.exists():
        return json.loads(report_path.read_text())

    patch = prediction.get(KEY_PREDICTION) or ""
    if not patch.strip():
        report = {
            instance_id: {
                "patch_is_None": prediction.get(KEY_PREDICTION) is None,
                "patch_exists": False,
                "patch_successfully_applied": False,
                "resolved": False,
                "skipped_empty_patch": True,
            }
        }
        report_path.write_text(json.dumps(report, indent=2))
        return report

    test_spec = make_test_spec(instance)
    sandbox = ensure_sandbox(instance)
    (work_dir / "patch.diff").write_text(patch)
    (work_dir / "eval.sh").write_text(test_spec.eval_script)
    verify_sandbox(instance, sandbox, work_dir)

    shell = f"""
set -uo pipefail
cd /testbed
git config --global --add safe.directory /testbed || true
git reset --hard HEAD
git clean -fd
applied=0
apply_output=/mnt/apply_patch.log
: > "$apply_output"
for cmd in "git apply --verbose /mnt/patch.diff" "git apply --verbose --reject /mnt/patch.diff" "patch --batch --fuzz=5 -p1 -i /mnt/patch.diff"; do
  echo "$cmd" >> "$apply_output"
  if bash -lc "$cmd" >> "$apply_output" 2>&1; then
    applied=1
    break
  fi
done
if [ "$applied" != "1" ]; then
  {{
    echo "{APPLY_PATCH_FAIL}"
    cat "$apply_output"
  }} > /mnt/test_output.txt
  exit 0
fi
git -c core.fileMode=false diff > /mnt/git_diff_before.diff 2>&1 || true
timeout {TIMEOUT_SECONDS} /bin/bash /mnt/eval.sh > /mnt/test_output.txt 2>&1
status=$?
git -c core.fileMode=false diff > /mnt/git_diff_after.diff 2>&1 || true
if [ "$status" = "124" ]; then
  echo "{TESTS_TIMEOUT}" >> /mnt/test_output.txt
fi
exit 0
"""
    proc = run(
        apptainer_base_cmd(sandbox, work_dir) + [shell],
        work_dir / "apptainer_exec.log",
        timeout=TIMEOUT_SECONDS + 300,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"apptainer exec failed for {instance_id}")

    test_output = work_dir / "test_output.txt"
    report = get_eval_report(
        test_spec=test_spec,
        prediction=prediction,
        test_log_path=str(test_output),
        include_tests_status=True,
    )
    report_path.write_text(json.dumps(report, indent=2))
    return report


def main():
    args = parse_args()
    global PREDICTIONS
    global SCORE_DIR
    global SANDBOX_ROOT
    global DATASET
    global SPLIT
    global TIMEOUT_SECONDS
    global APPTAINER_CACHE
    PREDICTIONS = args.predictions
    SCORE_DIR = args.score_dir
    SANDBOX_ROOT = args.sandbox_root
    DATASET = args.dataset
    SPLIT = args.split
    TIMEOUT_SECONDS = args.timeout_seconds
    APPTAINER_CACHE = args.apptainer_cache

    SCORE_DIR.mkdir(parents=True, exist_ok=True)
    predictions = load_predictions()
    wanted = set(predictions)
    dataset = [
        ex
        for ex in load_dataset(DATASET, split=SPLIT)
        if ex[KEY_INSTANCE_ID] in wanted
    ]
    reports = {}
    for instance in dataset:
        instance_id = instance[KEY_INSTANCE_ID]
        print(f"scoring {instance_id}", flush=True)
        report = score_instance(instance, predictions[instance_id])
        reports.update(report)
        print(
            f"{instance_id}: resolved={reports[instance_id].get('resolved')} "
            f"patch_applied={reports[instance_id].get('patch_successfully_applied')}",
            flush=True,
        )

    resolved = sum(1 for r in reports.values() if r.get("resolved"))
    summary = {
        "dataset": DATASET,
        "split": SPLIT,
        "predictions": str(PREDICTIONS),
        "score_dir": str(SCORE_DIR),
        "total": len(reports),
        "resolved": resolved,
        "unresolved": len(reports) - resolved,
        "reports": reports,
    }
    (SCORE_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
