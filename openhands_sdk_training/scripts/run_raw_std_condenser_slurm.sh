#!/usr/bin/env bash

DATASET=$1
if [ -z "$DATASET" ]; then
  echo "Usage: $0 DATASET_NAME" >&2
  exit 2
fi

REPO=${ADP_REPO:-/home/gneubig/work/adp/agent-data-protocol-pr244}
EXP_ROOT=${ADP_EXP_ROOT:-/home/gneubig/exp/adp}
PYTHON=${ADP_PYTHON:-/home/gneubig/work/adp/agent-data-protocol-pr244/.venv/bin/python}
OUT_ROOT=${ADP_OUT_ROOT:-$EXP_ROOT/datasets/software_agent_condenser_24k}
OUT_DIR=$OUT_ROOT/$DATASET
LOG_DIR=$OUT_ROOT/logs
FULL_SFT_DIR=$OUT_DIR/full_sft
MAX_TOKENS=${ADP_MAX_TOKENS:-24000}
TOKEN_LABEL=${ADP_TOKEN_LABEL:-}
if [ -z "$TOKEN_LABEL" ]; then
  TOKEN_LABEL=$((MAX_TOKENS / 1000))k
fi

mkdir -p "$OUT_DIR" "$LOG_DIR" "$FULL_SFT_DIR"

if [ -f "$EXP_ROOT/.env" ]; then
  set -a
  . "$EXP_ROOT/.env"
  set +a
fi

if [ -z "${LLM_MODEL:-}" ]; then
  echo "LLM_MODEL must be set in $EXP_ROOT/.env or the environment" >&2
  exit 1
fi
if [ -z "${LLM_API_KEY:-}" ]; then
  echo "LLM_API_KEY must be set in $EXP_ROOT/.env or the environment" >&2
  exit 1
fi

RAW_JSONL=$OUT_DIR/full_raw.jsonl
ATIF_JSONL=$OUT_DIR/full_atif.jsonl
STD_JSONL=$OUT_DIR/full_std.jsonl
CONDENSER_JSONL=$FULL_SFT_DIR/full_sft_openhands_sdk_condensed_${TOKEN_LABEL}.jsonl
MANIFEST=$OUT_DIR/manifest.json

started_at=$(date -Is)
echo "dataset=$DATASET"
echo "repo=$REPO"
echo "out_dir=$OUT_DIR"
echo "started_at=$started_at"
echo "llm_model=$LLM_MODEL"
echo "max_tokens=$MAX_TOKENS"
echo "token_label=$TOKEN_LABEL"
echo "python=$PYTHON"
repo_branch=$(git -C "$REPO" branch --show-current 2>/dev/null || true)
repo_commit=$(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || true)
echo "repo_branch=$repo_branch"
echo "repo_commit=$repo_commit"
EXPECTED_ADP_BRANCH=${ADP_EXPECTED_BRANCH:-main}
if [ -n "$EXPECTED_ADP_BRANCH" ] && [ "$repo_branch" != "$EXPECTED_ADP_BRANCH" ]; then
  echo "Expected ADP repo branch $EXPECTED_ADP_BRANCH, got $repo_branch" >&2
  exit 1
fi
if ! grep -q 'tool_prefix="dataset_tool"' "$REPO/agents/openhands_sdk/std_to_sft.py"; then
  echo "ADP repo does not include dataset tool truncation patch" >&2
  exit 1
fi

if [ ! -d "$REPO/datasets/$DATASET" ]; then
  echo "Dataset directory not found: $REPO/datasets/$DATASET" >&2
  exit 1
fi

if [ -s "$RAW_JSONL" ]; then
  extract_status=0
  raw_lines=$(wc -l < "$RAW_JSONL" 2>/dev/null || echo 0)
  echo "extract_status=0 raw_lines=$raw_lines reused=$RAW_JSONL"
else
  (
    cd "$REPO/datasets/$DATASET" || exit 1
    env -u PYTHONPATH "$PYTHON" extract_raw.py
  ) > "$RAW_JSONL.tmp" 2> "$LOG_DIR/${DATASET}.extract_raw.stderr"
  extract_status=$?
  raw_lines=$(wc -l < "$RAW_JSONL.tmp" 2>/dev/null || echo 0)
  echo "extract_status=$extract_status raw_lines=$raw_lines"
  if [ "$extract_status" -ne 0 ] || [ "$raw_lines" -eq 0 ]; then
    echo "extract_raw failed or produced no rows" >&2
    exit 1
  fi
  mv "$RAW_JSONL.tmp" "$RAW_JSONL"
fi

if [ -s "$STD_JSONL" ]; then
  atif_status=0
  atif_lines=$(wc -l < "$ATIF_JSONL" 2>/dev/null || echo 0)
  std_status=0
  std_lines=$(wc -l < "$STD_JSONL" 2>/dev/null || echo 0)
  echo "std_status=0 std_lines=$std_lines reused=$STD_JSONL"
elif [ -f "$REPO/datasets/$DATASET/raw_to_standardized.py" ]; then
  atif_status=0
  atif_lines=0
  (
    cd "$REPO/datasets/$DATASET" || exit 1
    PYTHONPATH="$REPO:${PYTHONPATH:-}" "$PYTHON" raw_to_standardized.py < "$RAW_JSONL"
  ) > "$STD_JSONL.tmp" 2> "$LOG_DIR/${DATASET}.raw_to_standardized.stderr"
  std_status=$?
  std_lines=$(wc -l < "$STD_JSONL.tmp" 2>/dev/null || echo 0)
  echo "std_status=$std_status std_lines=$std_lines"
  if [ "$std_status" -ne 0 ] || [ "$std_lines" -eq 0 ]; then
    echo "raw_to_standardized failed or produced no rows" >&2
    exit 1
  fi
  mv "$STD_JSONL.tmp" "$STD_JSONL"
else
  if [ -s "$ATIF_JSONL" ]; then
    atif_status=0
    atif_lines=$(wc -l < "$ATIF_JSONL" 2>/dev/null || echo 0)
    echo "atif_status=0 atif_lines=$atif_lines reused=$ATIF_JSONL"
  else
    (
      cd "$REPO/datasets/$DATASET" || exit 1
      PYTHONPATH="$REPO:${PYTHONPATH:-}" "$PYTHON" raw_to_atif.py < "$RAW_JSONL"
    ) > "$ATIF_JSONL.tmp" 2> "$LOG_DIR/${DATASET}.raw_to_atif.stderr"
    atif_status=$?
    atif_lines=$(wc -l < "$ATIF_JSONL.tmp" 2>/dev/null || echo 0)
    echo "atif_status=$atif_status atif_lines=$atif_lines"
    if [ "$atif_status" -ne 0 ] || [ "$atif_lines" -eq 0 ]; then
      echo "raw_to_atif failed or produced no rows" >&2
      exit 1
    fi
    mv "$ATIF_JSONL.tmp" "$ATIF_JSONL"
  fi

  (
    cd "$REPO/datasets/$DATASET" || exit 1
    PYTHONPATH="$REPO:${PYTHONPATH:-}" "$PYTHON" atif_to_std.py < "$ATIF_JSONL"
  ) > "$STD_JSONL.tmp" 2> "$LOG_DIR/${DATASET}.atif_to_std.stderr"
  std_status=$?
  std_lines=$(wc -l < "$STD_JSONL.tmp" 2>/dev/null || echo 0)
  echo "std_status=$std_status std_lines=$std_lines"
  if [ "$std_status" -ne 0 ] || [ "$std_lines" -eq 0 ]; then
    echo "atif_to_std failed or produced no rows" >&2
    exit 1
  fi
  mv "$STD_JSONL.tmp" "$STD_JSONL"
fi

if [ ! -f "$REPO/datasets/$DATASET/metadata.json" ]; then
  PYTHONPATH="$REPO:${PYTHONPATH:-}" "$PYTHON" - "$REPO" "$DATASET" "$STD_JSONL" <<'PY'
import ast
import json
import sys
from pathlib import Path

repo = Path(sys.argv[1])
dataset = sys.argv[2]
std_path = Path(sys.argv[3])
metadata_path = repo / 'datasets' / dataset / 'metadata.json'
api_path = repo / 'datasets' / dataset / 'api.py'
SUPPORTED_CODE = {'bash', 'sh', 'shell', 'python', 'python3', 'py'}
ALIASES = {'submit', 'stop', 'finish', 'str_replace_editor', 'think', 'task_tracker'}


def schema_for_annotation(annotation):
    if annotation is None:
        return {}
    text = ast.unparse(annotation)
    if text in {'str', 'Optional[str]'}:
        return {'type': 'string'}
    if text in {'int', 'Optional[int]'}:
        return {'type': 'integer'}
    if text in {'float', 'Optional[float]'}:
        return {'type': 'number'}
    if text in {'bool', 'Optional[bool]'}:
        return {'type': 'boolean'}
    if text in {'list', 'List', 'Optional[list]'} or text.startswith(('list[', 'List[')):
        return {'type': 'array'}
    if text in {'dict', 'Dict', 'Optional[dict]'} or text.startswith(('dict[', 'Dict[')):
        return {'type': 'object'}
    return {}


def load_api_tools():
    if not api_path.exists():
        return {}
    tree = ast.parse(api_path.read_text())
    tools = {}
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        properties = {}
        required = []
        defaults_start = len(node.args.args) - len(node.args.defaults)
        for index, arg in enumerate(node.args.args):
            properties[arg.arg] = schema_for_annotation(arg.annotation)
            if index < defaults_start:
                required.append(arg.arg)
        tools[node.name] = {
            'type': 'function',
            'function': {
                'name': node.name,
                'description': ast.get_docstring(node) or f'Dataset tool {node.name}.',
                'parameters': {
                    'type': 'object',
                    'properties': properties,
                    'additionalProperties': False,
                    'required': required,
                },
            },
        }
    return tools

api_functions = set()
code_languages = set()
with std_path.open() as handle:
    for line in handle:
        row = json.loads(line)
        for event in row.get('content', []):
            if event.get('class_') == 'api_action':
                api_functions.add(event.get('function'))
            elif event.get('class_') == 'code_action':
                lang = (event.get('language') or '').lower()
                if lang in {'sh', 'shell'}:
                    lang = 'bash'
                if lang in SUPPORTED_CODE:
                    code_languages.add(lang)
api_functions.discard(None)
api_tools = load_api_tools()
custom_tools = [api_tools[name] for name in sorted(api_functions & set(api_tools))]
metadata = {
    'custom_tools': custom_tools,
    'code_enabled': sorted(code_languages),
    'browser_enabled': False,
}
metadata_path.write_text(json.dumps(metadata, indent=2) + '\n')
print(f'wrote_metadata={metadata_path} custom_tools={len(custom_tools)} code_enabled={metadata["code_enabled"]}')
PY
fi


if [ -s "$CONDENSER_JSONL" ]; then
  cond_status=0
  cond_lines=$(wc -l < "$CONDENSER_JSONL" 2>/dev/null || echo 0)
  echo "condensation_status=0 condensation_lines=$cond_lines reused=$CONDENSER_JSONL"
else
  CONDENSER_TMP="$CONDENSER_JSONL.tmp"
  RESUME_STD_JSONL="$OUT_DIR/full_std.resume.jsonl"
  if [ -s "$CONDENSER_TMP" ]; then
    PYTHONPATH="$REPO:${PYTHONPATH:-}" "$PYTHON" - "$STD_JSONL" "$CONDENSER_TMP" "$RESUME_STD_JSONL" <<'PY_INNER'
import json
import sys
from pathlib import Path

std_path = Path(sys.argv[1])
partial_path = Path(sys.argv[2])
out_path = Path(sys.argv[3])
processed = set()
with partial_path.open(errors='replace') as handle:
    for line in handle:
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        source_id = row.get('metadata', {}).get('source_trajectory_id')
        if source_id:
            processed.add(source_id)
with std_path.open() as in_handle, out_path.open('w') as out_handle:
    kept = skipped = 0
    for line in in_handle:
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get('id') in processed:
            skipped += 1
            continue
        out_handle.write(line if line.endswith('\n') else line + '\n')
        kept += 1
print(f'resume_processed={len(processed)} resume_skipped={skipped} resume_remaining={kept}', flush=True)
PY_INNER
    COND_INPUT="$RESUME_STD_JSONL"
  else
    : > "$CONDENSER_TMP"
    COND_INPUT="$STD_JSONL"
  fi

  remaining_lines=$(wc -l < "$COND_INPUT" 2>/dev/null || echo 0)
  if [ "$remaining_lines" -eq 0 ]; then
    cond_status=0
    cond_lines=$(wc -l < "$CONDENSER_TMP" 2>/dev/null || echo 0)
    echo "condensation_status=0 condensation_lines=$cond_lines resumed_complete=1"
    mv "$CONDENSER_TMP" "$CONDENSER_JSONL"
  else
    (
      cd "$REPO" || exit 1
      MY_DATASET="$DATASET" PYTHONPATH="$REPO:${PYTHONPATH:-}" "$PYTHON" \
        agents/openhands_sdk/condensation_sft.py \
          --max-tokens "$MAX_TOKENS" \
          --model "$LLM_MODEL" \
          --concurrency 8 \
          --chunk-size 8 \
          --continue-on-error \
          < "$COND_INPUT"
    ) >> "$CONDENSER_TMP" 2>> "$LOG_DIR/${DATASET}.openhands_sdk_condensation.stderr"
    cond_status=$?
    cond_lines=$(wc -l < "$CONDENSER_TMP" 2>/dev/null || echo 0)
    echo "condensation_status=$cond_status condensation_lines=$cond_lines"
    if [ "$cond_status" -eq 0 ] && [ "$cond_lines" -gt 0 ]; then
      mv "$CONDENSER_TMP" "$CONDENSER_JSONL"
    else
      echo "condensation_sft failed or produced no rows; keeping partial $CONDENSER_TMP" >&2
    fi
  fi
fi

finished_at=$(date -Is)
cat > "$MANIFEST" <<JSON
{
  "dataset": "$DATASET",
  "started_at": "$started_at",
  "finished_at": "$finished_at",
  "raw_lines": $raw_lines,
  "std_lines": $std_lines,
  "atif_status": $atif_status,
  "atif_lines": $atif_lines,
  "condensation_status": $cond_status,
  "condensation_lines": $cond_lines,
  "max_tokens": $MAX_TOKENS,
  "token_label": "$TOKEN_LABEL",
  "llm_model": "$LLM_MODEL",
  "raw_jsonl": "$RAW_JSONL",
  "atif_jsonl": "$ATIF_JSONL",
  "std_jsonl": "$STD_JSONL",
  "condensed_openhands_sdk_jsonl": "$CONDENSER_JSONL"
}
JSON

echo "finished_at=$finished_at"
if [ "$cond_status" -ne 0 ] || [ "$cond_lines" -eq 0 ]; then
  exit 1
fi
