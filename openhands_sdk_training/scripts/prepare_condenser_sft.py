#!/usr/bin/env python3
from __future__ import annotations

import ast
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any

ADP_REPO = Path('/home/gneubig/work/adp/agent-data-protocol-pr244')
EXP_ROOT = Path('/home/gneubig/exp/adp')
SFT_ROOT = EXP_ROOT / 'datasets/hf_release'
STD_ROOT = EXP_ROOT / 'datasets/hf_std'
SPLIT_ROOT = EXP_ROOT / 'datasets/paper_openhands_nonweb_v1'
OUTPUT_ROOT = EXP_ROOT / 'datasets/condenser_sft/openhands_nonweb_12k_llm'
STANDARDIZED_ROOT = OUTPUT_ROOT / 'standardized'
LOCAL_METADATA_ROOT = OUTPUT_ROOT / 'metadata'
SUPPORTED_CONDENSER_CODE_LANGUAGES = {'bash', 'sh', 'shell', 'python', 'python3', 'py'}



MEDIA_TAG_REPLACEMENTS = {
    '<image>': '&lt;image&gt;',
    '</image>': '&lt;/image&gt;',
    '<video>': '&lt;video&gt;',
    '</video>': '&lt;/video&gt;',
    '<audio>': '&lt;audio&gt;',
    '</audio>': '&lt;/audio&gt;',
}


def clean_sft_record(line: bytes) -> bool:
    try:
        record = json.loads(line)
    except json.JSONDecodeError:
        return False
    if not isinstance(record, dict):
        return False
    conversations = record.get('conversations')
    if not isinstance(conversations, list) or not conversations:
        return False
    for message in conversations:
        if not isinstance(message, dict):
            return False
        if not isinstance(message.get('from'), str):
            return False
        if not isinstance(message.get('value'), str):
            return False
    return True


def line_hash(line: bytes) -> str:
    return hashlib.sha1(line).hexdigest()


def eval_count(raw_count: int) -> int:
    previous = min(100, max(10, math.ceil(0.02 * raw_count)))
    return max(1, previous // 2)


def select_train_indices(raw_count: int, eval_indices: set[int], weight: float, rng: random.Random) -> set[int]:
    remaining = [idx for idx in range(raw_count) if idx not in eval_indices]
    if weight == 1:
        return set(remaining)
    target_count = math.ceil(len(remaining) * weight)
    if weight < 1:
        return set(rng.sample(remaining, target_count))
    return set(rng.choices(remaining, k=target_count))


def schema_for_annotation(annotation: ast.expr | None) -> dict[str, Any]:
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


def load_api_tools(dataset: str) -> dict[str, dict[str, Any]]:
    api_path = ADP_REPO / 'datasets' / dataset / 'api.py'
    if not api_path.exists():
        return {}
    tree = ast.parse(api_path.read_text())
    tools: dict[str, dict[str, Any]] = {}
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        properties: dict[str, Any] = {}
        required: list[str] = []
        defaults_start = len(node.args.args) - len(node.args.defaults)
        for index, arg in enumerate(node.args.args):
            properties[arg.arg] = schema_for_annotation(arg.annotation)
            if index < defaults_start:
                required.append(arg.arg)
        description = ast.get_docstring(node) or f'Dataset tool {node.name}.'
        tools[node.name] = {
            'type': 'function',
            'function': {
                'name': node.name,
                'description': description,
                'parameters': {
                    'type': 'object',
                    'properties': properties,
                    'additionalProperties': False,
                    'required': required,
                },
            },
        }
    return tools


def collect_requirements(path: Path) -> tuple[set[str], set[str]]:
    api_functions: set[str] = set()
    code_languages: set[str] = set()
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            for event in row.get('content', []):
                if event.get('class_') == 'api_action':
                    api_functions.add(event.get('function', ''))
                elif event.get('class_') == 'code_action':
                    code_languages.add(event.get('language', ''))
    api_functions.discard('')
    code_languages.discard('')
    return api_functions, code_languages


def metadata_path(dataset: str) -> Path:
    return ADP_REPO / 'datasets' / dataset / 'metadata.json'


def ensure_metadata(dataset: str, api_functions: set[str], code_languages: set[str]) -> dict[str, Any]:
    existing = metadata_path(dataset)
    if existing.exists():
        metadata = json.loads(existing.read_text())
        out_path = LOCAL_METADATA_ROOT / f'{dataset}.metadata.json'
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(metadata, indent=2) + '\n')
        return metadata

    api_tools = load_api_tools(dataset)
    missing = sorted(api_functions - set(api_tools))
    alias_only = {'submit', 'stop', 'finish', 'str_replace_editor', 'think', 'task_tracker'}
    missing = [name for name in missing if name not in alias_only]
    if missing:
        print(f'WARNING {dataset}: API functions missing from api.py: {missing}')

    custom_tools = [api_tools[name] for name in sorted(api_functions & set(api_tools))]
    code_enabled = sorted(
        {'bash' if lang in {'bash', 'sh', 'shell'} else lang for lang in code_languages}
        & SUPPORTED_CONDENSER_CODE_LANGUAGES
    )
    metadata = {
        'custom_tools': custom_tools,
        'code_enabled': code_enabled,
        'browser_enabled': False,
    }
    out_path = LOCAL_METADATA_ROOT / f'{dataset}.metadata.json'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metadata, indent=2) + '\n')
    existing.write_text(json.dumps(metadata, indent=2) + '\n')
    return metadata


def main() -> None:
    manifest = json.loads((SPLIT_ROOT / 'paper_openhands_nonweb.manifest.json').read_text())
    datasets = manifest['datasets']
    STANDARDIZED_ROOT.mkdir(parents=True, exist_ok=True)
    LOCAL_METADATA_ROOT.mkdir(parents=True, exist_ok=True)

    eval_rng = random.Random(manifest['eval_seed'])
    train_rng = random.Random(manifest['train_seed'])
    summary: dict[str, Any] = {
        'datasets': {},
        'source_manifest': str(SPLIT_ROOT / 'paper_openhands_nonweb.manifest.json'),
        'selection': 'reproduced source indices from manifest seeds; invalid SFT rows skipped with the same cleaner used for LLaMA-Factory',
    }

    for dataset in datasets:
        print('extracting', dataset, flush=True)
        stats = manifest['stats'][dataset]
        raw_count = stats['raw_count']
        eval_indices = set(eval_rng.sample(range(raw_count), eval_count(raw_count)))
        train_indices = select_train_indices(
            raw_count,
            eval_indices,
            stats['weight'],
            train_rng,
        )

        sft_path = SFT_ROOT / dataset / 'full_sft/full_sft_openhands.jsonl'
        std_path = STD_ROOT / dataset / 'full_std.jsonl'
        selected_indices = eval_indices | train_indices
        sft_lines: dict[int, bytes] = {}
        std_lines: dict[int, bytes] = {}
        eval_hashes: set[str] = set()

        with sft_path.open('rb') as handle:
            for index, line in enumerate(handle):
                if index in selected_indices:
                    sft_lines[index] = line
                if index in eval_indices:
                    eval_hashes.add(line_hash(line))

        with std_path.open('rb') as handle:
            for index, line in enumerate(handle):
                if index in selected_indices:
                    std_lines[index] = line

        paths = {
            split: STANDARDIZED_ROOT / f'{split}_{dataset}.jsonl'
            for split in ('train', 'eval')
        }
        counts = {'train': 0, 'eval': 0}
        skipped = {'train_invalid_sft': 0, 'train_eval_hash_match': 0, 'eval_invalid_sft': 0}
        with paths['train'].open('wb') as train_out, paths['eval'].open('wb') as eval_out:
            for index in sorted(eval_indices):
                if not clean_sft_record(sft_lines[index]):
                    skipped['eval_invalid_sft'] += 1
                    continue
                eval_out.write(std_lines[index])
                counts['eval'] += 1

            for index in sorted(train_indices):
                if line_hash(sft_lines[index]) in eval_hashes:
                    skipped['train_eval_hash_match'] += 1
                    continue
                if not clean_sft_record(sft_lines[index]):
                    skipped['train_invalid_sft'] += 1
                    continue
                train_out.write(std_lines[index])
                counts['train'] += 1

        api_functions: set[str] = set()
        code_languages: set[str] = set()
        for path in paths.values():
            funcs, langs = collect_requirements(path)
            api_functions.update(funcs)
            code_languages.update(langs)
        metadata = ensure_metadata(dataset, api_functions, code_languages)
        summary['datasets'][dataset] = {
            'counts': counts,
            'skipped': skipped,
            'api_functions': sorted(api_functions),
            'code_languages': sorted(code_languages),
            'metadata_custom_tool_count': len(metadata.get('custom_tools', [])),
            'metadata_code_enabled': metadata.get('code_enabled', []),
        }
        print(dataset, counts, skipped, flush=True)

    (OUTPUT_ROOT / 'prepare_manifest.json').write_text(json.dumps(summary, indent=2) + '\n')
    print('wrote', OUTPUT_ROOT / 'prepare_manifest.json')


if __name__ == '__main__':
    main()
