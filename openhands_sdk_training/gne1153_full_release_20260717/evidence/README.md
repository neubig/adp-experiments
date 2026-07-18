# GNE-1153 ADP-v2 full-release launch evidence

This directory indexes durable evidence for the corrected Babel and Orchard
training launches. Large cluster-generated logs and data manifests live under
the non-overwriting run roots named by the launch scripts.

## Incorrect Babel job cancellation

- Target: Slurm job `9325086` (`gne1153-qwen35-4b-2d`) on `babel-p9-20`.
- Read-only check at `2026-07-17T20:25:15-04:00`: job was `RUNNING` with
  elapsed time `1-10:28:06`.
- Authorized command: `scancel 9325086`; exit status `0`.
- Terminal accounting at `2026-07-17T20:25:20-04:00`: `CANCELLED by 17414`,
  start `2026-07-16T09:57:09`, end `2026-07-17T20:25:15`, node
  `babel-p9-20`.
- `squeue -h -j 9325086` returned no rows at that verification time.

## Reproducible artifacts

- Artifact commit: `7dfac38b447c71eece2dc5ef3b671c017b3b6c59` on
  `gne1153-qwen35-4b-babel-fix`.
- Persistent-monitor and skip-validator commit:
  `962d9330aa6998e428ebcde30df4863c6bc34b3a`.
- Exact SHA-256 inventory: `ARTIFACTS.sha256` in this directory.
- Build job `9354064` executed `build_full_release.py` at SHA-256
  `2c6747b4cfd0454b382219caa8f126c12aee90be09eabc49f4bf477e0f7c5572`
  and `build_full_release.sbatch` at SHA-256
  `4f1dff55369a17575bb88787bf9f85ea32db1f3abb227fcf991f33b20aa80233`.
- Failed job `9356360` executed a preserved Slurm-spooled script at
  `/home/gneubig/exp/adp/runs/openhands_sdk_training/adpv2_full_24k_eb95b66_20260717/evidence/train_qwen35_4b_babel.9356360.restart_0.sbatch`,
  SHA-256 `345b3fe043220854305e3f0a506ec46d30873fee6821fa37758784face4a8671`.
  Its source YAML SHA-256 was `08a5a67559dc450c0dace7f6e75ce1c95c72a0ef7190b763ac2838b8547d11d0`
  and its ZeRO config SHA-256 was
  `2868fc47fa4f4006aeca2e81443338679dbf9b4cb14980e735bd0af6099aae6d`.
- The committed corrected Slurm script uses immutable, job/restart-specific
  evidence filenames and prints the hashes of the exact Slurm-spooled script,
  source YAML, and ZeRO config before work begins.

## Corrected Babel jobs and validation

- Full-release validation/adaptation build: `9354064`, submitted
  `2026-07-17T20:31:43-04:00`; running on `babel-t5-24` from
  `2026-07-17T20:31:44-04:00`.
- Build job `9354064` reached terminal `COMPLETED` (`0:0`) after validating all
  52 configs and 9,640,563 pinned source rows.
- First production attempt `9354098` reached allocation but failed before
  tokenization because heterogeneous, loader-ignored top-level `metadata`
  produced an Arrow schema mismatch. It logged no optimizer step and is not an
  accepted launch.
- The canonical-adapter outputs were preserved. Projection job `9356262`
  removed only the final top-level `metadata` field while byte-preserving all
  training fields; it completed 52/52 files and 9,196,689 adapted rows.
- Independent projection validation job `9356352` completed `0:0`. Its closed
  manifest SHA-256 is
  `c26b00d521262170ebfeb3f9250475a8072e74c28ba2474b379f562a5ec37a32`,
  projection-manifest SHA-256 is
  `54d64ec4f1816c68f9509c897adefbfa8fa1abd1c76de2ffdd138f38e324a9c9`,
  and loader `dataset_info.json` SHA-256 is
  `1c0fabfed54551b49360c4cdee42774a4fc4e2b52fc1400b1457e561a8d6087f`.
- Second production attempt `9356360` reached allocation and loaded configs
  00--36, then failed on the intentionally empty config 37 file. The manifest
  proves `omniact` has 6,789 source rows and zero trainable adapter outputs,
  with all 6,789 excluded for `record contains no trainable assistant/function
  response`. The corrected committed YAML loads the other 51 configs and the
  Slurm preflight asserts that this is the sole zero-trainable config; the
  52-config release remains fully represented in the manifest.
- An earlier pending submission `9354074` was cancelled before allocation and
  replaced by `9354098` solely to make Slurm restart logs append durably rather
  than reuse an unexpanded `%r` filename.
- Cluster evidence root:
  `/home/gneubig/exp/adp/runs/openhands_sdk_training/adpv2_full_24k_eb95b66_20260717/evidence`.

## Pinned release

- Repository: `neulab/adp-v2`
- Revision: `eb95b66ca30ef071d5e49495d5d780c29f9aef7b`
- Split: `sft_openhands_24k`
- Dataset-card assertion: 52 configs and 9,640,563 rows.
- The exact per-config table is embedded as an assertion in
  `../build_full_release.py`; a mismatch aborts the build.

## Restart and acceptance status

- The source config uses `num_train_epochs: 1.0` and has no `max_steps`.
- Runtime configs are created by
  `openhands_sdk_training/scripts/gne1153_prepare_restartable_config.py`, which
  accepts a checkpoint only when `trainer_state.json` and model artifacts are
  present and `trainer_state.global_step` equals the checkpoint directory step.
- `WANDB_RUN_ID=adpv2-full24k-qwen35-4b-babel-1ep-eb95b66-20260717` and
  `WANDB_RESUME=allow` are stable across Slurm restarts.
- These configuration properties are not restart evidence by themselves. A
  controlled USR1/requeue after a complete checkpoint, followed by proof of a
  larger resumed optimizer step and the same W&B run URL/ID, is required before
  Babel acceptance.
- As of the artifact commit above, Babel Stage 1 has **not** passed: there is no
  accepted real optimizer loss step or W&B metric sync yet.

## Persisted continuation deadline

- Agent Canvas backend allocation `9353684` is expected to end at
  `2026-07-18T07:47:52-04:00`; its requested extension was denied.
- Cluster evidence is written outside that allocation under the run-root
  `evidence/` directory, and Slurm jobs remain independently queryable through
  `sacct`/`squeue`. Any monitor installed before that deadline must record its
  own script, job ID, and output path here.
- Orchard work starts only after Babel Stage 1 passes. Prior H100 resources must
  be resolved through the proven `flame-earlybirds` Slurm jobs/configs rather
  than assuming an `ssh orchard` alias exists.

## Cognitivekernel skip validation

- Validation job `9356577` completed `0:0` on 2026-07-18 and recomputed the
  canonical adapter outcome for all 47,271 `cognitivekernel_pro_sft` source
  rows using the manifest-pinned source and adapter hashes.
- Source schema distribution is exact: all 47,271 rows contain `system,user`;
  exactly 87 additionally contain `assistant`. No row contains a function
  response, tool call, or trainable flag. The only top-level keys are
  `id,messages,metadata,tools`, and every message has only `content,role`.
- Consequently, 47,184 rows are `system,user` and end on `user`; 87 rows are
  `system,user,assistant` and end on `assistant`.
- The independent response test classified the same 47,184 rows as lacking any
  assistant/function-call response indicator and the same 87 rows as containing
  one. There were zero response-bearing skipped rows and zero response-free
  retained rows.
- Five deterministic samples per outcome use the smallest
  `sha256(config + NUL + record id + NUL + source line)` keys. Evidence records
  IDs, record hashes, roles, schemas, and content lengths, but no conversation
  text.
- Conclusion: the high skip rate is correct canonical-adapter behavior, not a
  conversion bug. The skipped records contain prompts but no response target.
- Durable evidence:
  `/home/gneubig/exp/adp/runs/openhands_sdk_training/adpv2_full_24k_eb95b66_20260717/evidence/cognitivekernel_skip_validation.json`,
  SHA-256 `74d86c23ded124ed603ccc7c15a82e9fa5037a8ba6c2fc4a00b3368d875f426c`.
- The pre-validation and pre-schema-note manifests were preserved at the paths
  recorded in the validation note. After adding the explicit schema evidence,
  the manifest SHA-256 is
  `2c9b15665569e3beb5e296c186d50c11960bcfc6a37e2d166ea9ceec57c11304`
  (prior SHA-256
  `a2cb08664d2b7ec1a22bc24147d9afc8b954858d89d8da060a103858d114953c`).

## Persistent Babel monitor

- Training job under observation: `9356517`.
- Independent monitor job: `9356578`, with a requested wall time of 11:55:00.
- Latest atomic snapshot:
  `/home/gneubig/exp/adp/runs/openhands_sdk_training/adpv2_full_24k_eb95b66_20260717/evidence/monitor_9356517.snapshot.json`;
  append-only history uses the same basename with `.history.jsonl`.
- The monitor records evidence only and does not mutate or accept the training
  job. At its first snapshot, `9356517` was pending and all runtime acceptance
  flags other than the full-52 release manifest were false.
