# Babel job 9357329 checkpoint-interval replacement

Job `9357329` proved the aligned-v4 training path on four L40S GPUs, but its
executed runtime snapshot retained `save_steps: 500`.  The exact executed
snapshot remains at:

`/home/gneubig/exp/adp/runs/openhands_sdk_training/adpv2_full_24k_eb95b66_20260717/evidence/qwen35_4b_adpv2_full24k_aligned_babel_one_epoch.source.9357329.restart_0.yaml`

Before replacement it had:

- loaded exactly 9,196,689 train examples and 500 validation examples;
- configured one epoch and 287,397 optimizer steps, with no `max_steps`;
- logged at least ten real optimizer losses, beginning with `0.6288`;
- driven all four L40S GPUs at up to 100% utilization and 33,951 MiB used;
- initialized and server-synced the fixed W&B run ID
  `adpv2-full24k-aligned-v4-qwen35-4b-babel-1ep-eb95b66-20260717`;
- created no complete checkpoint by optimizer step 10, as expected from the
  500-step save interval.

Observed early optimizer steps took several minutes each.  Since preprocessing
and tokenized-artifact saving consumed about seven hours of the 48-hour Slurm
allocation, retaining a 500-step interval risked reaching Babel's wall-time
USR1 before any complete checkpoint existed.  A requeue in that state would
restart training from the beginning and violate the explicit restartability
acceptance criterion.

The source config was corrected to `save_steps: 10` in commit
`bee32bf8e075094c6abcc9efe2473ed18f4162fe`; its SHA-256 at that commit was
`08c0b682df12c79dca3133ca517a6018d196f6c5f2173e815525dee15aa4c7b1`.
The replacement intentionally reuses the already-complete tokenized artifact,
the same output directory (which contained no checkpoint), and the same W&B
run ID with resume semantics.  Job submission is not acceptance: the
replacement must again prove a real loss, active GPUs, server-synced W&B, a
complete checkpoint, and progress past that checkpoint after one deliberate
USR1 requeue.
