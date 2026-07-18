# Babel Stage-1 acceptance: job 9358921

Accepted at 2026-07-18 11:00 EDT only after exercising the checkpoint-backed restart path.

- Dataset: pinned ADP-v2 revision `eb95b66ca30ef071d5e49495d5d780c29f9aef7b`, 52 source configs, 9,640,563 source rows, 9,196,689 adapted/tokenized train rows, 500 validation rows, and zero tokenizer skips. Tokenized manifest SHA256: `fa500cff3543f2d0a622a75e611eb43c9772fe32f70b73492cd3439c89370773`.
- Training: Slurm job `9358921`, four L40S GPUs, one epoch (`287,397` optimizer steps), `save_steps: 10`.
- Before the deliberate signal, checkpoint-20 was complete and stable with four model shards, four optimizer shards, four RNG states, and trainer global step 20.
- `scancel --batch --signal=USR1 9358921` was sent at 2026-07-18 10:44:54 EDT. Bash deferred the trap while waiting on foreground `torchrun`; the exact fail-closed unblock record is `requeue_unblock_9358921.out` (SHA256 `a39312f0ae1d6dc5ccf9ba329285519f6b6396a04d49f4a136c87a4b3e90a75c`). It terminated only torchrun PID 791036, after which the pending USR1 trap logged at 10:49:11 and requeued the same job.
- Slurm then reported `Restarts=1`; job 9358921 returned to `RUNNING` at 10:51:37. Restart evidence selected checkpoint-20 and the trainer logged `Resuming training from checkpoint with epoch 0 and global step 20`, followed by progress `21/287397` and post-restart loss `0.5931`.
- Restart checkpoint-selection evidence: `/home/gneubig/exp/adp/runs/openhands_sdk_training/adpv2_full_24k_eb95b66_20260717/evidence/checkpoint_selection.9358921.restart_1.json`, SHA256 `4d79e515fb3e655c90c1d190255ebd860bb8f15a227e793f0bf19008ddc4a069`.
- GPUs after restart were active at 99-100% utilization with approximately 30-33.5 GiB used per device.
- W&B resumed the same run ID `adpv2-full24k-aligned-v4-qwen35-4b-babel-1ep-eb95b66-20260717`. Server verification at 2026-07-18 11:00:32 EDT found 39 synced loss rows (33 before requeue), latest `_step=38`, `train/loss=0.5931179523468018`, state `running`, URL `https://wandb.ai/gneubig/adp-experiments/runs/adpv2-full24k-aligned-v4-qwen35-4b-babel-1ep-eb95b66-20260717`.
- W&B evidence: `/home/gneubig/exp/adp/runs/openhands_sdk_training/adpv2_full_24k_eb95b66_20260717/evidence/wandb_babel_9358921.restart_1.json`, SHA256 `f233b929f6fa4ebda51c51e7000625c9fab709067a8e9e76d1e6bf54e409ca5f`.

