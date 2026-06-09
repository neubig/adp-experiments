SWE-bench smoke evaluation summary
==================================

Date: 2026-06-09

Model checkpoint:
/home/gneubig/exp/adp/runs/openhands_sdk_training/full_condenser_24k_all_records_adapted/output_qwen35_4b_seq32768_zero3_one_epoch_flashattn/checkpoint-2000

Best completed inference run:
/home/gneubig/exp/adp/evals/2000/swe-bench-smoke/princeton-nlp__SWE-bench_Verified-test/openai/local-qwen35-4b_sdk_c950fdb_maxiter_12_N_smoke5_native_tools_32k_tp2_repo_root_2048_max12

Configuration:
- vLLM served through Apptainer on two L40S GPUs, port 8012
- tensor_parallel_size=2
- max_model_len=32768
- native_tool_calling=true
- vLLM tool parser=qwen3_coder
- OpenHands workspace=apptainer
- dataset=princeton-nlp/SWE-bench_Verified, split=test
- n_limit=5, num_workers=1, max_iterations=12
- public skills disabled
- prompt=benchmarks/benchmarks/swebench/prompts/smoke_native_repo_root.j2
- output tokens=2048

Inference result:
- output.jsonl rows: 5
- non-empty patches: 2/5
- empty patches: 3/5
- errors recorded in output rows: 0/5
- conversion to SWE-bench format: 5 entries converted, 0 conversion errors

Apptainer scoring result:
- resolved: 2/5
- unresolved: 3/5
- non-empty patches scored: 2/2
- empty patches treated as unresolved: 3/3
- scorer artifact: /home/gneubig/exp/adp/evals/2000/swe-bench-smoke/apptainer_score_smoke.py
- scorer output: /home/gneubig/exp/adp/evals/2000/swe-bench-smoke/apptainer_official_score_32k_tp2/summary.json

Instances:
- scikit-learn__scikit-learn-13439: non-empty patch, 579 chars, resolved
- django__django-12155: non-empty patch, 2470 chars, resolved; includes a new local test script in addition to the code change
- scikit-learn__scikit-learn-25232: empty patch, unresolved
- django__django-14434: empty patch, unresolved
- django__django-13279: empty patch, unresolved

Converted SWE-bench predictions:
/home/gneubig/exp/adp/evals/2000/swe-bench-smoke/princeton-nlp__SWE-bench_Verified-test/openai/local-qwen35-4b_sdk_c950fdb_maxiter_12_N_smoke5_native_tools_32k_tp2_repo_root_2048_max12/output.swebench.jsonl

Logs:
- inference: /home/gneubig/exp/adp/evals/2000/swe-bench-smoke/logs/swebench_infer_smoke5_native_tools_32k_tp2_repo_root_2048_max12.log
- conversion: /home/gneubig/exp/adp/evals/2000/swe-bench-smoke/logs/swebench_eval_convert_smoke5_native_tools_32k_tp2_repo_root_2048_max12.log
- Modal official scoring probe: /home/gneubig/exp/adp/evals/2000/swe-bench-smoke/logs/swebench_eval_official_probe_smoke5_native_tools_32k_tp2_repo_root_2048_max12.log
- Docker official scoring probe: /home/gneubig/exp/adp/evals/2000/swe-bench-smoke/logs/swebench_eval_docker_probe_smoke5_native_tools_32k_tp2_repo_root_2048_max12.log
- Apptainer scoring wrapper: /home/gneubig/exp/adp/evals/2000/swe-bench-smoke/logs/apptainer_score_smoke_32k_tp2.log

Official SWE-bench scoring:
- Modal scoring failed before running tests: modal.exception.AuthError: Token not found
- Docker scoring failed before running tests: Docker socket/runtime is unavailable
- The OpenHands benchmarks scoring wrapper exposes Modal and Docker scoring flags, but no Apptainer scoring flag.
- To continue without Modal/Docker, I reproduced the harness flow with Apptainer: apply the model patch, run make_test_spec().eval_script, then grade with swebench.harness.grading.get_eval_report().
- The Apptainer scoring wrapper successfully scored the 5-row smoke run as 2/5 resolved.

Diagnostic notes:
- The 32k TP2 native-tool setup completed inference cleanly for all 5 instances and produced the same 2/5 non-empty patch count as the 16k native-tool run.
- Native vLLM tool parsing was materially better than text-only/non-native mode: the model used file_editor correctly and produced usable diffs for the two non-empty rows.
- The model still struggled behaviorally: 3/5 instances produced empty patches, and the first instance kept verifying from /workspace instead of the repository subdirectory even after the prompt emphasized the repo path.
- The official Docker Hub SWE-bench images were usable under Apptainer only after binding the sandbox /opt over /opt; the cluster host /opt bind otherwise hid /opt/miniconda3.

Continuation: additional available-image instances
=================================================

Inference run:
/home/gneubig/exp/adp/evals/2000/swe-bench-smoke/princeton-nlp__SWE-bench_Verified-test/openai/local-qwen35-4b_sdk_c950fdb_maxiter_12_N_available_next5_native_tools_32k_tp2_repo_root_2048_max12

Selection:
- The next 15 dataset-order instances were all Astropy and failed before inference because their OpenHands Apptainer image tags were not present in GHCR.
- I queried available source-minimal tags and selected the next five instances with available images:
  django__django-11999, django__django-13670, scikit-learn__scikit-learn-25973, sphinx-doc__sphinx-7757, sympy__sympy-15599.

Inference result:
- output.jsonl rows: 5
- raw non-empty patches: 4/5
- raw empty patches: 1/5
- errors recorded in output rows: 0/5
- conversion to SWE-bench format: 5 entries converted, 0 conversion errors
- converted non-empty patches: 3/5
- converted empty patches: 2/5

Apptainer scoring result:
- resolved: 0/5
- unresolved: 5/5
- patch applied but unresolved: django__django-11999, django__django-13670, sympy__sympy-15599
- empty after conversion: scikit-learn__scikit-learn-25973, sphinx-doc__sphinx-7757
- scorer output: /home/gneubig/exp/adp/evals/2000/swe-bench-smoke/apptainer_official_score_available_next5_32k_tp2/summary.json

Instances:
- django__django-11999: raw patch 2790 chars, converted non-empty, applied, unresolved
- django__django-13670: raw patch 417 chars, converted non-empty, applied, unresolved
- scikit-learn__scikit-learn-25973: empty patch, unresolved
- sphinx-doc__sphinx-7757: raw patch 1314 chars, converted to empty because the patch only changed setup.py and tox.ini, which the OpenHands SWE-bench converter removes via SETUP_FILES_TO_REMOVE
- sympy__sympy-15599: raw patch 672 chars, converted non-empty, applied, unresolved

Converted SWE-bench predictions:
/home/gneubig/exp/adp/evals/2000/swe-bench-smoke/princeton-nlp__SWE-bench_Verified-test/openai/local-qwen35-4b_sdk_c950fdb_maxiter_12_N_available_next5_native_tools_32k_tp2_repo_root_2048_max12/output.swebench.jsonl

Logs:
- inference: /home/gneubig/exp/adp/evals/2000/swe-bench-smoke/logs/swebench_infer_available_next5_native_tools_32k_tp2_repo_root_2048_max12.log
- failed Astropy batch: /home/gneubig/exp/adp/evals/2000/swe-bench-smoke/logs/swebench_infer_next15_native_tools_32k_tp2_repo_root_2048_max12.log
- conversion: /home/gneubig/exp/adp/evals/2000/swe-bench-smoke/logs/swebench_eval_convert_available_next5_native_tools_32k_tp2_repo_root_2048_max12.log
- Apptainer scoring: /home/gneubig/exp/adp/evals/2000/swe-bench-smoke/logs/apptainer_score_available_next5_32k_tp2.log

Combined scored result:
- available-image scored instances: 10
- resolved: 2/10
- unresolved: 8/10
- checkpoint and serving configuration were unchanged from the initial smoke run.
