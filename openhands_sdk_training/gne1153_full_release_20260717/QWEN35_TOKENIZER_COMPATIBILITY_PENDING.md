# Qwen3.5 4B/35B tokenized-artifact compatibility: pending local Flame check

Recorded 2026-07-18 before any 35B submission.

The Babel 4B run loaded `Qwen/Qwen3.5-4B-Base` from Hugging Face commit
`1001bb4d826a52d1f399e183466143f4da7b741b`. Its local tokenizer artifact
hashes are:

- `tokenizer.json`: `fe000e3ed39ed12b8d2481d527d44f93c65d37e87645d2dcc80d1bf9d50d2927`
- `tokenizer_config.json`: `3891e840d7dc5fca0af33d3a25083a735e36fe06214e3f707024820cb6b9f89c`
- `vocab.json`: `ce99b4cb2983d118806ce0a8b777a35b093e2000a503ebde25853284c9dfa003`
- `merges.txt`: `a9d356d7bdf1ef4949e3e748e95b8e10ad9d4e2e838eddc38a0a7b6b94d1db8d`

The official `Qwen/Qwen3.5-35B-A3B` repository reported commit
`59d61f3ce65a6d9863b86d2e96597125219dc754` at inspection time. Its
`tokenizer.json` SHA256 is
`5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42`,
so it is not byte-identical to the 4B tokenizer. Structural comparison found
that the 35B tokenizer adds these tokens absent from the 4B tokenizer:

- ID 248066: `<tool_response>`
- ID 248067: `</tool_response>`
- ID 248068: `<think>`
- ID 248069: `</think>`

The tokenizer configs also declare different EOS tokens: `<|endoftext|>` for
4B Base and `<|im_end|>` for 35B-A3B. These differences are potentially
training-relevant because the selected Qwen3.5 tool/thinking template emits
the affected strings.

Therefore the 4B-produced tokenized dataset must not be reused for 35B merely
on model-family identity. Before launch on `flame-earlybirds`, inspect and hash
the tokenizer files in the actual local model directory
`/project/flame/gneubig/adp/models/Qwen3.5-35B-A3B`, compare token IDs on
deterministic rendered tool/thinking examples, and either prove token-ID
compatibility or create a distinct 35B-tokenized artifact from the identical
validated aligned-v4 9,196,689/500 rows. The prepared 35B recipe remains held
until that check and Babel acceptance both pass.
