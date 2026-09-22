# FlashLoop

**Training-free inference optimization for looped Transformers.**

FlashLoop exploits cross-loop redundancy through token-sparse activation
updates, loop-aware sparse attention, and cross-loop KV sharing with residual
quantization. This repository provides two implementations for **Ouro**.

| Implementation | Intended use | KV representation |
| --- | --- | --- |
| `flashloop_engine` | Optimized, batch-one CUDA inference | Physically packed int4 cache with custom readers |
| `flashloop_torch` | Readable algorithm reference and quality experiments | Fake quantization with materialized tensors |


## Installation

Requirements: Linux, NVIDIA CUDA GPU, Python 3.10+, CUDA-compatible PyTorch,
and Transformers 4.56.2. Building the optimized reader additionally requires
a CUDA toolkit (`nvcc`) and compatible C++ compiler. Install PyTorch for your
CUDA environment before installing this project.

```bash
python -m pip install -e .
# Only needed for the optimized engine:
bash scripts/build_kernels.sh
export PYTHONPATH="$PWD/build:${PYTHONPATH:-}"
```

The PyTorch reference does not require compiling FlashLoop's CUDA extension.
The tested environment and current validation scope are summarized below.

## Quick start

Obtain a trusted official Ouro checkpoint separately, then run:

```bash
python examples/generate.py --backend engine --model /path/to/Ouro-1.4B \
  --prompt "What is the capital of France?" --max-new-tokens 64

python examples/generate.py --backend torch --model /path/to/Ouro-1.4B \
  --prompt "What is the capital of France?" --max-new-tokens 64
```


To run three prompts through both backends and save actual outputs:

```bash
bash scripts/smoke_test.sh /path/to/Ouro-1.4B
```

## Python API

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from flashloop_engine import FlashLoopEngine
from flashloop_torch import flashloop

model_path = "/path/to/Ouro-1.4B"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
input_ids = tokenizer.apply_chat_template(
    [{"role": "user", "content": "What is the capital of France?"}],
    tokenize=True, add_generation_prompt=True, return_tensors="pt",
).cuda()

# Optimized engine.
engine = FlashLoopEngine.from_pretrained(model_path)
output_ids = engine.generate(input_ids, max_new_tokens=64)
print(tokenizer.decode(output_ids[0, input_ids.shape[1]:], skip_special_tokens=True))
del engine
torch.cuda.empty_cache()

# PyTorch reference: use a fresh hook context for each request.
model = AutoModelForCausalLM.from_pretrained(
    model_path, trust_remote_code=True, torch_dtype=torch.bfloat16,
    attn_implementation="eager",
).cuda().eval()
with torch.inference_mode(), flashloop(model) as hooks:
    output_ids = model.generate(input_ids, max_new_tokens=64,
                               do_sample=False, use_cache=True)
    audit = {name: hook.audit() for name, hook in hooks.items()}
```

The explicit prefill/decode path in `examples/generate.py` is the path used
in the packaged smoke test. The high-level API example above is illustrative;
it has not received the same separate end-to-end validation.

## Configuration and supported scope

Defaults: four loops; dense loops 1–2; token retention 25%/10% in loops 3–4;
10% key retention for sparse late-loop decode; 4-bit K/V, group size 64,
and a 64-token BF16 residual tail.

The engine targets official Ouro-1.4B and Ouro-2.6B configurations, batch
size one, unpadded inputs, BF16 CUDA execution, head dimension 128 and equal
query/KV head counts. This packaged release has a short functional smoke
test on **Ouro-1.4B only**.

## Repository layout

```text
flashloop_engine/       Optimized execution, packed cache, CUDA kernels
flashloop_torch/        Reference API and internal algorithm implementations
examples/generate.py   Shared real-generation example
scripts/               Kernel build and two-backend smoke commands
tests/                 CPU-safe repository checks
```

Run CPU-safe checks with `python -m pip install -e '.[test]'` followed by
`python -m pytest`. These checks do not replace GPU inference tests.
No model weights, evaluation datasets, cluster credentials, private logs or
artificial-delay demonstrations are included.

## Citation

Please cite the FlashLoop paper when using this code in research. Public
paper links and the final BibTeX entry will be added when available.

## License

Project code is released under the [MIT License](LICENSE). Third-party
notices, including KIVI attribution, are preserved in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Model checkpoints and
external dependencies retain their own terms.
