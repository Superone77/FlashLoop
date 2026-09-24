"""Small real-generation smoke test, not a performance benchmark."""
import argparse
import contextlib
import json
from pathlib import Path


def attention_implementation(backend):
    return 'eager' if backend == 'torch' else 'sdpa'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--backend', choices=['engine', 'torch'], required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--prompt', action='append')
    p.add_argument('--max-new-tokens', type=int, default=24)
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    if args.max_new_tokens < 2:
        p.error('Use at least two tokens to exercise decode')
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.manual_seed(2026)
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, torch_dtype=torch.bfloat16,
        attn_implementation=attention_implementation(args.backend)).cuda().eval()
    if model.config.model_type != 'ouro' or model.config.total_ut_steps != 4:
        raise ValueError('This smoke test requires four-loop Ouro')
    engine = None
    if args.backend == 'engine':
        from flashloop import FlashLoopEngine
        engine = FlashLoopEngine(model)
    else:
        from flashloop_torch import flashloop
    prompts = args.prompt or [
        'What is the capital of France? Answer in one sentence.',
        'Alice has 12 apples and gives 5 to Bob. How many apples remain? Explain briefly.',
        'Explain why the sky looks blue in two short sentences.',
    ]
    rows = []
    for prompt in prompts:
        ids = tok.apply_chat_template([{'role': 'user', 'content': prompt}],
            tokenize=True, add_generation_prompt=True, return_tensors='pt').cuda()
        completion, cache = [], None
        context = contextlib.nullcontext(None) if engine else flashloop(model)
        with torch.inference_mode(), context as hooks:
            if engine:
                engine.reset()
            for step in range(args.max_new_tokens):
                if engine:
                    result = engine.prefill(ids) if step == 0 else engine.decode(token)
                    logits = result.logits
                else:
                    result = model(input_ids=ids if step == 0 else token,
                                   past_key_values=cache, use_cache=True, return_dict=True)
                    cache = result.past_key_values
                    logits = result.logits[:, -1]
                if not bool(torch.isfinite(logits).all().item()):
                    raise RuntimeError('Nonfinite logits')
                token = logits.argmax(-1, keepdim=True)
                completion.append(int(token.item()))
                if completion[-1] == tok.eos_token_id:
                    break
            if engine:
                audit = {'cache_storage_bytes': int(engine.state.cache.storage_bytes),
                         'loop3_active': int(engine.state.mask_loop3.sum()),
                         'loop4_active': int(engine.state.mask_loop4.sum())}
                assert audit['cache_storage_bytes'] > 0
            else:
                audit = {name: hook.audit() for name, hook in hooks.items()}
                assert hooks['kv'].cache_updates > 0
                assert hooks['attention_and_prefill'].attention.prefill_attention_calls > 0
            text = tok.decode(completion, skip_special_tokens=True)
            if not text.strip() or len(completion) < 2:
                raise RuntimeError('Empty or immediate-EOS completion; not a valid smoke result')
            row = dict(prompt=prompt, input_ids=ids[0].tolist(), generated_ids=completion,
                       text=text, finite_logits=True, audit=audit)
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
    result = dict(backend=args.backend, model=args.model, gpu=torch.cuda.get_device_name(),
                  model_config=model.config.to_dict(), samples=rows, status='PASS')
    if args.output:
        args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print('OURO_PACKAGE_SMOKE_OK', args.backend, len(rows))


if __name__ == '__main__':
    main()
