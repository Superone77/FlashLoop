"""PyTorch algorithm-quality reference; no packed-cache memory claim."""
from contextlib import contextmanager


@contextmanager
def flashloop(model, *, attention_fraction=0.10, fraction_loop3=0.25,
              fraction_loop4=0.10):
    from ._reference.integration import TokenSparsePrefillCachedMassIntervention
    from ._reference.kv_quantization import CrossLoopKVQuantizationIntervention
    joint = TokenSparsePrefillCachedMassIntervention(
        model, attention_fraction=attention_fraction,
        fraction_loop3=fraction_loop3, fraction_loop4=fraction_loop4)
    kv = None
    try:
        kv = CrossLoopKVQuantizationIntervention(
            model, prefill_intervention=joint.prefill, group_size=64,
            residual_length=64, method='flashloop_kivi4_channel_delta')
        yield {'attention_and_prefill': joint, 'kv': kv}
    finally:
        if kv is not None:
            kv.close()
        joint.close()
