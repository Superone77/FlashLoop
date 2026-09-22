# Third-party notices

## KIVI

The CUDA reader under `flashloop_engine/csrc/` includes an adaptation of
[KIVI](https://github.com/jy-yuan/KIVI), copyright (c) 2024 jiayi yuan,
licensed under MIT. The full original notice is preserved in
[`flashloop_engine/csrc/KIVI_NOTICE`](flashloop_engine/csrc/KIVI_NOTICE).
Do not remove it when redistributing these files.

## External dependencies and model checkpoints

PyTorch and Hugging Face Transformers are installed separately and retain
their respective licenses. Ouro checkpoint weights, tokenizer and remote
modeling code are not distributed here and remain subject to their own
terms. FlashLoop's MIT license does not relicense these dependencies.

The PyTorch reference's sparse attention code follows the cross-loop
cached-mass method used in this project. No copy of the separate Chipmunk
repository is bundled. Methodological attribution is distinct from source
code licensing; preserve any existing source notices.
