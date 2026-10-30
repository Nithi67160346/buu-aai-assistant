# Chat model

The application refers to the local Ollama tag `aai-qwen-1.7b` in both
`rag_pi4.py` and `hybrid_engine.py`. The tag name is not proof of model size.

Read-only inspection of the original Pi on 2026-10-30 reported:

- Architecture: `qwen3`
- Parameters reported by Ollama: `2.0B`
- Quantization: `Q4_K_M`
- Context length reported by Ollama: `40960`
- Weight blob SHA256: `3d0b790534fe4b79525fc3692950408dca41171676ed7e21db57af5c65ef6ab6`

Weights are not bundled. To reproduce the same model, obtain the original weight
blob from the model owner and verify its SHA256. Save it as
`models/aai-qwen-1.7b.gguf`, then run `ollama create aai-qwen-1.7b -f Modelfile.example`
from the project root. The example uses the template and numeric sampling parameters
observed in the original model. This is not a distribution of model weights or their license.

The historical `Modelfile_aai` referenced a missing 0.5B GGUF and is deliberately
excluded from this package because it does not describe the deployed model.
