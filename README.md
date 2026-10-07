# Orch Neural Network

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23142805.svg)](https://doi.org/10.5281/zenodo.23142805)

An iOS app that runs a real GPT-2 (DistilGPT2, 82M parameters) on device and visualises every step of
next-word prediction: tokens, embeddings, attention heads, MLP neurons, the logit lens through all six
layers, the final probabilities, and every weight matrix as a heatmap. Framed as "how your keyboard's
suggestions work".

## Layout
- `project.yml` — xcodegen spec. Run `xcodegen generate` to (re)create `OrchNeuralNetwork.xcodeproj`.
- `OrchNeuralNetwork/Engine/` — `WeightFile` (mmap'd fp16 container), `BPETokenizer` (byte-level BPE),
  `GPT2Model` (forward pass on Accelerate with full tracing), `InferenceEngine` (debounce + suggestions).
- `OrchNeuralNetwork/Views/` — SwiftUI screens: Predict, Inside, Weights, Learn, About.
- `Model/` — `distilgpt2.tsw` (164 MB, fp16), `vocab.json`, `merges.txt`, `reference.json`.
- `Tools/convert.py` — converts HF safetensors to the `.tsw` container. `Tools/reference.py` — independent
  numpy forward pass used to generate `reference.json`.
- `OrchNeuralNetworkTests/` — asserts the Swift forward pass matches the numpy reference.
- `APPSTORE.md` — listing copy, review notes, screenshot plan, upload commands.

## Build
```bash
xcodegen generate
xcodebuild -project OrchNeuralNetwork.xcodeproj -scheme OrchNeuralNetwork \
  -destination 'platform=iOS Simulator,name=iPhone 18 Pro' test
```

## Weight file format (`TSW1`)
`"TSW1"` · UInt32 LE header length · JSON header (config + tensor table) · pad to 64 bytes · fp16 tensors,
each 64-byte aligned, HF Conv1D layout `[in, out]` preserved. Regenerate with
`~/.venvs/mflux/bin/python Tools/convert.py`.

## Paper and benchmarks
`paper/orch-neural-network.tex` is a technical report on the engine (verification against the numpy oracle,
float16 fidelity, latency/memory, and an analytic cost model). Every number in it comes from
`results/macros.tex`, which `scripts/make_figures.py` writes from `results/bench_*.json` and `results/fidelity.json`.

- `scripts/run_benchmark.sh` runs `BenchmarkTests` on the simulator (default) or a device
  (`scripts/run_benchmark.sh "id=<udid>" results/bench_iphone17.json`) and saves the JSON. The app also accepts
  the launch argument `-benchmark`, which writes `Documents/benchmark.json` on a physical phone.
- `scripts/fidelity.py` compares the float16 `.tsw` weights with the original float32 release over 30 prompts.
- `scripts/make_figures.py` regenerates `figures/*.pdf` and `results/macros.tex`; then `cd paper && latexmk -pdf orch-neural-network.tex`.
- `scripts/make_arxiv.sh` builds a flat arXiv submission folder in `paper/arxiv/`.

## Licence
App code © 2026 Wonmo (John) Seong. Model weights: DistilGPT2, Apache 2.0, https://huggingface.co/distilgpt2.

## Licence and citation
Code is MIT licensed (see LICENSE); model weights are DistilGPT2 under Apache 2.0. If you use or build on this work, please cite it: see `CITATION.cff` (GitHub's "Cite this repository" button).

## Acknowledgement
Conceived, designed and built by the author alone; AI-assisted software tools were used in preparing the code. The author takes full responsibility for the design and content.
