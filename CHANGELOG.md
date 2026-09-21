# Changelog

All notable changes to this project are documented in this file. Versions follow
[Semantic Versioning](https://semver.org/); while the major version is 0, a minor release may contain
breaking changes, and each one is marked **Breaking** with a migration note.

## [0.69.0] - 2026-09-21

### Added

- **`LMSupply.Reranker`: the alias `multilingual-fast`** loads bge-reranker-v2-m3 — the model behind
  `multilingual` — as a Q4_K_M GGUF through llama-server: about 440 MB instead of 2.3 GB and a fraction
  of the CPU latency, with the same 0..1 scores. It works on `LoadAsync`, `IsModelDownloaded` and
  `DownloadModelAsync`, accepts a quantization qualifier (`multilingual-fast:Q8_0`), and a user alias of
  the same name overrides it. `multilingual` and `auto` still resolve to ONNX: nothing starts a
  llama-server process unless you ask for this alias or a `gguf:` id.

### Changed

- **Breaking — `LMSupply.Reranker`: a GGUF reranker now scores on the same 0..1 scale as an ONNX one.**
  The llama-server route returned the classifier's raw logit (e.g. `2.8`, and negative for most
  candidates) where `RankedResult.Score` documents a relevance between 0 and 1 and the ONNX route has
  always delivered one. The logit now goes through the same sigmoid. Ranking is unchanged (the mapping
  is monotonic); a threshold you calibrated on the ONNX scale now means the same thing on a `gguf:`
  model, and a downstream "drop scores below 0" default no longer empties the result list.
  Migration: if you compared GGUF scores against a logit threshold `t`, compare against
  `1 / (1 + exp(-t))` instead.

### Fixed

- **`LocalReranker.IsModelDownloaded` answers for GGUF models.** It had no GGUF branch, so a `gguf:`
  model that was cached and loaded fine was always reported as not downloaded. It now looks where the
  GGUF loader looks and picks the file the loader picks — `true` exactly when a load with
  `DisableAutoDownload = true` would find its file. Git LFS pointers are not counted as a model.
- **`LocalReranker.DownloadModelAsync` fetches a GGUF model the way the loader does** — one file, into
  the loader's cache layout. It used to fall through to the raw-repository download, which pulls every
  quantization of the repository into a layout the GGUF loader never reads.
- **An interrupted GGUF reranker download no longer leaves a truncated model in the cache.** The
  reranker's GGUF downloader wrote straight to the final file name; it now uses the same resumable
  download as the embedder and generator (`.part` until complete, length checked against the
  repository listing), so a cached file of the wrong length is fetched again instead of being loaded.
- **`RerankerOptions.QuantizationHint` is honoured on the GGUF route** (e.g. `"Q8_0"`). It was read by
  nothing there; the route always asked for `Q4_K_M`, which stays the default.
- **Reranker model sizes and languages are stated as they are.** `quality` (bge-reranker-base) declared
  440 MB; the fp32 ONNX it downloads is 1.1 GB — `ModelInfo.SizeBytes`, and with it the pool's memory
  estimate, were off by 2.5×. The package README listed `large` and `multilingual` at half their download
  size. `quality` and `large` were described as multilingual; their model cards list English and Chinese,
  and the docs no longer recommend them for other languages.
- The GGUF error message, XML docs and `docs/reranker.md` named `BAAI/bge-reranker-v2-m3-GGUF` and
  `jinaai/...-GGUF`, neither of which exists on HuggingFace. They now name `gpustack/` repositories
  that do. The docs also said the largest quantization that fits in memory is chosen; the route asks
  for `Q4_K_M` (or your hint) first and only falls back to that rule.

## [0.68.3] - 2026-09-19

This file starts at 0.68.3. Changes in earlier releases were not recorded here; the commit history is
the record for them.
