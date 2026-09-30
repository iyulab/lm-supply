# LMSupply.Core

Core components shared across all LMSupply libraries.

## Features

- **ExecutionProvider**: Unified GPU/CPU selection (Auto, CUDA, DirectML, CoreML, CPU)
- **HuggingFaceDownloader**: Model downloading with HuggingFace Hub standard caching
- **DownloadProgress**: Detailed download progress reporting
- **Exception Hierarchy**: Consistent error handling across libraries

## Usage

This package is typically consumed as a dependency by other LMSupply packages:

- `LMSupply.Embedder`
- `LMSupply.Reranker`
- `LMSupply.Generator`
- `LMSupply.Transcriber`
- etc.

## Cache Location

Models follow the HuggingFace Hub standard (`CacheManager`):
- `~/.cache/huggingface/hub` (default)
- `HF_HUB_CACHE` environment variable (override)

Lookups (`ModelFileExists`, `GetModelFilePath`, `FindSnapshotDirectory`, `GetMissingFiles`) read the snapshot a
`refs/<revision>` entry names first — what other Hugging Face tools write — and then `snapshots/<revision>/`, where
downloads by LMSupply are written (`GetModelDirectory`). `GetSnapshotDirectories` returns both, in that order.

Non-HF artifacts (ONNX runtime packages, llama-server builds) live outside any hub,
under a single LMSupply root (`LMSupplyCachePaths`):
- `%LOCALAPPDATA%/LMSupply/cache` (default)
- `LMSUPPLY_CACHE_DIR` environment variable (override)
