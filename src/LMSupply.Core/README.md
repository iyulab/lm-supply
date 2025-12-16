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

Follows HuggingFace Hub standard:
- `~/.cache/huggingface/hub` (default)
- `HF_HUB_CACHE` environment variable (override)
