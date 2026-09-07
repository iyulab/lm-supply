# LMSupply.Llama

Shared llama-server management for GGUF model support in LMSupply.

## Overview

`LMSupply.Llama` provides centralized management of [llama-server](https://github.com/ggml-org/llama.cpp) for GGUF model support across LMSupply packages (Generator, Embedder, Reranker).

This package follows LMSupply's on-demand philosophy: native binaries are downloaded only when first needed, with automatic GPU backend detection and fallback.

## Architecture

LMSupply.Llama uses llama-server as a unified backend for all GGUF operations:

```
LMSupply.Llama/Server/
├── LlamaServerDownloader.cs   - Downloads binaries from GitHub releases
├── LlamaServerPool.cs         - Manages server instance pooling
├── LlamaServerProcess.cs      - Server process lifecycle management
├── LlamaServerClient.cs       - HTTP client for API calls
└── LlamaServerStateManager.cs - Server state and health tracking
```

### Server Modes

llama-server operates in different modes depending on the use case:

| Mode | Flag | Use Case | Consumer Package |
|------|------|----------|------------------|
| Generation | (default) | Text generation | LMSupply.Generator |
| Embedding | `--embedding` | Vector embeddings | LMSupply.Embedder |
| Reranking | `--embedding --pooling rank` | Document reranking | LMSupply.Reranker |

## Features

- **On-demand binary download**: Native binaries downloaded on first use from GitHub releases
- **Automatic backend selection**: Detects hardware and selects optimal GPU backend
- **Fallback chain**: Automatically falls back to CPU if GPU backends fail
- **Cross-platform**: Windows, Linux, macOS (including Apple Silicon)
- **Server pooling**: Efficient server instance reuse across multiple requests
- **Mode-aware pooling**: Separate server instances for generation, embedding, and reranking

## Supported Backends

| Backend | Platform | Hardware |
|---------|----------|----------|
| `Cuda12/13` | Windows/Linux | NVIDIA GPU with CUDA 12.x/13.x |
| `Vulkan` | Windows/Linux | AMD/Intel/NVIDIA GPU |
| `Metal` | macOS | Apple Silicon (M1/M2/M3/M4) |
| `Hip` | Windows/Linux | AMD ROCm |
| `Cpu` | All | CPU with AVX2/AVX512 optimization |

## Backend Selection

Under `ExecutionProvider.Auto` the backend is chosen by `LlamaBackendSelector` (one shared
implementation used by the generator, embedder, and reranker). Vendor decides the GPU backend:

```
macOS (Apple):       Metal
NVIDIA GPU:          CUDA 12
AMD GPU:             Hip (Linux) / Vulkan (Windows)
Intel GPU:           Vulkan (modern: Iris/Arc/Xe/UHD)  |  CPU (legacy HD Graphics)
Other DirectML GPU:  Vulkan
No GPU:              CPU
```

**VRAM-budget gate.** A dedicated-VRAM GPU backend is demoted to **CPU** when the VRAM budget is too
small for any meaningful offload (`< LlamaBackendSelector.MinVramForGpuOffloadBytes`, 2 GB) — this
avoids downloading/initializing a GPU binary that would offload zero layers. **Integrated GPUs**
(Intel Iris Xe, AMD APUs — detected via `GpuInfo.IsIntegrated`) are routed to CPU regardless of their
unreliable reported dedicated-VRAM size. Metal is exempt (Apple unified memory). An explicit GPU pin
(`Cuda`/`DirectML`/`CoreML`) is honored and never demoted; set `LMSUPPLY_VRAM_BUDGET_MB` to force a
budget and keep a GPU backend on an integrated/low-VRAM machine.

## Usage

### Generator (Text Generation)

Generator GGUF models use llama-server in default generation mode:

```csharp
using LMSupply.Generator;

// llama-server is automatically downloaded and started
await using var model = await LocalGenerator.LoadAsync("gguf:default");

// Server pooling: same model reuses existing server instance
await using var model2 = await LocalGenerator.LoadAsync("gguf:default");

// Generate text
await foreach (var token in model.GenerateAsync("Hello!"))
{
    Console.Write(token);
}
```

### Embedder (Vector Embeddings)

Embedder GGUF models use llama-server with `--embedding` flag:

```csharp
using LMSupply.Embedder;

// llama-server is automatically started in embedding mode
await using var model = await LocalEmbedder.LoadAsync("nomic-ai/nomic-embed-text-v1.5-GGUF");

// Generate embeddings
float[] embedding = await model.EmbedAsync("Hello!");
float[][] embeddings = await model.EmbedAsync(new[] { "Text 1", "Text 2" });
```

### Reranker (Document Ranking)

Reranker GGUF models use llama-server with `--embedding --pooling rank`:

```csharp
using LMSupply.Reranker;

// llama-server is automatically started in reranking mode
await using var model = await LocalReranker.LoadAsync("BAAI/bge-reranker-v2-m3-GGUF");

// Rerank documents
var results = await model.RerankAsync(
    "What is machine learning?",
    new[] { "ML is a subset of AI...", "Weather is sunny today..." }
);
```

## Server Pooling

The llama-server backend uses intelligent server pooling:

- **Shared instances**: Same model + mode reuses existing server
- **Mode isolation**: Generation, embedding, and reranking use separate servers
- **Automatic cleanup**: Idle servers are stopped after timeout
- **Health monitoring**: Unhealthy servers are restarted
- **Resource management**: Memory-aware server allocation

Pool key format: `{modelPath}|{backend}|{contextSize}|{mode}`

## Advanced Configuration

### LlamaOptions (Generator)

`LlamaOptions` provides fine-grained control over llama-server behavior:

```csharp
using LMSupply.Generator;

var options = new GeneratorOptions
{
    LlamaOptions = new LlamaOptions
    {
        // GPU layers (-1 = all, 0 = CPU only)
        GpuLayerCount = -1,

        // KV cache quantization (reduces VRAM usage by 50-75%)
        TypeK = KvCacheQuantizationType.Q8_0,
        TypeV = KvCacheQuantizationType.Q8_0,

        // Memory options
        UseMemoryMap = true,
        UseMemoryLock = false,

        // Multi-GPU
        MainGpu = 0,

        // Performance
        FlashAttention = true,
        BatchSize = 2048,
        UBatchSize = 512
    }
};

await using var model = await LocalGenerator.LoadAsync("gguf:default", options);
```

### KV Cache Quantization

Quantizing the KV cache can significantly reduce VRAM usage:

| Type | Memory Savings | Quality Impact |
|------|----------------|----------------|
| `F16` (default) | 0% | None |
| `Q8_0` | ~50% | Minimal |
| `Q4_0` | ~75% | Noticeable |
| `F32` | -100% (doubles) | Maximum quality |

### Sampling Parameters

`GenerationOptions` provides comprehensive sampling control:

```csharp
var genOpts = new GenerationOptions
{
    Temperature = 0.7f,
    TopP = 0.9f,
    TopK = 50,
    MinP = 0.05f,
    RepetitionPenalty = 1.1f,
    FrequencyPenalty = 0.0f,
    PresencePenalty = 0.0f,
    Seed = 42
};

await foreach (var token in model.GenerateAsync("Hello!", genOpts))
{
    Console.Write(token);
}
```

### Grammar Constraints

Constrain output to match specific patterns:

```csharp
// GBNF grammar for yes/no answers
var options = new GenerationOptions
{
    Grammar = "root ::= (\"yes\" | \"no\")"
};

// JSON schema constraint
var jsonOptions = new GenerationOptions
{
    JsonSchema = """
    {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"}
        },
        "required": ["name", "age"]
    }
    """
};
```

`JsonSchema` is a JSON **string**. It is parsed to an object and sent to llama-server via the
OpenAI-compatible `response_format` (chat endpoint) or the native root `json_schema` field (raw
completion) — never as a quoted string, which the server rejects with HTTP 400. Notes:

- **`Grammar` and `JsonSchema` are mutually exclusive.** llama-server rejects a request that carries
  both, so setting both throws `ArgumentException` at request time.
- An unparseable `JsonSchema` string throws `ArgumentException` (with the JSON error) instead of
  producing an opaque HTTP 400.
- The ONNX backend does not constrain output to the schema.

### Hardware-Optimized Defaults

`LlamaOptions.GetOptimalForHardware()` automatically configures based on your system:

| Hardware Tier | GPU Layers | Batch Size | KV Cache | Flash Attention |
|---------------|------------|------------|----------|-----------------|
| Ultra (32GB+ VRAM) | All | 4096 | Q8_0 | Yes |
| High (12-32GB VRAM) | All | 2048 | Q8_0 | Yes |
| Medium (6-12GB VRAM) | All | 1024 | Q4_0 | No |
| Low (< 6GB or CPU) | 0 | 512 | F16 | No |

```csharp
var options = new GeneratorOptions
{
    LlamaOptions = LlamaOptions.GetOptimalForHardware()
};
```

## Consumer Packages

The following packages use LMSupply.Llama for GGUF support:

| Package | Server Mode | Use Case |
|---------|-------------|----------|
| **LMSupply.Generator** | Generation | GGUF language models (Llama, Qwen, etc.) |
| **LMSupply.Embedder** | Embedding | GGUF embedding models (nomic-embed, etc.) |
| **LMSupply.Reranker** | Reranking | GGUF reranker models (bge-reranker, etc.) |

## Troubleshooting

### Server download failed

- Check network access to GitHub releases
- Verify cache directory permissions
- Try setting `HF_HUB_OFFLINE=0` to force online mode

### Server won't start

- Check if port is already in use (servers use random available ports)
- Verify model file exists and is valid GGUF
- Check system logs for GPU driver issues

### `Gemma4Assistant requires ctx_other to be set` on load

This means a speculative-decoding draft/assistant companion file (an `mtp-*`-prefixed GGUF) was
loaded as the main model instead of a standalone chat model — those files are not usable on their
own and this specific architecture crashes llama.cpp's automatic memory fitting on load (a
confirmed, unresolved upstream bug: `ggml-org/llama.cpp#24343`). `GgufModelDownloader` excludes
`mmproj-*`/`mtp-*`/`dflash-*` companion files from selection (`IsCompanionFile`) as of v0.40.2 —
if you hit this on a registry alias, the registry's `DefaultFile` may not match what the backing
HuggingFace repo actually publishes; verify against the repo's file listing before reporting.

### GPU not detected

- Install appropriate GPU drivers
- For CUDA: Ensure NVIDIA drivers are installed
- For Vulkan: Install Vulkan runtime
- For Metal: macOS 11+ required

### Force CPU backend

```csharp
var options = new GeneratorOptions { Provider = ExecutionProvider.Cpu };
await using var model = await LocalGenerator.LoadAsync("gguf:default", options);
```

### Clear Cache

Delete cached binaries to force re-download:

```bash
# Windows
del /s /q %LOCALAPPDATA%\LMSupply\cache\llama-server

# Linux/macOS
rm -rf ~/.local/share/LMSupply/cache/llama-server
```

## Version Information

- **llama-server**: Downloaded from [llama.cpp GitHub releases](https://github.com/ggml-org/llama.cpp/releases)
- Binaries are versioned and cached by build number (tag format: `b<NNNN>`, e.g., `b8672`)
- **"Latest" resolution** (v0.58.0+): llama.cpp's stable line is its versioned releases (`vX.Y.Z`), whose
  single asset `nightly-tag.txt` names the build they were cut from; the per-commit `b<NNNN>` build
  releases carry the binaries and are marked prerelease. By default `LlamaServerDownloader` reads
  `releases/latest`, follows `nightly-tag.txt` to the build it names, and downloads that build. If the
  pointer is missing it falls back to the newest build release with assets. Set
  `LlamaServerUpdateOptions.IncludePrerelease = true` to follow the newest build release directly.
  Whichever path is taken, the resolved version is always a build tag — `PinnedVersion` may name either
  form (`"b10809"` or `"v0.4.0"`) and is normalized the same way.

### Minimum Version Requirements

`LlamaServerVersionRequirements` validates that the installed llama-server is new enough for the model architecture being loaded. Requirements are keyed by chat format name and checked at model load time.

| Chat Format | Minimum Build | Reason |
|-------------|---------------|--------|
| `gemma4` | `b8672` | Native Gemma 4 architecture support (GGUF metadata auto-detect) |

#### Auto-upgrade retry (v0.34.1+)

Before throwing, `LlamaServerGeneratorModel.CreateAsync` calls `CheckAndApplyUpdateAsync` automatically when the cached binary fails the minimum-build check. This means **you no longer need to manually delete the cache** in most cases:

1. LoadAsync detects cached version is too old via `LlamaServerVersionRequirements.MeetsMinimum(serverVersion, chatFormatName)`
2. `CheckAndApplyUpdateAsync` downloads the latest release and replaces the cached binary
3. `Validate` is then called on the new version — throws `InvalidOperationException` only if the upgrade itself fails or the new version still doesn't meet the requirement

```
llama-server b7898 is too old for gemma4 models. Minimum required: b8672.
Delete the cached llama-server to trigger a fresh download of the latest version.
```

The error message above is only reached if the auto-upgrade path also fails. Validation degrades gracefully — if the version string can't be parsed (non-standard builds), model loading proceeds without error.

**With a pinned version** (`GeneratorOptions.ServerUpdateOptions.PinnedVersion`, see [README](../src/LMSupply.Generator/README.md#pinning-or-pre-provisioning-the-llama-server-binary)), the auto-upgrade retry above is intentionally a no-op — a pin must never be silently exceeded. If the pinned build is too old for the model, load fails with the same error; the fix is to pin a newer build, not to delete the cache (deleting it just re-downloads the same pinned version).

#### Public API

`LlamaServerVersionRequirements` exposes three public members:

| Member | Description |
|--------|-------------|
| `ParseBuildNumber(string? version)` | Parses the build number from a version tag (e.g., `"b8672"` → `8672`). Returns `null` if unparseable. |
| `GetMinimumBuild(string chatFormatName)` | Returns the minimum build number for a given chat format name, or `null` if none required. |
| `MeetsMinimum(string? serverVersion, string chatFormatName)` | Returns `true` if the server version satisfies the minimum for the format. Returns `true` on unparseable version (graceful degradation). |
| `Validate(string? serverVersion, string chatFormatName)` | Throws `InvalidOperationException` if the version is too old; silently passes on unparseable version. |

```csharp
// Check programmatically before loading
bool ok = LlamaServerVersionRequirements.MeetsMinimum(serverVersion, "gemma4");
if (!ok)
{
    Console.WriteLine("Cached llama-server is too old; will auto-upgrade on next LoadAsync.");
}
```

## Split GGUF Downloads

Some large models are distributed as multiple shards using the `-NNNNN-of-NNNNN.gguf` naming convention (e.g., Qwen 3.5 122B comes as 3 parts in a `Q4_K_M/` subfolder). The `GgufModelDownloader` handles this automatically:

- Models in the registry declare `ShardCount` and point `DefaultFile` at the first shard
- All shards are fetched sequentially with progress reporting per shard
- llama-server is started with the first shard path and auto-loads the remaining parts from the same directory
