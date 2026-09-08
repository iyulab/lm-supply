# LMSupply.Generator

Local text generation and chat with ONNX Runtime GenAI and GGUF (llama-server) support.

## Installation

```bash
dotnet add package LMSupply.Generator
```

## Quick Start

### Simple Text Generation

```csharp
using LMSupply.Generator;

// Using the builder pattern
var generator = await TextGeneratorBuilder.Create()
    .WithDefaultModel()        // Platform-aware: GGUF on NVIDIA/CPU/Mac/Linux/integrated-GPU, Phi-4 Mini ONNX on discrete DirectML+non-NVIDIA
    .BuildAsync();

// Generate text
string response = await generator.GenerateCompleteAsync("What is machine learning?");
Console.WriteLine(response);

await generator.DisposeAsync();
```

### Chat Completion

```csharp
using LMSupply.Generator;
using LMSupply.Generator.Models;

var generator = await TextGeneratorBuilder.Create()
    .WithDefaultModel()  // see "Default and auto selection" below
    .BuildAsync();

// Chat format
var messages = new[]
{
    new ChatMessage(ChatRole.System, "You are a helpful assistant."),
    new ChatMessage(ChatRole.User, "Explain quantum computing in simple terms.")
};

string response = await generator.GenerateChatCompleteAsync(messages);
Console.WriteLine(response);
```

### Streaming Generation

```csharp
await foreach (var token in generator.GenerateAsync("Write a short story about a robot:"))
{
    Console.Write(token);
}
```

## Default and auto selection

`LocalGenerator.LoadAsync("default")` and `LocalGenerator.LoadAsync("auto")` both delegate to a hardware-aware selection:

| Platform | Backend | Model |
|----------|---------|-------|
| NVIDIA GPU (any OS) | GGUF (llama.cpp CUDA) | Qwen3 via `gguf:auto` (VRAM-aware) |
| Apple Silicon | GGUF (llama.cpp Metal) | Qwen3 via `gguf:auto` |
| CPU-only / integrated GPU (any OS) | GGUF (llama.cpp CPU) | Qwen3 via `gguf:auto` (RAM-aware) |
| Linux + discrete GPU | GGUF (llama.cpp) | Qwen3 via `gguf:auto` |
| Windows + **discrete** AMD/Intel GPU (Arc, Radeon) | ONNX (DirectML) | Phi-4 Mini (FC-capable, MIT) |

`gguf:auto` selects the largest Qwen3 model (`qwen3-fast/default/balanced/quality` pool) that fits the
VRAM budget, or — when VRAM is insufficient — the largest that fits the system RAM budget (CPU). On an
integrated GPU the llama.cpp backend is CPU (see [llama.md](llama.md#backend-selection)).

After the family is chosen, the download step picks the **quantization that fits** the backend-consistent
budget: a capable host keeps the registry default quant (e.g. `Q4_K_M`), a tight-memory host downscales
(`Q4 → Q3 → Q2`) so it loads instead of OOMing; if nothing fits, the smallest is used with a warning.

### Explicit model selection

```csharp
// Pin a specific ONNX model (DirectML + non-NVIDIA users)
await using var onnx = await LocalGenerator.LoadAsync("microsoft/Phi-4-mini-instruct-onnx");
await using var onnxAlias = await LocalGenerator.LoadAsync("phi-4-mini");

// Pin a specific GGUF model
await using var gguf = await LocalGenerator.LoadAsync("gguf:gemma4-default"); // Gemma 4 E4B
await using var ggufXL = await LocalGenerator.LoadAsync("gguf:gemma4-large"); // Gemma 4 31B

// Let the hardware decide (GGUF on most platforms, ONNX on DirectML+non-NVIDIA)
await using var auto = await LocalGenerator.LoadAsync("auto");
await using var def  = await LocalGenerator.LoadAsync("default"); // same as "auto"
```

## Model Selection

### Preset Models

```csharp
// Default: Phi-4 Mini (balanced, MIT license)
.WithDefaultModel()

// Or use presets
.WithModel(GeneratorModelPreset.Default)   // Phi-4 Mini
.WithModel(GeneratorModelPreset.Fast)      // Phi-4 Mini (smallest FC-capable)
.WithModel(GeneratorModelPreset.Quality)   // Phi-4
```

### HuggingFace Models

```csharp
// Use any ONNX model from HuggingFace
.WithHuggingFaceModel("microsoft/Phi-4-mini-instruct-onnx")
```

### Local Models

```csharp
// Use a local model directory
.WithModelPath("C:/models/my-model-onnx")
```

### Downloading ahead of time

`LocalGenerator.DownloadModelAsync` pulls exactly the files `LoadAsync` would open — same alias
resolution, same `default`/`auto` hardware selection, same quantization pick — and returns their
local path without loading the model, provisioning a runtime or starting llama-server. Use it from
an installer, a first-run screen or a CI cache-warming step; the later `LoadAsync` then downloads
nothing.

```csharp
var options = new GeneratorOptions { Provider = ExecutionProvider.Auto };
var path = await LocalGenerator.DownloadModelAsync("gguf:gemma4-default", options,
    progress: new Progress<DownloadProgress>(p => Console.WriteLine(p.Phase)));
// later, same id + options:
await using var model = await LocalGenerator.LoadAsync("gguf:gemma4-default", options);
```

Pass the same `Provider` both times: GGUF auto-quantization picks the file from the provider's
memory budget, so a different provider can warm a different file. Local paths are returned as-is.

## Configuration Options

### Execution Provider

```csharp
var generator = await TextGeneratorBuilder.Create()
    .WithDefaultModel()
    .WithProvider(ExecutionProvider.Auto)      // Auto-detect best provider
    .WithProvider(ExecutionProvider.Cuda)      // NVIDIA GPU
    .WithProvider(ExecutionProvider.DirectML)  // Windows GPU (AMD, Intel, NVIDIA)
    .WithProvider(ExecutionProvider.CoreML)    // macOS Apple Silicon
    .WithProvider(ExecutionProvider.Cpu)       // CPU only
    .BuildAsync();
```

### Generation Options

```csharp
var options = new GenerationOptions
{
    MaxTokens = 512,              // Maximum tokens to generate
    Temperature = 0.7f,           // Randomness (0.0 = deterministic)
    TopP = 0.9f,                  // Nucleus sampling
    TopK = 50,                    // Top-K sampling
    RepetitionPenalty = 1.1f,     // Discourage repetition
    DoSample = true               // Enable sampling (vs greedy)
};

string response = await generator.GenerateCompleteAsync(prompt, options);

// Or use presets
string creative = await generator.GenerateCompleteAsync(prompt, GenerationOptions.Creative);
string precise = await generator.GenerateCompleteAsync(prompt, GenerationOptions.Precise);
```

### Memory Management

```csharp
// Limit memory usage
var generator = await TextGeneratorBuilder.Create()
    .WithDefaultModel()
    .WithMemoryLimit(8.0)    // 8GB limit
    .BuildAsync();

// Or with detailed options
var memoryOptions = new MemoryAwareOptions
{
    MaxMemoryBytes = 8L * 1024 * 1024 * 1024,  // 8GB
    WarningThreshold = 0.80,                    // GC at 80%
    CriticalThreshold = 0.95,                   // Fail at 95%
    AutoGcOnWarning = true
};

var generator = await TextGeneratorBuilder.Create()
    .WithDefaultModel()
    .WithMemoryManagement(memoryOptions)
    .BuildAsync();
```

## Hardware Detection

```csharp
using LMSupply.Generator;

// Get hardware recommendations
var recommendation = HardwareDetector.GetRecommendation();

Console.WriteLine(recommendation.GetSummary());
// Output:
// Hardware: NVIDIA RTX 4090 (24.0GB)
// System Memory: 64.0GB
// Provider: Cuda
// Quantization: FP16
// Max Context: 16384
// Recommended Models: microsoft/Phi-4-mini-instruct-onnx, microsoft/phi-4-onnx

// Auto-select best provider
var provider = HardwareDetector.GetBestProvider();
```

## Speculative Decoding

Speed up generation by using a smaller draft model:

```csharp
using LMSupply.Generator;

// Create draft (small/fast) and target (large/accurate) models
var draftModel = await TextGeneratorBuilder.Create()
    .WithModel(GeneratorModelPreset.Fast)
    .BuildAsync();

var targetModel = await TextGeneratorBuilder.Create()
    .WithModel(GeneratorModelPreset.Quality)
    .BuildAsync();

// Create speculative decoder
var decoder = SpeculativeDecoderBuilder.Create()
    .WithDraftModel(draftModel)
    .WithTargetModel(targetModel)
    .WithSpeculationLength(5)
    .WithAdaptiveSpeculation(true)
    .Build();

// Generate with speculative decoding
var result = await decoder.GenerateCompleteAsync("Explain neural networks:");

Console.WriteLine(result.Text);
Console.WriteLine(result.Stats.GetSummary());
// Output:
// Total Tokens: 256
// Draft/Accepted: 200/180 (90.0%)
// Target Tokens: 76
// Throughput: 45.2 tok/s
// Time: 5667ms
```

## Model Factory

For advanced scenarios with multiple models:

```csharp
using LMSupply.Generator;

using var factory = new OnnxGeneratorModelFactory();

// Check if model is available locally
if (!factory.IsModelAvailable("microsoft/Phi-4-mini-instruct-onnx"))
{
    // Download model
    await factory.DownloadModelAsync(
        "microsoft/Phi-4-mini-instruct-onnx",
        progress: new Progress<double>(p => Console.WriteLine($"Downloading: {p:P0}"))
    );
}

// Create model instance
var model = await factory.CreateAsync("microsoft/Phi-4-mini-instruct-onnx");

// List available models
foreach (var modelId in factory.GetAvailableModels())
{
    Console.WriteLine(modelId);
}
```

## Well-Known Models

| Alias | Model | Parameters | License |
|-------|-------|------------|---------|
| Default | Phi-4-mini-instruct | 3.8B | MIT |
| Fast | Phi-4-mini-instruct | 3.8B | MIT |
| Quality | phi-4 | 14B | MIT |

## Chat Formats

The library automatically detects chat format based on model ID:

| Format | Models |
|--------|--------|
| Phi-3 | Phi-3, Phi-3.5, Phi-4 (ONNX) |
| ChatML | Hermes 3, Qwen 3 (GGUF) |
| Mistral Nemo | Ministral, Mistral Nemo (GGUF) |

Or specify explicitly:

```csharp
var generator = await TextGeneratorBuilder.Create()
    .WithHuggingFaceModel("my-model")
    .WithChatFormat("phi3")   // phi3, llama3, chatml, gemma
    .BuildAsync();
```

## GPU Support

GPU acceleration is **automatic** — LMSupply detects your hardware and downloads appropriate runtime binaries on first use:

- **NVIDIA CUDA**: Automatically detected and used
- **Windows DirectML**: AMD, Intel, NVIDIA via Direct3D
- **macOS CoreML**: Apple Silicon optimization

No additional packages required. Use `ExecutionProvider.Auto` (default) or force specific provider in options.

## GGUF Model Support

GGUF models are loaded via [llama-server](https://github.com/ggml-org/llama.cpp) (llama.cpp HTTP server), providing access to the vast ecosystem of quantized models on HuggingFace. The llama-server binaries are automatically downloaded and managed.

### Quick Start with GGUF

```csharp
using LMSupply.Generator;

// Load a GGUF model using the "gguf:" prefix
await using var model = await LocalGenerator.LoadAsync("gguf:auto");  // Hardware-optimized (Qwen3 pool)

// Generate text
await foreach (var token in model.GenerateAsync("Hello, my name is"))
{
    Console.Write(token);
}
```

### GGUF Model Aliases

The registry has two primary tiers: **Gemma 4** (multimodal, Apache 2.0) for explicit use, and **Qwen3/3.5/3.6** as the `gguf:auto` selection pool.

**Gemma 4 aliases** — requires llama.cpp **b8672+** (auto-validated at load time):

| Alias | Model | Parameters | Quant | Weights | KV @ 4k | Total | Use Case |
|-------|-------|------------|-------|---------|---------|-------|----------|
| `gguf:gemma4-fast` | Gemma 4 E2B Instruct | 2.3B | Q4_K_M | ~3.1 GB | ~1.0 GB | ~4.1 GB | 4-6GB VRAM, iGPU/mobile |
| `gguf:gemma4-default` | Gemma 4 E4B Instruct | 4.5B | Q4_K_M | ~5.3 GB | ~1.4 GB | ~6.7 GB | 8GB VRAM |
| `gguf:gemma4-balanced` | Gemma 4 E4B Instruct | 4.5B | Q8_0 | ~7.5 GB | ~1.4 GB | ~8.9 GB | 12GB VRAM (higher quality E4B) |
| `gguf:gemma4-quality` | Gemma 4 26B A4B (MoE) | 26B (4B active) | Q4_K_M | ~16.8 GB | ~2.9 GB | ~19.7 GB | 24GB VRAM |
| `gguf:gemma4-large` | Gemma 4 31B Instruct | 31B (Dense) | Q4_K_M | ~18.7 GB | ~5.1 GB | ~23.8 GB | 32GB+ VRAM |

**Qwen3/3.5/3.6 aliases** — Apache 2.0, ChatML. `gguf:auto` selects from this pool:

| Alias | Model | Parameters | Quant | Weights | KV @ 4k | Total | Notes |
|-------|-------|------------|-------|---------|---------|-------|-------|
| `gguf:auto` | Hardware-optimized (qwen3 pool) | varies | varies | varies | varies | varies | Auto-select by VRAM |
| `gguf:qwen3-fast` | Qwen 3.5 2B Instruct | 2B | Q4_K_M | ~1.5 GB | ~0.75 GB | ~2.25 GB | |
| `gguf:qwen3-default` | Qwen 3.5 4B Instruct | 4B | Q4_K_M | ~3.0 GB | ~1.25 GB | ~4.25 GB | thinking ON by default |
| `gguf:qwen3-balanced` | Qwen3 8B Instruct | 8B | Q4_K_M | ~5.0 GB | ~2.25 GB | ~7.25 GB | |
| `gguf:qwen3-quality` | Qwen 3.6 35B A3B (IQ4_XS, MoE) | 35B (3B active) | IQ4_XS | ~17.7 GB | ~1.3 GB | ~19.0 GB | thinking ON; auto-pool |
| `gguf:qwen3-large` | Qwen 3.6 35B A3B (Q4_K_M, MoE) | 35B (3B active) | Q4_K_M | ~22.1 GB | ~1.3 GB | ~23.4 GB | thinking ON; auto-pool excluded |

**Other aliases:**

| Alias | Model | Parameters | Quant | Size | Notes |
|-------|-------|------------|-------|------|-------|
| `gguf:phi-4-mini` | Phi-4 Mini Instruct | 3.8B | Q4_K_M | ~2.4 GB | Phi3 chat format; strong KO/EN |
| `gguf:qwen2.5-7b` | Qwen 2.5 7B Instruct | 7.6B | Q4_K_M | ~4.7 GB | ChatML; reliable tool calling |
| `gguf:xlarge` | Qwen 3.5 122B A10B (MoE, split) | 122B (10B active) | Q4_K_M | ~76.5 GB (3 shards) | 96GB+ server |

> **KV cache footprint is included in auto-selection.** The `KV @ 4k` column shows the FP16 KV cache size at the default 4096 budget context length. llama-server reserves the full `--ctx-size` KV cache at load time, so a model that fits in weights but not in weights + KV will OOM at runtime. Use `LlamaOptions.TypeK = TypeV = KvCacheQuantizationType.Q8_0` (or `Q4_0`) to halve/quarter KV memory at the cost of slight quality loss.

> **Split GGUF support**: `gguf:xlarge` is distributed as 3 shards (`-00001-of-00003`, etc.) in a `Q4_K_M/` subfolder. The downloader automatically fetches all shards; llama-server auto-loads the remaining parts when given the first shard path.

#### Hardware-Optimized Selection (`gguf:auto`)

`gguf:auto` selects the largest model whose **weights + KV cache** fit in the available VRAM budget. Budget = `freeVRAM × (1 − safetyMargin)` where the safety margin is platform-aware:

| Platform | Safety margin | Reason |
|----------|---------------|--------|
| Windows + NVIDIA + total VRAM ≤ 6GB | 25% | Compositor + driver overhead is proportionally larger on small dedicated cards (e.g., RTX 4060 Laptop 4GB) |
| Everything else | 15% | Default for desktop / server / Apple Silicon / Linux |

KV cache is estimated at the default budget context length (4096 tokens, FP16), see `GgufModelRegistry.DefaultBudgetContextLength`.

Auto-selection pool: `qwen3-fast`, `qwen3-default`, `qwen3-balanced`, `qwen3-quality` (`qwen3-large` is excluded — exceeds 24 GB × 85% budget). Models with `ThinkingEnabledByDefault` generate reasoning before answering. To stop the model from thinking at all, set `Thinking = ThinkingMode.Off` (forwards `enable_thinking=false` to the chat template — no reasoning tokens generated). To let it think but hide the `<think>...</think>` block from the returned text, set `FilterReasoningTokens = true` (reasoning is still generated). `ThinkingMode.Auto` (default) preserves each model's built-in behavior.

| Free VRAM | Budget | Selected Model | Reason |
|-----------|--------|----------------|--------|
| 0 / CPU only | 0 | `gguf:qwen3-fast` (Qwen 3.5 2B) | FallbackToSmallest — runtime CPU offload |
| 3 GB | ~2.55 GB | `gguf:qwen3-fast` | Fits (~2.25 GB total) |
| 6 GB | ~5.1 GB | `gguf:qwen3-default` (Qwen 3.5 4B) | Fits (~4.25 GB); thinking ON |
| 10 GB | ~8.5 GB | `gguf:qwen3-balanced` (Qwen3 8B) | Fits (~7.25 GB) |
| 24 GB | ~20.4 GB | `gguf:qwen3-quality` (Qwen 3.6 35B MoE) | Fits (~19.0 GB); thinking ON |

> **Low-VRAM laptop guidance.** On Windows laptops with ≤4 GB NVIDIA VRAM (RTX 4050/4060 Laptop, etc.), the auto path will still select `gguf:qwen3-fast` and emit a `FallbackToSmallest` warning to `Trace`. Even the 2B model may exceed budget once Windows compositor + driver reserve their share. For these hosts, prefer the ONNX path explicitly: `LocalGenerator.LoadAsync("phi-4-mini")` (DirectML or CPU). The Trace line `[LocalGenerator.auto] WARNING: ...` indicates this fallback so downstream consumers can intercept it.

```csharp
// Let LMSupply choose the optimal model for your hardware
await using var model = await LocalGenerator.LoadAsync("gguf:auto");
```

### Using HuggingFace GGUF Repositories

Load any GGUF model directly with `owner/repo-name` format:

```csharp
// Load from any GGUF repository (auto-detected by -GGUF suffix)
await using var model = await LocalGenerator.LoadAsync(
    "NousResearch/Hermes-3-Llama-3.1-8B-GGUF");

// Other popular repositories
await using var model = await LocalGenerator.LoadAsync("bartowski/Mistral-Nemo-Instruct-2407-GGUF");
await using var model = await LocalGenerator.LoadAsync("unsloth/Qwen3-32B-GGUF");
await using var model = await LocalGenerator.LoadAsync("mistralai/Ministral-3-3B-Instruct-2512-GGUF");

// Specify a particular quantization file
await using var model = await LocalGenerator.LoadAsync(
    "bartowski/Qwen2.5-7B-Instruct-GGUF",
    new GeneratorOptions { GgufFileName = "Qwen2.5-7B-Instruct-Q5_K_M.gguf" });
```

The system automatically:
- Detects GGUF repositories by `-GGUF` or `_gguf` suffix in repo name
- Selects the optimal quantization file based on available memory (VRAM + RAM), choosing the largest quantization that fits
- Downloads and caches the model for reuse

> **Hardware-aware selection:** LMSupply measures available VRAM and RAM, then picks the highest-quality quantization (e.g., Q8 over Q4) that fits in memory. Use `GeneratorOptions.GgufFileName` to override with a specific file.

### GGUF Configuration Options

```csharp
var options = new GeneratorOptions
{
    // Context length (default: from model metadata)
    MaxContextLength = 4096
};

await using var model = await LocalGenerator.LoadAsync("gguf:gemma4-default", options);
```

### Advanced GGUF Options (LlamaOptions)

For fine-grained control over llama.cpp behavior:

```csharp
var options = new GeneratorOptions
{
    MaxContextLength = 8192,
    LlamaOptions = new LlamaOptions
    {
        // GPU layer offloading (-1 = all on GPU, 0 = CPU only, N = N layers)
        GpuLayerCount = -1,

        // Batch size for prompt processing (default: 512)
        BatchSize = 1024,

        // Physical batch size for memory control (must be <= BatchSize)
        UBatchSize = 512,

        // Enable Flash Attention for better performance (requires compatible GPU)
        FlashAttention = true,

        // KV cache quantization for memory savings
        TypeK = KvCacheQuantizationType.Q8_0,  // ~50% KV cache memory reduction
        TypeV = KvCacheQuantizationType.Q8_0,

        // Memory mapping for faster model loading
        UseMemoryMap = true,

        // Lock model in memory to prevent swapping
        UseMemoryLock = false,

        // RoPE frequency settings for context extension
        RopeFrequencyBase = null,
        RopeFrequencyScale = null,

        // Multi-GPU: select primary GPU (0-based index)
        MainGpu = 0,

        // CPU thread count (default: llama-server's physical-core detection; set explicitly
        // on VMs/containers that under-report topology, e.g. Environment.ProcessorCount)
        Threads = null,

        // Startup wait: stall window restarts on any process activity (stderr line, working-set
        // growth, CPU time); the absolute cap only bounds a start that never converges.
        // Defaults: 120 s stall / 10 min cap — see docs/llama.md "Startup wait".
        StartupStallTimeout = null,
        StartupTimeout = null
    }
};

await using var model = await LocalGenerator.LoadAsync("gguf:gemma4-quality", options);
```

#### KV Cache Quantization

Quantizing the KV (Key-Value) cache significantly reduces memory usage with minimal quality impact:

| Type | Memory Savings | Quality Impact |
|------|----------------|----------------|
| `F16` (default) | 0% | Best |
| `Q8_0` | ~50% | Minimal |
| `Q4_0` | ~75% | Noticeable on long contexts |
| `F32` | -100% (increases) | Identical to F16 |

```csharp
// Example: Large context with aggressive memory optimization
var options = new GeneratorOptions
{
    MaxContextLength = 32768,
    LlamaOptions = new LlamaOptions
    {
        TypeK = KvCacheQuantizationType.Q4_0,
        TypeV = KvCacheQuantizationType.Q4_0,
        BatchSize = 2048,
        UBatchSize = 256
    }
};
```

#### Automatic Hardware Optimization

If `LlamaOptions` is not specified, LMSupply automatically configures optimal settings:

```csharp
// LlamaOptions.GetOptimalForHardware() is called internally
var autoOptions = LlamaOptions.GetOptimalForHardware();
```

| Tier | BatchSize | UBatchSize | FlashAttention | TypeK/TypeV | GpuLayerCount |
|------|-----------|------------|----------------|-------------|---------------|
| Ultra | 4096 | 1024 | true | Q8_0 | -1 (all GPU) |
| High | 2048 | 512 | true | Q8_0 | -1 (all GPU) |
| Medium | 1024 | 512 | false | Q4_0 | -1 (all GPU) |
| Low | 512 | 256 | false | F16 | 0 (CPU only) |

### Performance Tuning Guide

#### Maximum Throughput

For highest tokens/second on capable hardware:

```csharp
var options = new GeneratorOptions
{
    LlamaOptions = new LlamaOptions
    {
        GpuLayerCount = -1,      // All layers on GPU
        BatchSize = 2048,         // Large batch for throughput
        UBatchSize = 512,         // Balanced physical batch
        FlashAttention = true,    // Faster attention computation
        UseMemoryMap = true       // Faster model loading
    }
};
```

#### Minimum Memory Footprint

For systems with limited VRAM:

```csharp
var options = new GeneratorOptions
{
    MaxContextLength = 4096,     // Limit context for memory
    LlamaOptions = new LlamaOptions
    {
        TypeK = KvCacheQuantizationType.Q4_0,  // Aggressive KV quantization
        TypeV = KvCacheQuantizationType.Q4_0,
        BatchSize = 256,          // Smaller batch size
        UBatchSize = 128,
        FlashAttention = false    // May save memory on some GPUs
    }
};
```

#### Long Context Processing

For handling long documents (8K+ tokens):

```csharp
var options = new GeneratorOptions
{
    MaxContextLength = 32768,
    LlamaOptions = new LlamaOptions
    {
        TypeK = KvCacheQuantizationType.Q8_0,  // Balance memory/quality
        TypeV = KvCacheQuantizationType.Q8_0,
        BatchSize = 2048,
        UBatchSize = 512,
        RopeFrequencyBase = 1000000f  // For YaRN-scaled models
    }
};
```

#### CPU-Only Systems

For systems without GPU acceleration:

```csharp
var options = new GeneratorOptions
{
    Provider = ExecutionProvider.Cpu,
    LlamaOptions = new LlamaOptions
    {
        GpuLayerCount = 0,
        Threads = Environment.ProcessorCount,  // Use all CPU cores
        BatchSize = 512,
        UseMemoryMap = true,
        UseMemoryLock = true   // Prevent swapping (requires privileges)
    }
};
```

### Chat Generation with GGUF

```csharp
using LMSupply.Generator;
using LMSupply.Generator.Models;

await using var model = await LocalGenerator.LoadAsync("gguf:auto");

var messages = new[]
{
    ChatMessage.System("You are a helpful assistant."),
    ChatMessage.User("What is the capital of France?")
};

await foreach (var token in model.GenerateChatAsync(messages))
{
    Console.Write(token);
}
```

### Multimodal Content (Vision Models)

`ChatMessage` supports multimodal content via `ContentPart` for vision-capable models like Gemma 4 multimodal:

```csharp
using LMSupply.Generator.Models;

// Convenience: text + single image
var msg = ChatMessage.UserWithImage(
    "What is in this image?",
    "data:image/jpeg;base64,/9j/4AAQSkZJRg...");

// Generic: arbitrary content parts
var multiPart = ChatMessage.UserWithContent(new ContentPart[]
{
    new TextContentPart("Compare these two images:"),
    new ImageContentPart { Url = "https://example.com/a.jpg" },
    new ImageContentPart { Url = "https://example.com/b.jpg" },
    new TextContentPart("Which has more cats?")
});

// IsMultimodal flag for inspection
if (msg.IsMultimodal) { /* contains at least one image */ }
```

**Backward compatibility**: `ChatMessage.User("hi")` works unchanged. `ContentParts` is null for text-only messages, and the `Content` field always holds a text fallback so non-vision formatters keep working.

### Tool Calling with GGUF

GGUF models support native tool calling via the `--jinja` flag (enabled by default).

```csharp
// Tool calling is automatically available with GGUF models
await using var model = await LocalGenerator.LoadAsync("gguf:gemma4-default");  // Gemma 4 E4B (gemma4 formatter)

// Tool definitions use OpenAI-compatible format via llama-server
```

> **Gemma 4 W1 advisory (v0.34.3+):** When loading any `gguf:gemma4-*` alias, a W1-level warning is emitted via `Trace.TraceWarning` at load time. Upstream llama.cpp PRs #21375 (chat-template/rope) and #21882 (instruction-following) are not yet in a stable release; tool-use with Korean instructional prompts and Q4_K_M quants may produce empty responses. Subscribe to `System.Diagnostics.Trace` listeners to receive this advisory. For production tool-calling workloads, `gguf:qwen2.5-7b` is a reliable alternative until the upstream PRs land.

> **Gemma 4 tool prompt injection:** `Gemma4ChatFormatter` automatically injects tool parameter hints as a system message, working around Gemma 4's tendency to emit empty tool arguments under llama-server's native Jinja template. The fragment content adapts to `GenerationOptions.Thinking`: when not `ThinkingMode.On` (default `Auto`), the full `Required parameters (MUST be provided): name (type)` block is injected; when `ThinkingMode.On`, only a compact per-tool required-params hint is injected (e.g. `search_knowledge: query (string)`) since the Jinja2 structured schema already covers full definitions, reducing system-prompt pressure on small models.

### Generation Options

```csharp
var genOptions = new GenerationOptions
{
    MaxTokens = 256,          // Maximum tokens to generate
    Temperature = 0.7f,        // Randomness (0.0 = deterministic)
    TopP = 0.9f,              // Nucleus sampling
    TopK = 40                  // Top-K sampling
};

await foreach (var token in model.GenerateAsync(prompt, genOptions))
{
    Console.Write(token);
}
```

#### Sampling Presets

`GenerationOptions` ships several static presets for common use cases:

| Preset | Temperature | TopP | TopK | MinP | RepetitionPenalty | Use Case |
|--------|-------------|------|------|------|-------------------|----------|
| `Default` | 0.7 | 0.9 | 50 | 0.05 | 1.1 | General purpose |
| `Creative` | 0.9 | 0.95 | 100 | 0.05 | 1.2 | Creative writing |
| `Precise` | 0.1 | 0.5 | 10 | 0.05 | 1.0 | Deterministic/factual |
| `Gemma4` | 1.0 | 0.95 | 64 | 0.05 | 1.0 | Gemma 4 models (Google recommended) |
| `Qwen3` | 0.6 | 0.95 | 20 | 0.0 | 1.0 | Qwen3 thinking mode (official recommendation) |

```csharp
// Use official Qwen3 sampling parameters for thinking mode
var options = GenerationOptions.Qwen3;
await foreach (var token in model.GenerateChatAsync(messages, options))
{
    Console.Write(token);
}

// Use Google's recommended params for Gemma 4 tool calling
await foreach (var token in model.GenerateChatAsync(messages, GenerationOptions.Gemma4))
{
    Console.Write(token);
}
```

> **Low-end caveat:** `Precise`, `Gemma4`, and `Qwen3` ship `RepetitionPenalty = 1.0`, which **disables**
> the primary repetition defense. That matches each vendor's full-precision recommendation, but on a
> low-end or heavily quantized model it raises the risk of degenerate run-on (the model ignores EOS and
> emits text up to the length cap). See **Anti-repetition and run-on defense** below to restore a safe floor.

#### Anti-repetition and run-on defense

Low-end and quantized models are prone to *degenerate run-on* — emitting plausible-but-aimless text to
the token limit instead of stopping. Beyond `RepetitionPenalty`/`FrequencyPenalty`/`PresencePenalty`,
`GenerationOptions` exposes the standard samplers that target this directly. All default to `null`
(= backend default), so leaving them unset changes nothing.

| Option | Wire key | Backend | Purpose |
|--------|----------|---------|---------|
| `DryMultiplier` | `dry_multiplier` | llama-server | DRY sampler strength (0 = off; ~0.8 enabled). Most effective run-on defense. |
| `DryBase` | `dry_base` | llama-server | DRY penalty growth base (default 1.75). |
| `DryAllowedLength` | `dry_allowed_length` | llama-server | Sequences up to this length are not penalized (default 2). |
| `DryPenaltyLastN` | `dry_penalty_last_n` | llama-server | DRY look-back window (-1 = context size). |
| `RepeatLastN` | `repeat_last_n` | llama-server | Window `RepetitionPenalty` considers (default 64). |
| `NoRepeatNgramSize` | `no_repeat_ngram_size` | ONNX only | Hard-blocks any n-gram of this size from recurring. |

> **Backend support differs.** DRY and `repeat_last_n` apply to the llama-server (GGUF) backend only;
> `no_repeat_ngram_size` applies to the ONNX backend only (llama-server does not support it). Options
> not supported by the active backend are omitted from the request rather than rejected.

```csharp
// Strong run-on defense on the GGUF/llama-server path
var options = new GenerationOptions
{
    DryMultiplier = 0.8f,   // enable DRY
    RepeatLastN = 256       // widen the repetition-penalty window
};
```

**Adaptive low-end safeguard.** `AdaptiveSamplingPolicy` is the single source of truth for the low-end-safe
anti-repetition floor. It is pure — it never changes behavior on its own; apply it where the user
expressed no explicit preference:

```csharp
// Raise a preset's disabled penalty (1.0) to the safe floor (1.1) only on a low-end/quantized model.
var opts = GenerationOptions.Qwen3;
opts.RepetitionPenalty = AdaptiveSamplingPolicy.ResolveRepetitionPenalty(opts.RepetitionPenalty, isLowEnd: true);
// isLowEnd=false leaves the value unchanged; it only ever raises, never lowers.
```

#### Output length safety

`MaxTokens` (default 512) is a hard cap on generated tokens enforced by **both** backends. `MaxNewTokens`,
when set, takes precedence as the output-only cap. A non-positive value is treated as *unset* (normalized
to the default), never as "unlimited" — generation can never be unbounded. The shared resolution is
`GenerationOptions.ResolveMaxOutputTokens()`.

### Gemma 4 Thinking Mode (v0.34+)

Gemma 4 models support an **extended thinking** mode activated via `GenerationOptions.Thinking`. Google recommends this for E2B/E4B models when complex function calling is required:

```csharp
await using var model = await LocalGenerator.LoadAsync("gguf:gemma4-default");  // Gemma 4 E4B

var messages = new[]
{
    ChatMessage.System("You are a helpful assistant."),
    ChatMessage.User("Solve this step by step: 17 × 23")
};

var options = new GenerationOptions { Thinking = ThinkingMode.On };
await foreach (var token in model.GenerateChatAsync(messages, options))
{
    Console.Write(token);
}
```

When `Thinking = ThinkingMode.On`, LMSupply prepends `<|think|>` to the first system message before sending the request to llama-server. The server (b8994+) separates internal reasoning into a `reasoning_content` field; LMSupply transparently skips those tokens, so the caller only receives the final response.

**Thinking + tool calling:** When `Thinking = ThinkingMode.On` and tools are provided, the Gemma 4 tool prompt fragment (see note above) is automatically reduced to compact required-params hints. This avoids doubling the tool schema (Jinja2 structured schema + text fragment) in the system prompt, which would increase context pressure on small models.

> `Gemma4ChatFormatter.GetThinkingToken()` returns `"<|think|>"` — used by orchestration layers that inject the thinking activation token programmatically.

> For DeepSeek R1's `<think>...</think>` tag format, see the section below.

### Reasoning Model Support (DeepSeek R1)

For reasoning models like DeepSeek R1 that output `<think>...</think>` tags:

```csharp
await using var model = await LocalGenerator.LoadAsync("unsloth/DeepSeek-R1-Distill-Llama-8B-GGUF");

// Option 1: Filter reasoning tokens (only show final answer)
var options = new GenerationOptions
{
    FilterReasoningTokens = true
};

await foreach (var token in model.GenerateChatAsync(messages, options))
{
    Console.Write(token); // Reasoning content is filtered out
}

// Option 2: Extract reasoning separately
var result = await model.GenerateChatWithReasoningAsync(messages);
Console.WriteLine($"Answer: {result.Response}");
Console.WriteLine($"Reasoning: {result.Reasoning}");
```

Supported reasoning tag formats:
- `<think>...</think>` (DeepSeek R1)
- `<｜begin▁of▁thinking｜>...<｜end▁of▁thinking｜>` (DeepSeek native format)

### Supported Chat Formats

The library auto-detects chat format from model filenames:

| Format | Models | Notes |
|--------|--------|-------|
| Llama 3 | Llama-3, Llama-3.1, Llama-3.2, CodeLlama | |
| ChatML | Qwen, Yi, InternLM, OpenChat | |
| Gemma | Gemma, Gemma-2 | system → user mapping |
| **Gemma 4** | Gemma 4 (E2B/E4B/26B/31B) | **Native system role** (`<start_of_turn>system`) |
| Phi-3 | Phi-3, Phi-3.5, Phi-4 | |
| Mistral | Mistral, Mixtral | |
| EXAONE | EXAONE | |
| DeepSeek | DeepSeek, DeepSeek-R1 | |
| Vicuna | Vicuna | |
| Zephyr | Zephyr | |

The detector distinguishes Gemma 4 (e.g., `gemma-4-E4B-it-...gguf`) from earlier Gemma generations and routes them to `Gemma4ChatFormatter`, which preserves the system role natively instead of mapping it to user.

### Model Information

```csharp
await using var model = await LocalGenerator.LoadAsync("gguf:auto");

var info = model.GetModelInfo();

Console.WriteLine($"Model: {info.ModelId}");
Console.WriteLine($"Path: {info.ModelPath}");
Console.WriteLine($"Context: {info.MaxContextLength}");
Console.WriteLine($"Format: {info.ChatFormat}");
Console.WriteLine($"Provider: {info.ExecutionProvider}");  // "llama-server-Vulkan"

// Partial GPU offload stats (null when all layers fit in VRAM)
if (info.GpuLayers.HasValue)
{
    Console.WriteLine($"GPU layers: {info.GpuLayers}/{info.TotalLayers}");
    Console.WriteLine($"VRAM: {info.EstimatedVramBytes / 1_073_741_824.0:F1} GB");
    Console.WriteLine($"RAM:  {info.EstimatedRamBytes / 1_073_741_824.0:F1} GB");
}
```

### GGUF vs ONNX

| Feature | GGUF (llama-server) | ONNX (GenAI) |
|---------|---------------------|--------------|
| Model availability | Extensive | Limited |
| Quantization options | Many (Q2-Q8) | FP16, INT4 |
| Setup complexity | Simple | Simple |
| GPU support | CUDA, Vulkan, Metal, ROCm | CUDA, DirectML, CoreML |
| Server pooling | Yes (reuses servers) | N/A |
| Memory efficiency | Good | Good |
| Inference speed | Fast | Fast |

## Known Issues

### Qwen3 Thinking Mode Enabled by Default

Several Qwen3/3.5/3.6 models (`gguf:qwen3-default`, `gguf:qwen3-quality`, `gguf:qwen3-large`) have thinking mode enabled by default, causing `<think>...</think>` blocks to appear in all responses. This is tagged with `GgufModelKnownIssues.ThinkingEnabledByDefault`.

To suppress think blocks, set `FilterReasoningTokens = true`:

```csharp
await using var model = await LocalGenerator.LoadAsync("gguf:qwen3-default");

var options = new GenerationOptions { FilterReasoningTokens = true };
await foreach (var token in model.GenerateChatAsync(messages, options))
{
    Console.Write(token); // Think blocks are filtered out
}
```

To check whether a model has this issue programmatically:

```csharp
var info = GgufModelRegistry.Resolve("gguf:qwen3-default");
bool thinkingOn = info?.KnownIssues.Contains(GgufModelKnownIssues.ThinkingEnabledByDefault) == true;
```

### ONNX GenAI Memory Leak Warnings

When using ONNX models, you may see stderr warnings like:

```
OGA Error: 1 instances of struct Generators::Model were leaked.
OGA Error: 1 instances of struct Generators::Tokenizer were leaked.
```

**This is a known upstream issue** in Microsoft's ONNX Runtime GenAI library, particularly affecting the DirectML backend. The warnings indicate internal resource tracking but **do not affect functionality**.

**Relevant upstream issues:**
- [microsoft/onnxruntime-genai#590](https://github.com/microsoft/onnxruntime-genai/issues/590) - Memory leak during back-to-back inferences
- [microsoft/onnxruntime-genai#1677](https://github.com/microsoft/onnxruntime-genai/issues/1677) - Memory Leak on CUDA

**Workarounds:**
1. The warnings can be safely ignored for most use cases
2. For long-running applications, consider periodic process restarts
3. GGUF models (via llama-server) do not exhibit this issue

**Status:** Tracking upstream fixes. LMSupply will update when OGA releases a fix.

## Requirements

- .NET 10.0+
- ONNX Runtime GenAI 0.7+ (for ONNX models)
- llama-server (auto-downloaded for GGUF models)
- Windows, Linux, or macOS

## License

MIT License
