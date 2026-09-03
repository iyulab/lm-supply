# LMSupply.Generator

Local text generation and chat via GGUF (llama-server), with an optional ONNX Runtime GenAI backend.

## Features

- **Zero-config**: Models download automatically from HuggingFace
- **GPU Acceleration**: CUDA, Vulkan, DirectML, CoreML, Metal
- **GGUF Support**: Load any GGUF model via llama-server (auto-downloaded)
- **Server Pooling**: Reuses llama-server instances for fast model switching
- **MIT Models**: Phi-4 and Phi-3.5 models with no usage restrictions
- **Chat Support**: Built-in chat formatters for various models

## ONNX Runtime GenAI models

This package alone covers GGUF/llama-server. Loading an ONNX Runtime GenAI model (a
HuggingFace repo whose name or path indicates ONNX, or a local `genai_config.json` +
`model.onnx` directory) requires the separate `LMSupply.Generator.Onnx` package, registered
once at startup:

```csharp
dotnet add package LMSupply.Generator.Onnx
```

```csharp
LMSupply.Generator.Onnx.OnnxGeneratorBackend.Register();
```

Without it, loading an ONNX model throws a `NotSupportedException` naming the package. The
hardware-aware `WithDefaultModel()`/`"auto"` path is unaffected either way — it only ever
selects ONNX on a discrete non-NVIDIA Windows GPU, and only when this backend is registered;
otherwise it falls back to its GGUF selection, same as on any other hardware profile.

## Quick Start

```csharp
using LMSupply.Generator;

// Using the builder pattern
var generator = await TextGeneratorBuilder.Create()
    .WithDefaultModel()
    .BuildAsync();

// Generate text
string response = await generator.GenerateCompleteAsync("What is AI?");
Console.WriteLine(response);

await generator.DisposeAsync();
```

## Chat Completion

```csharp
var messages = new[]
{
    new ChatMessage(ChatRole.System, "You are a helpful assistant."),
    new ChatMessage(ChatRole.User, "Explain quantum computing.")
};

string response = await generator.GenerateChatCompleteAsync(messages);
```

## Available Models

| Model | Parameters | License | Description |
|-------|------------|---------|-------------|
| Phi-4 Mini | 3.8B | MIT | Default, best balance |
| Phi-3.5 Mini | 3.8B | MIT | Fast, reliable |
| Phi-4 | 14B | MIT | Highest quality |
| Llama 3.2 1B | 1B | Conditional | Ultra-lightweight |
| Llama 3.2 3B | 3B | Conditional | Balanced |

## GPU Acceleration

The GGUF/llama-server path auto-downloads the right `llama-server` binary for the detected GPU
(CUDA, Vulkan, Metal) — no extra NuGet package needed. GPU acceleration for ONNX Runtime GenAI
models is a `LMSupply.Generator.Onnx` concern; see that package's README.

## Advanced GGUF Tuning

Fine-grained llama-server control (GPU offload, batch size, RoPE/YaRN scaling, KV cache
quantization, speculative decoding, LoRA, and more) is available via `LlamaOptions`:

```csharp
var generator = await TextGeneratorBuilder.Create()
    .WithDefaultModel()
    .WithLlamaOptions(new LlamaOptions
    {
        GpuLayerCount = -1,
        AdditionalArgs = ["--verbose"], // raw llama-server CLI passthrough for flags not modeled above
    })
    .BuildAsync();
```

### Pinning or pre-provisioning the `llama-server` binary

By default, the `llama-server` binary itself (not the model) is acquired from an unauthenticated,
unpinned GitHub Releases "latest" lookup on first use. For offline, air-gapped, or
security-reviewed deployments, `ServerUpdateOptions` lets a consumer take over that acquisition:

```csharp
var generator = await TextGeneratorBuilder.Create()
    .WithDefaultModel()
    .WithServerUpdateOptions(new LlamaServerUpdateOptions
    {
        // Pin an exact release tag — a cache hit makes zero network calls; a cache miss downloads
        // exactly this tagged asset instead of "latest". A pinned installation also never
        // auto-checks for or applies a newer version.
        PinnedVersion = "b7898",

        // Or bypass acquisition entirely by pointing at an already-provisioned binary — takes
        // precedence over PinnedVersion when both are set.
        // ServerBinaryPath = "/opt/myapp/bin/llama-server",
    })
    .BuildAsync();
```

Reuse the same `LlamaServerUpdateOptions` instance across loads (rather than constructing a new one
each time) so they share one background-update timer and state file instead of each spawning its
own. `EmbedderOptions.ServerUpdateOptions` and `RerankerOptions.ServerUpdateOptions` follow the same
shape for the embedding and reranking GGUF paths.
