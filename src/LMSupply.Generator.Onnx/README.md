# LMSupply.Generator.Onnx

ONNX Runtime GenAI backend for [LMSupply.Generator](https://www.nuget.org/packages/LMSupply.Generator).
Adds local text generation on ONNX Runtime GenAI models (Phi-4 Mini ONNX and similar) with
CUDA/DirectML GPU acceleration. Optional — `LMSupply.Generator` alone covers GGUF/llama-server
models without this package.

## Quick Start

```csharp
using LMSupply.Generator;

// Register once at startup, before loading any ONNX model.
LMSupply.Generator.Onnx.OnnxGeneratorBackend.Register();

var generator = await TextGeneratorBuilder.Create()
    .WithHuggingFaceModel("microsoft/Phi-4-mini-instruct-onnx")
    .BuildAsync();

string response = await generator.GenerateCompleteAsync("What is AI?");
```

Without registering this package, `TextGeneratorBuilder`/`LocalGenerator` throw a
`NotSupportedException` naming this package when asked to load an ONNX model. The hardware-aware
`WithDefaultModel()`/`"auto"` path never silently depends on it either — on a discrete non-NVIDIA
Windows GPU it only prefers ONNX/DirectML when this backend is registered, otherwise it falls back
to its GGUF selection.

## GPU Acceleration

```bash
# NVIDIA GPU
dotnet add package Microsoft.ML.OnnxRuntime.Gpu

# Windows (AMD/Intel/NVIDIA)
dotnet add package Microsoft.ML.OnnxRuntime.DirectML
```
