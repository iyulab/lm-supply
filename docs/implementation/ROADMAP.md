# LMSupply.Generator Implementation Roadmap

> Based on Research 01-06 findings and detailed analysis reports

## Implementation Status (Updated)

| Phase | Description | Status | Tests |
|-------|-------------|--------|-------|
| Phase 1 | Project Foundation | ✅ Complete | 16 |
| Phase 2 | Core Inference Engine | ✅ Complete | 9 |
| Phase 3 | Streaming & Concurrency | ✅ Complete | 12 |
| Phase 4 | Model Registry & Licensing | ✅ Complete | 25 |
| Phase 5 | Speculative Decoding (Interface) | ✅ Complete | - |
| Phase 6 | Builder API & Model Factory | ✅ Complete | - |

**Total Tests: 62 passing**

### Key Implementations from Research 05-06

| Feature | Implementation | Source |
|---------|---------------|--------|
| GeneratorOptions Extensions | DoSample, NumBeams, PastPresentShareBuffer, MaxNewTokens | research-05 |
| Concurrency Limiting | SemaphoreSlim in OnnxGeneratorModel | research-05 |
| Memory Estimation | MemoryEstimator with KV cache calculation | research-05 |
| License Tiering | ModelRegistry with MIT/Conditional classification | research-06 |
| Model Pool | GeneratorPool with LRU eviction | research-06 |
| Speculative Decoding | ISpeculativeDecoder interface prepared | research-06 |
| Builder API | TextGeneratorBuilder fluent pattern | Phase 6 |
| WellKnownModels | Model presets with license info | Phase 6 |
| Model Factory | OnnxGeneratorModelFactory with caching | Phase 6 |

---

## Executive Summary

| Item | Decision | Rationale |
|------|----------|-----------|
| **Inference Engine** | ONNX Runtime GenAI v0.11.4 | LMSupply stack consistency, Microsoft support, 9.46x perf vs llama.cpp |
| **Default Model** | microsoft/Phi-3.5-mini-instruct-onnx | MIT license, 3.8B params, 128K context |
| **Quantization** | INT4 (AWQ/RTN) | 87.5% memory reduction, minimal quality loss |
| **Tokenizer** | GenAI built-in | No separate library needed |
| **Chat Template** | Hardcoded formatters | No Jinja2 runtime overhead |
| **Streaming** | IAsyncEnumerable<string> | .NET standard async pattern |
| **KV Cache** | GenAI auto-managed | No manual implementation needed |

---

## Phase 1: Project Foundation (Week 1)

### 1.1 Project Structure Setup

**Task 1.1.1**: Create LMSupply.Generator project
```
src/
├── LMSupply.Generator/
│   ├── LMSupply.Generator.csproj
│   ├── Abstractions/
│   │   ├── ITextGenerator.cs
│   │   ├── IChatFormatter.cs
│   │   └── IModelLoader.cs
│   ├── Models/
│   │   ├── ChatMessage.cs
│   │   ├── ChatRole.cs
│   │   ├── GeneratorOptions.cs
│   │   └── ModelInfo.cs
│   └── Internal/
│       └── (implementation details)
```

**Task 1.1.2**: Configure .csproj with dependencies
```xml
<Project Sdk="Microsoft.NET.Sdk">
  <PropertyGroup>
    <TargetFramework>net10.0</TargetFramework>
    <ImplicitUsings>enable</ImplicitUsings>
    <Nullable>enable</Nullable>
  </PropertyGroup>

  <ItemGroup>
    <!-- CPU package as base, GPU via runtime detection -->
    <PackageReference Include="Microsoft.ML.OnnxRuntimeGenAI" />
    <ProjectReference Include="..\LMSupply.Core\LMSupply.Core.csproj" />
  </ItemGroup>
</Project>
```

**Task 1.1.3**: Add package versions to Directory.Packages.props
```xml
<!-- ONNX Runtime GenAI - pin versions due to preview status -->
<PackageVersion Include="Microsoft.ML.OnnxRuntimeGenAI" Version="0.11.4" />
<PackageVersion Include="Microsoft.ML.OnnxRuntimeGenAI.Cuda" Version="0.11.2" />
<PackageVersion Include="Microsoft.ML.OnnxRuntimeGenAI.DirectML" Version="0.11.4" />
```

### 1.2 Core Abstractions

**Task 1.2.1**: Define ITextGenerator interface
```csharp
public interface ITextGenerator : IAsyncDisposable
{
    IAsyncEnumerable<string> GenerateAsync(
        string prompt,
        GeneratorOptions? options = null,
        CancellationToken cancellationToken = default);

    IAsyncEnumerable<string> GenerateChatAsync(
        IEnumerable<ChatMessage> messages,
        GeneratorOptions? options = null,
        CancellationToken cancellationToken = default);

    Task<string> GenerateCompleteAsync(
        string prompt,
        GeneratorOptions? options = null,
        CancellationToken cancellationToken = default);

    ModelInfo ModelInfo { get; }
}
```

**Task 1.2.2**: Define GeneratorOptions
```csharp
public class GeneratorOptions
{
    public int MaxTokens { get; set; } = 512;
    public float Temperature { get; set; } = 0.7f;
    public float TopP { get; set; } = 0.9f;
    public int TopK { get; set; } = 50;
    public float RepetitionPenalty { get; set; } = 1.1f;
    public IReadOnlyList<string>? StopSequences { get; set; }
    public int? RandomSeed { get; set; }
}
```

**Task 1.2.3**: Define ChatMessage and ChatRole
```csharp
public enum ChatRole { System, User, Assistant }
public record ChatMessage(ChatRole Role, string Content);
```

### Deliverables
- [ ] LMSupply.Generator.csproj created
- [ ] Core interfaces defined (ITextGenerator, IChatFormatter)
- [ ] Model classes defined (GeneratorOptions, ChatMessage, ModelInfo)
- [ ] Package references configured

---

## Phase 2: Core Inference Engine (Week 2)

### 2.1 ONNX GenAI Integration

**Task 2.1.1**: Implement OnnxTextGenerator
```csharp
public sealed class OnnxTextGenerator : ITextGenerator
{
    private readonly Model _model;
    private readonly Tokenizer _tokenizer;
    private readonly IChatFormatter _chatFormatter;
    private readonly ModelInfo _modelInfo;

    public OnnxTextGenerator(string modelPath, IChatFormatter? chatFormatter = null)
    {
        _model = new Model(modelPath);
        _tokenizer = new Tokenizer(_model);
        _modelInfo = LoadModelInfo(modelPath);
        _chatFormatter = chatFormatter ?? ChatFormatterFactory.Create(_modelInfo.Name);
    }

    public async IAsyncEnumerable<string> GenerateAsync(
        string prompt,
        GeneratorOptions? options = null,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        options ??= new GeneratorOptions();
        var tokens = _tokenizer.Encode(prompt);

        using var generatorParams = new GeneratorParams(_model);
        ConfigureParams(generatorParams, options);
        generatorParams.SetInputSequences(tokens);

        using var tokenizerStream = _tokenizer.CreateStream();
        using var generator = new Generator(_model, generatorParams);

        while (!generator.IsDone())
        {
            cancellationToken.ThrowIfCancellationRequested();

            generator.ComputeLogits();
            generator.GenerateNextToken();

            var outputTokens = generator.GetSequence(0);
            var newToken = outputTokens[^1];
            var decoded = tokenizerStream.Decode(newToken);

            if (ShouldStop(decoded, options.StopSequences))
                break;

            yield return decoded;
            await Task.Yield();
        }
    }
}
```

**Task 2.1.2**: Implement GeneratorParams configuration
```csharp
private static void ConfigureParams(GeneratorParams genParams, GeneratorOptions options)
{
    genParams.SetSearchOption("max_length", options.MaxTokens);
    genParams.SetSearchOption("do_sample", options.Temperature > 0);
    genParams.SetSearchOption("temperature", options.Temperature);
    genParams.SetSearchOption("top_p", options.TopP);
    genParams.SetSearchOption("top_k", options.TopK);
    genParams.SetSearchOption("repetition_penalty", options.RepetitionPenalty);

    if (options.RandomSeed.HasValue)
        genParams.SetSearchOption("random_seed", options.RandomSeed.Value);
}
```

### 2.2 Stop Sequence Detection

**Task 2.2.1**: Implement stop sequence logic
```csharp
public sealed class StopSequenceDetector
{
    private readonly IReadOnlyList<string> _stopSequences;
    private readonly StringBuilder _buffer = new();

    public StopSequenceDetector(IReadOnlyList<string>? stopSequences)
    {
        _stopSequences = stopSequences ?? [];
    }

    public (string Output, bool ShouldStop) Process(string token)
    {
        _buffer.Append(token);
        var text = _buffer.ToString();

        foreach (var stop in _stopSequences)
        {
            if (text.Contains(stop))
            {
                var idx = text.IndexOf(stop);
                return (text[..idx], true);
            }
        }

        // Check for partial matches at end
        foreach (var stop in _stopSequences)
        {
            for (int i = 1; i < stop.Length; i++)
            {
                if (text.EndsWith(stop[..i]))
                    return (text[..^i], false); // Buffer partial match
            }
        }

        var output = _buffer.ToString();
        _buffer.Clear();
        return (output, false);
    }
}
```

### Deliverables
- [ ] OnnxTextGenerator implemented with streaming
- [ ] GeneratorParams configuration
- [ ] Stop sequence detection
- [ ] Resource disposal (IAsyncDisposable)

---

## Phase 3: Chat/Conversation Support (Week 3)

### 3.1 Chat Formatter Interface

**Task 3.1.1**: Define IChatFormatter interface
```csharp
public interface IChatFormatter
{
    string FormatPrompt(IEnumerable<ChatMessage> messages);
    string GetStopToken();
    IReadOnlyList<string> GetStopSequences();
    string ModelFamily { get; }
}
```

### 3.2 Model-Specific Formatters

**Task 3.2.1**: Implement Phi-3 formatter
```csharp
public sealed class Phi3ChatFormatter : IChatFormatter
{
    public string ModelFamily => "phi";

    public string FormatPrompt(IEnumerable<ChatMessage> messages)
    {
        var sb = new StringBuilder();
        foreach (var msg in messages)
        {
            var role = msg.Role switch
            {
                ChatRole.System => "system",
                ChatRole.User => "user",
                ChatRole.Assistant => "assistant",
                _ => throw new ArgumentOutOfRangeException()
            };
            sb.AppendLine($"<|{role}|>");
            sb.Append(msg.Content);
            sb.AppendLine("<|end|>");
        }
        sb.Append("<|assistant|>");
        return sb.ToString();
    }

    public string GetStopToken() => "<|end|>";
    public IReadOnlyList<string> GetStopSequences() =>
        ["<|end|>", "<|user|>", "<|system|>", "<|endoftext|>"];
}
```

**Task 3.2.2**: Implement Llama-3 formatter
```csharp
public sealed class Llama3ChatFormatter : IChatFormatter
{
    public string ModelFamily => "llama";

    public string FormatPrompt(IEnumerable<ChatMessage> messages)
    {
        var sb = new StringBuilder();
        sb.Append("<|begin_of_text|>");

        foreach (var msg in messages)
        {
            var role = msg.Role switch
            {
                ChatRole.System => "system",
                ChatRole.User => "user",
                ChatRole.Assistant => "assistant",
                _ => throw new ArgumentOutOfRangeException()
            };
            sb.Append($"<|start_header_id|>{role}<|end_header_id|>\n\n");
            sb.Append(msg.Content);
            sb.Append("<|eot_id|>");
        }
        sb.Append("<|start_header_id|>assistant<|end_header_id|>\n\n");
        return sb.ToString();
    }

    public string GetStopToken() => "<|eot_id|>";
    public IReadOnlyList<string> GetStopSequences() =>
        ["<|eot_id|>", "<|end_of_text|>"];
}
```

**Task 3.2.3**: Implement Qwen (ChatML) formatter
```csharp
public sealed class QwenChatFormatter : IChatFormatter
{
    public string ModelFamily => "qwen";

    public string FormatPrompt(IEnumerable<ChatMessage> messages)
    {
        var sb = new StringBuilder();
        foreach (var msg in messages)
        {
            var role = msg.Role switch
            {
                ChatRole.System => "system",
                ChatRole.User => "user",
                ChatRole.Assistant => "assistant",
                _ => throw new ArgumentOutOfRangeException()
            };
            sb.AppendLine($"<|im_start|>{role}");
            sb.Append(msg.Content);
            sb.AppendLine("<|im_end|>");
        }
        sb.Append("<|im_start|>assistant\n");
        return sb.ToString();
    }

    public string GetStopToken() => "<|im_end|>";
    public IReadOnlyList<string> GetStopSequences() =>
        ["<|im_end|>", "<|endoftext|>"];
}
```

**Task 3.2.4**: Implement ChatFormatterFactory
```csharp
public static class ChatFormatterFactory
{
    public static IChatFormatter Create(string modelName)
    {
        var name = modelName.ToLowerInvariant();
        return name switch
        {
            var n when n.Contains("phi") => new Phi3ChatFormatter(),
            var n when n.Contains("llama") => new Llama3ChatFormatter(),
            var n when n.Contains("qwen") => new QwenChatFormatter(),
            var n when n.Contains("mistral") => new MistralChatFormatter(),
            var n when n.Contains("gemma") => new GemmaChatFormatter(),
            _ => new GenericChatFormatter()
        };
    }
}
```

### 3.3 Chat Generation

**Task 3.3.1**: Implement GenerateChatAsync
```csharp
public async IAsyncEnumerable<string> GenerateChatAsync(
    IEnumerable<ChatMessage> messages,
    GeneratorOptions? options = null,
    [EnumeratorCancellation] CancellationToken cancellationToken = default)
{
    options ??= new GeneratorOptions();

    // Add model's stop sequences
    var stopSeqs = options.StopSequences?.ToList() ?? [];
    stopSeqs.AddRange(_chatFormatter.GetStopSequences());
    options = options with { StopSequences = stopSeqs.Distinct().ToList() };

    var prompt = _chatFormatter.FormatPrompt(messages);

    await foreach (var token in GenerateAsync(prompt, options, cancellationToken))
    {
        yield return token;
    }
}
```

### Deliverables
- [ ] IChatFormatter interface and implementations
- [ ] Phi-3, Llama-3, Qwen formatters
- [ ] ChatFormatterFactory with auto-detection
- [ ] GenerateChatAsync implementation

---

## Phase 4: Advanced Features (Week 4)

### 4.1 Hardware Detection

**Task 4.1.1**: Implement HardwareDetector
```csharp
public static class HardwareDetector
{
    public static HardwareProfile Detect()
    {
        var systemMemoryMB = GetSystemMemory();
        var gpuInfo = DetectGpu();

        return new HardwareProfile
        {
            SystemMemoryMB = systemMemoryMB,
            GpuMemoryMB = gpuInfo?.MemoryMB,
            GpuVendor = gpuInfo?.Vendor,
            HasCuda = DetectCuda(),
            HasDirectML = OperatingSystem.IsWindows(),
            RecommendedProvider = DetermineProvider(gpuInfo),
            RecommendedQuantization = DetermineQuantization(gpuInfo, systemMemoryMB)
        };
    }

    private static string DetermineProvider(GpuInfo? gpu)
    {
        if (gpu?.HasCuda == true) return "cuda";
        if (OperatingSystem.IsWindows() && gpu != null) return "directml";
        return "cpu";
    }

    private static QuantizationType DetermineQuantization(GpuInfo? gpu, long systemMB)
    {
        return (gpu?.MemoryMB, systemMB) switch
        {
            (>= 16000, _) => QuantizationType.FP16,
            (>= 8000, _) => QuantizationType.INT8,
            (>= 4000, _) => QuantizationType.INT4,
            (_, >= 32000) => QuantizationType.INT8,
            _ => QuantizationType.INT4
        };
    }
}
```

### 4.2 Model Management

**Task 4.2.1**: Implement ModelDownloader (extend LMSupply.Core pattern)
```csharp
public sealed class GeneratorModelDownloader
{
    private readonly HuggingFaceModelDownloader _downloader;

    public async Task<string> EnsureModelAsync(
        string modelId,
        string? variant = null,
        CancellationToken cancellationToken = default)
    {
        // Default variant based on hardware
        variant ??= DetermineVariant();

        var localPath = await _downloader.DownloadModelAsync(
            modelId,
            includePatterns: GetPatterns(variant),
            cancellationToken: cancellationToken);

        // Validate ONNX GenAI structure
        ValidateModelStructure(localPath);

        return localPath;
    }

    private static string[] GetPatterns(string variant) => variant switch
    {
        "cpu-int4" => ["cpu-int4-rtn-block-32-acc-level-4/*"],
        "cuda-int4" => ["cuda-int4-rtn-block-32/*"],
        "directml-int4" => ["directml-int4-awq-block-128/*"],
        "cuda-fp16" => ["cuda-fp16/*"],
        _ => ["cpu-int4-rtn-block-32-acc-level-4/*"]
    };

    private static void ValidateModelStructure(string path)
    {
        var required = new[] { "genai_config.json", "model.onnx", "tokenizer.json" };
        foreach (var file in required)
        {
            if (!File.Exists(Path.Combine(path, file)))
                throw new InvalidOperationException($"Missing required file: {file}");
        }
    }
}
```

### 4.3 Memory-Aware Generation

**Task 4.3.1**: Implement memory safeguards
```csharp
public sealed class MemoryAwareGenerator : ITextGenerator
{
    private readonly ITextGenerator _inner;
    private readonly long _maxMemoryBytes;

    public async IAsyncEnumerable<string> GenerateAsync(
        string prompt,
        GeneratorOptions? options = null,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        CheckMemoryBudget();

        await foreach (var token in _inner.GenerateAsync(prompt, options, cancellationToken))
        {
            yield return token;

            // Periodic check
            if (GC.GetTotalMemory(false) > _maxMemoryBytes * 0.95)
            {
                GC.Collect(2, GCCollectionMode.Optimized, blocking: false);
            }
        }
    }

    private void CheckMemoryBudget()
    {
        var current = GC.GetTotalMemory(false);
        if (current > _maxMemoryBytes * 0.9)
        {
            GC.Collect();
            GC.WaitForPendingFinalizers();

            if (GC.GetTotalMemory(true) > _maxMemoryBytes * 0.9)
                throw new OutOfMemoryException("Insufficient memory for generation");
        }
    }
}
```

### 4.4 Concurrent Request Handling

**Task 4.4.1**: Implement request queue
```csharp
public sealed class GeneratorPool : IAsyncDisposable
{
    private readonly SemaphoreSlim _semaphore;
    private readonly ITextGenerator _generator;

    public GeneratorPool(ITextGenerator generator, int maxConcurrency = 1)
    {
        _generator = generator;
        _semaphore = new SemaphoreSlim(maxConcurrency, maxConcurrency);
    }

    public async IAsyncEnumerable<string> GenerateAsync(
        string prompt,
        GeneratorOptions? options = null,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        await _semaphore.WaitAsync(cancellationToken);
        try
        {
            await foreach (var token in _generator.GenerateAsync(prompt, options, cancellationToken))
            {
                yield return token;
            }
        }
        finally
        {
            _semaphore.Release();
        }
    }
}
```

### Deliverables
- [ ] HardwareDetector implementation
- [ ] GeneratorModelDownloader
- [ ] MemoryAwareGenerator wrapper
- [ ] GeneratorPool for concurrent requests

---

## Phase 5: Integration & Documentation (Week 5)

### 5.1 Builder Pattern API

**Task 5.1.1**: Implement TextGeneratorBuilder
```csharp
public sealed class TextGeneratorBuilder
{
    private string? _modelPath;
    private string? _modelId;
    private GeneratorOptions? _defaultOptions;
    private IChatFormatter? _chatFormatter;
    private int? _maxMemoryMB;
    private int _maxConcurrency = 1;

    public TextGeneratorBuilder WithModel(string modelPath)
    {
        _modelPath = modelPath;
        return this;
    }

    public TextGeneratorBuilder WithHuggingFaceModel(string modelId)
    {
        _modelId = modelId;
        return this;
    }

    public TextGeneratorBuilder WithDefaultOptions(GeneratorOptions options)
    {
        _defaultOptions = options;
        return this;
    }

    public TextGeneratorBuilder WithChatFormatter(IChatFormatter formatter)
    {
        _chatFormatter = formatter;
        return this;
    }

    public TextGeneratorBuilder WithMemoryLimit(int maxMemoryMB)
    {
        _maxMemoryMB = maxMemoryMB;
        return this;
    }

    public TextGeneratorBuilder WithConcurrency(int maxConcurrent)
    {
        _maxConcurrency = maxConcurrent;
        return this;
    }

    public async Task<ITextGenerator> BuildAsync(CancellationToken cancellationToken = default)
    {
        var modelPath = _modelPath;

        if (modelPath == null && _modelId != null)
        {
            var downloader = new GeneratorModelDownloader();
            modelPath = await downloader.EnsureModelAsync(_modelId, cancellationToken: cancellationToken);
        }

        if (modelPath == null)
            throw new InvalidOperationException("Model path or ID required");

        ITextGenerator generator = new OnnxTextGenerator(modelPath, _chatFormatter);

        if (_maxMemoryMB.HasValue)
            generator = new MemoryAwareGenerator(generator, _maxMemoryMB.Value);

        if (_maxConcurrency > 1)
            generator = new GeneratorPool(generator, _maxConcurrency);

        return generator;
    }
}
```

### 5.2 Default Model Registry

**Task 5.2.1**: Define well-known models
```csharp
public static class WellKnownModels
{
    public static class Generator
    {
        /// <summary>Default balanced model - Phi-3.5 mini, MIT license</summary>
        public const string Default = "microsoft/Phi-3.5-mini-instruct-onnx";

        /// <summary>Fast/small model - Llama 3.2 1B</summary>
        public const string Fast = "onnx-community/Llama-3.2-1B-Instruct-ONNX";

        /// <summary>Quality model - Phi-4</summary>
        public const string Quality = "microsoft/phi-4-onnx";

        /// <summary>Multilingual - Qwen 2.5 3B</summary>
        public const string Multilingual = "Qwen/Qwen2.5-3B-Instruct"; // Requires conversion
    }
}
```

### 5.3 Test Suite

**Task 5.3.1**: Unit tests structure
```
tests/
└── LMSupply.Generator.Tests/
    ├── ChatFormatterTests.cs
    ├── StopSequenceDetectorTests.cs
    ├── GeneratorOptionsTests.cs
    └── Integration/
        └── OnnxTextGeneratorTests.cs
```

### 5.4 Documentation

**Task 5.4.1**: Update main README
**Task 5.4.2**: Create Generator-specific documentation
**Task 5.4.3**: Add usage examples

### Deliverables
- [ ] TextGeneratorBuilder fluent API
- [ ] WellKnownModels registry
- [ ] Unit and integration tests
- [ ] Documentation updates

---

## Risk Mitigation

### DLL Hell Prevention (Critical)
- LMSupply.Generator uses separate ONNX Runtime GenAI binaries
- Pin exact package versions in Directory.Packages.props
- Test coexistence with LMSupply.Embedder/Reranker

### DirectML Chat Mode Limitation
- Document that continuous decoding not supported on DirectML
- Recommend CUDA for chat applications
- Implement session restart as workaround

### Preview API Stability
- Pin to tested version (0.11.4)
- Wrap GenAI types to isolate breaking changes
- Monitor release notes for migration needs

---

## Success Criteria

### Phase 1
- [ ] Project compiles and references resolve
- [ ] Interfaces define complete contract

### Phase 2
- [ ] Single prompt generation works with streaming
- [ ] Memory properly disposed

### Phase 3
- [ ] Chat conversations work for Phi-3, Llama-3, Qwen
- [ ] Stop sequences properly detected

### Phase 4
- [ ] Hardware auto-detection works
- [ ] Model download integrates with LMSupply.Core
- [ ] Memory limits enforced

### Phase 5
- [ ] Builder API intuitive to use
- [ ] Tests pass
- [ ] Documentation complete

---

## Recommended Start Point

**Begin with Phase 1, Task 1.1.1**: Create the LMSupply.Generator project structure.

```bash
# Create project
dotnet new classlib -n LMSupply.Generator -o src/LMSupply.Generator
cd src/LMSupply.Generator

# Add to solution
dotnet sln ../../LMSupply.sln add LMSupply.Generator.csproj

# Add project reference
dotnet add reference ../LMSupply.Core/LMSupply.Core.csproj
```

Then proceed with interface definitions before any implementation.
