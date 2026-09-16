using System.Diagnostics;
using LMSupply;
using LMSupply.Runtime;
using LMSupply.Transcriber;

Console.WriteLine("=== LMSupply Transcriber GPU Test ===\n");

// Parse arguments
var providerArg = args.FirstOrDefault(a => a.StartsWith("--provider="));
var languageArg = args.FirstOrDefault(a => a.StartsWith("--language="));

var requestedProvider = ExecutionProvider.Auto;
if (providerArg != null)
{
    var providerName = providerArg.Split('=')[1].ToLower();
    requestedProvider = providerName switch
    {
        "cuda" => ExecutionProvider.Cuda,
        "cpu" => ExecutionProvider.Cpu,
        _ => ExecutionProvider.Auto
    };
}

string? requestedLanguage = languageArg?.Split('=')[1];

// 1. 환경 감지 테스트
Console.WriteLine("## 1. Environment Detection");

// CUDA environment diagnostics
var cudaPath = Environment.GetEnvironmentVariable("CUDA_PATH");
Console.WriteLine($"CUDA_PATH: {cudaPath ?? "(not set)"}");
if (!string.IsNullOrEmpty(cudaPath))
{
    var cudaBin = Path.Combine(cudaPath, "bin");
    var cuBlasLt = Path.Combine(cudaBin, "cublasLt64_12.dll");
    Console.WriteLine($"  bin dir exists: {Directory.Exists(cudaBin)}");
    Console.WriteLine($"  cublasLt64_12.dll exists: {File.Exists(cuBlasLt)}");
}

// Check for versioned CUDA paths
var cudaPathVars = Environment.GetEnvironmentVariables()
    .Keys.Cast<string>()
    .Where(k => k.StartsWith("CUDA_PATH_V", StringComparison.OrdinalIgnoreCase))
    .OrderByDescending(k => k)
    .ToList();
if (cudaPathVars.Count > 0)
{
    Console.WriteLine($"Versioned CUDA paths: {string.Join(", ", cudaPathVars)}");
}

// Check cuDNN
var cudnnPath = @"C:\Program Files\NVIDIA\CUDNN";
if (Directory.Exists(cudnnPath))
{
    var cudnnVersions = Directory.GetDirectories(cudnnPath, "v*").Select(Path.GetFileName);
    Console.WriteLine($"cuDNN versions found: {string.Join(", ", cudnnVersions)}");
}
else
{
    Console.WriteLine("cuDNN directory not found");
}
Console.WriteLine();

await RuntimeManager.Instance.InitializeAsync();
Console.WriteLine($"Platform: {RuntimeManager.Instance.Platform}");
Console.WriteLine($"GPU: {RuntimeManager.Instance.Gpu}");
Console.WriteLine($"Recommended Provider: {RuntimeManager.Instance.RecommendedProvider}");
Console.WriteLine($"Fallback Chain: {string.Join(" -> ", RuntimeManager.Instance.GetProviderFallbackChain())}");

// Diagnostic: Check cuDNN path detection
var cudaEnv = CudaEnvironment.Instance;
var cudaMajor = RuntimeManager.Instance.Gpu?.CudaDriverVersionMajor ?? 12;
Console.WriteLine($"\n### cuDNN Path Diagnostics (CUDA Major: {cudaMajor})");
var dllPaths = cudaEnv.GetDllSearchPaths(cudaMajor).ToList();
Console.WriteLine($"DLL Search Paths ({dllPaths.Count}):");
foreach (var p in dllPaths)
{
    var zlibExists = File.Exists(Path.Combine(p, "zlibwapi.dll"));
    var cudnnExists = File.Exists(Path.Combine(p, "cudnn64_9.dll"));
    Console.WriteLine($"  - {p}");
    Console.WriteLine($"    cudnn64_9.dll: {cudnnExists}, zlibwapi.dll: {zlibExists}");
}

// Check if PATH was modified
var pathVar = Environment.GetEnvironmentVariable("PATH") ?? "";
var cudnnInPath = dllPaths.Any(p => pathVar.Contains(p, StringComparison.OrdinalIgnoreCase));
Console.WriteLine($"cuDNN dir in PATH: {cudnnInPath}");
Console.WriteLine();

// 2. 모델 로드 테스트
Console.WriteLine($"## 2. Model Load Test (Provider: {requestedProvider})");
Console.WriteLine($"Loading 'default' model with {requestedProvider} provider...");

var options = new TranscriberOptions
{
    Provider = requestedProvider
};

var sw = Stopwatch.StartNew();
await using var model = await LocalTranscriber.LoadAsync("default", options);
sw.Stop();

Console.WriteLine($"Model loaded in {sw.ElapsedMilliseconds}ms");
Console.WriteLine($"Model ID: {model.ModelId}");
Console.WriteLine($"GPU Active: {model.IsGpuActive}");
Console.WriteLine($"Active Providers: [{string.Join(", ", model.ActiveProviders)}]");
Console.WriteLine($"Requested Provider: {model.RequestedProvider}");
Console.WriteLine();

// 3. 테스트 오디오 파일이 있으면 실제 추론 테스트
var testAudioPath = args.FirstOrDefault(a => !a.StartsWith("--") && File.Exists(a));

if (!string.IsNullOrEmpty(testAudioPath) && File.Exists(testAudioPath))
{
    Console.WriteLine("## 3. Transcription Test");
    Console.WriteLine($"Audio file: {testAudioPath}");

    // WordTimestamps 테스트
    var transcribeOptions = new TranscribeOptions
    {
        WordTimestamps = true,  // segment-level timestamps
        Language = requestedLanguage  // null = auto-detect, or specify: zh, en, ko, ja, etc.
    };

    if (requestedLanguage != null)
        Console.WriteLine($"Language: {requestedLanguage}");

    sw.Restart();
    var result = await model.TranscribeAsync(testAudioPath, transcribeOptions);
    sw.Stop();

    Console.WriteLine($"\nTranscription completed in {sw.ElapsedMilliseconds}ms");
    Console.WriteLine($"Audio Duration: {result.DurationSeconds:F2}s");
    Console.WriteLine($"Inference Time: {result.InferenceTimeMs:F2}ms");
    Console.WriteLine($"Real-time Factor: {result.InferenceTimeMs / (result.DurationSeconds * 1000):F2}x");
    Console.WriteLine($"Detected Language: {result.Language}");
    Console.WriteLine($"Segments: {result.Segments.Count}");
    Console.WriteLine();

    Console.WriteLine("### Transcription Result:");
    Console.WriteLine(result.Text);
    Console.WriteLine();

    if (result.Segments.Count > 0)
    {
        Console.WriteLine("### Segments (first 5):");
        foreach (var seg in result.Segments.Take(5))
        {
            Console.WriteLine($"  [{seg.Start:F2}s - {seg.End:F2}s] {seg.Text}");
        }
    }
}
else
{
    Console.WriteLine("## 3. Transcription Test (SKIPPED)");
    Console.WriteLine("To test transcription, provide an audio file path as argument:");
    Console.WriteLine("  dotnet run -- path/to/audio.wav");
}

Console.WriteLine();
Console.WriteLine("=== Test Complete ===");

// 4. Large 모델 테스트 (선택적)
if (args.Contains("--large"))
{
    Console.WriteLine("\n## 4. Large Model Test");
    Console.WriteLine("Loading 'large' model (whisper-large-v3-turbo)...");
    Console.WriteLine("This will download ~3GB on first run.");

    sw.Restart();
    await using var largeModel = await LocalTranscriber.LoadAsync("large", options);
    sw.Stop();

    Console.WriteLine($"Large model loaded in {sw.ElapsedMilliseconds}ms");
    Console.WriteLine($"GPU Active: {largeModel.IsGpuActive}");
    Console.WriteLine($"Active Providers: [{string.Join(", ", largeModel.ActiveProviders)}]");
}
