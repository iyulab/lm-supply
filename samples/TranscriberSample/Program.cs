using System.Diagnostics;
using System.Text.Encodings.Web;
using System.Text.Json;
using LMSupply;
using LMSupply.Transcriber;

namespace TranscriberSample;

/// <summary>
/// Sample application demonstrating LMSupply.Transcriber capabilities.
/// Tests GPU/CPU provider selection, different models, and timestamp options.
/// </summary>
public static class Program
{
    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        WriteIndented = true,
        Encoder = JavaScriptEncoder.UnsafeRelaxedJsonEscaping
    };

    public static async Task Main(string[] args)
    {
        Console.WriteLine("=== LMSupply Transcriber Sample ===\n");

        // Parse command line arguments
        var audioPath = args.Length > 0 ? args[0] : null;
        var modelAlias = args.Length > 1 ? args[1] : "default";
        var provider = ParseProvider(args.Length > 2 ? args[2] : "auto");

        // Display available models
        Console.WriteLine("Available models:");
        foreach (var model in LocalTranscriber.GetAllModels())
        {
            Console.WriteLine($"  - {model.AliasName,-12} : {model.DisplayName} ({model.ParametersM:F0}M params)");
        }
        Console.WriteLine();

        // If no audio file provided, just show system info
        if (string.IsNullOrEmpty(audioPath))
        {
            await ShowSystemInfoAsync(modelAlias, provider);
            Console.WriteLine("\nUsage: TranscriberSample <audio-file> [model-alias] [provider]");
            Console.WriteLine("  model-alias: default, fast, quality, large, turbo");
            Console.WriteLine("  provider: auto, cpu, cuda, coreml");
            return;
        }

        // Verify audio file exists
        if (!File.Exists(audioPath))
        {
            Console.WriteLine($"Error: Audio file not found: {audioPath}");
            return;
        }

        // Run transcription tests
        await RunTranscriptionTestsAsync(audioPath, modelAlias, provider);
    }

    private static async Task ShowSystemInfoAsync(string modelAlias, ExecutionProvider provider)
    {
        Console.WriteLine("=== System Information ===\n");

        var options = new TranscriberOptions
        {
            ModelId = modelAlias,
            Provider = provider
        };

        Console.WriteLine($"Requested Provider: {provider}");

        try
        {
            await using var model = await LocalTranscriber.LoadAsync(options);

            Console.WriteLine($"Model ID: {model.ModelId}");
            Console.WriteLine($"GPU Active: {model.IsGpuActive}");
            Console.WriteLine($"Active Providers: [{string.Join(", ", model.ActiveProviders)}]");
            Console.WriteLine($"Requested Provider: {model.RequestedProvider}");

            if (provider != ExecutionProvider.Cpu && !model.IsGpuActive)
            {
                Console.WriteLine("\nWARNING: GPU provider was requested but only CPU is active.");
                Console.WriteLine("Possible causes:");
                Console.WriteLine("  - CUDA toolkit not installed (for CUDA provider)");
                Console.WriteLine("  - GPU driver issues");
                Console.WriteLine("  - Incompatible GPU");
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error loading model: {ex.Message}");
        }
    }

    private static async Task RunTranscriptionTestsAsync(
        string audioPath,
        string modelAlias,
        ExecutionProvider provider)
    {
        Console.WriteLine($"=== Transcription Test ===\n");
        Console.WriteLine($"Audio file: {audioPath}");
        Console.WriteLine($"Model: {modelAlias}");
        Console.WriteLine($"Provider: {provider}");
        Console.WriteLine();

        var options = new TranscriberOptions
        {
            ModelId = modelAlias,
            Provider = provider
        };

        // Load model
        Console.WriteLine("Loading model...");
        var loadStart = Stopwatch.StartNew();
        await using var model = await LocalTranscriber.LoadAsync(options);
        loadStart.Stop();

        Console.WriteLine($"Model loaded in {loadStart.ElapsedMilliseconds}ms");
        Console.WriteLine($"GPU Active: {model.IsGpuActive}");
        Console.WriteLine($"Active Providers: [{string.Join(", ", model.ActiveProviders)}]");
        Console.WriteLine();

        // Warmup
        Console.WriteLine("Warming up...");
        await model.WarmupAsync();

        // Test 1: Basic transcription (no timestamps)
        Console.WriteLine("\n--- Test 1: Basic Transcription ---");
        var result1 = await model.TranscribeAsync(audioPath, new TranscribeOptions
        {
            WordTimestamps = false
        });
        PrintResult("Basic", result1);

        // Test 2: With segment timestamps
        Console.WriteLine("\n--- Test 2: Segment Timestamps ---");
        var result2 = await model.TranscribeAsync(audioPath, new TranscribeOptions
        {
            WordTimestamps = true
        });
        PrintResult("Timestamps", result2);

        // Test 3: With language specification (Korean)
        Console.WriteLine("\n--- Test 3: Language Specified (Korean) ---");
        var result3 = await model.TranscribeAsync(audioPath, new TranscribeOptions
        {
            Language = "ko",
            WordTimestamps = true
        });
        PrintResult("Korean", result3);

        // Save results to JSON
        var outputPath = Path.ChangeExtension(audioPath, ".transcription.json");
        await SaveResultsAsync(outputPath, new
        {
            AudioFile = audioPath,
            Model = model.ModelId,
            GpuActive = model.IsGpuActive,
            Providers = model.ActiveProviders,
            BasicResult = result1,
            TimestampResult = result2,
            KoreanResult = result3
        });
        Console.WriteLine($"\nResults saved to: {outputPath}");
    }

    private static void PrintResult(string testName, TranscriptionResult result)
    {
        Console.WriteLine($"Language: {result.Language}");
        Console.WriteLine($"Duration: {result.DurationSeconds:F2}s");
        Console.WriteLine($"Inference: {result.InferenceTimeMs:F0}ms");
        Console.WriteLine($"RTF: {result.RealTimeFactor:F2}x (>1 = faster than real-time)");
        Console.WriteLine($"Segments: {result.Segments.Count}");
        Console.WriteLine($"Text preview: {Truncate(result.Text, 100)}");

        if (result.Segments.Count > 0)
        {
            Console.WriteLine("First segments:");
            foreach (var segment in result.Segments.Take(3))
            {
                Console.WriteLine($"  {segment}");
            }
            if (result.Segments.Count > 3)
            {
                Console.WriteLine($"  ... and {result.Segments.Count - 3} more segments");
            }
        }
    }

    private static async Task SaveResultsAsync(string path, object results)
    {
        var json = JsonSerializer.Serialize(results, JsonOptions);
        await File.WriteAllTextAsync(path, json);
    }

    private static string Truncate(string text, int maxLength)
    {
        if (string.IsNullOrEmpty(text) || text.Length <= maxLength)
            return text;
        return text[..maxLength] + "...";
    }

    private static ExecutionProvider ParseProvider(string value)
    {
        return value.ToLowerInvariant() switch
        {
            "cpu" => ExecutionProvider.Cpu,
            "cuda" => ExecutionProvider.Cuda,
            "coreml" => ExecutionProvider.CoreML,
            _ => ExecutionProvider.Auto
        };
    }
}
