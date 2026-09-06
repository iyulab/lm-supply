using System.Diagnostics;
using System.Globalization;
using LMSupply;
using LMSupply.Embedder;

namespace EmbedderSample;

/// <summary>
/// Sample application demonstrating LMSupply.Embedder capabilities.
/// Tests GPU/CPU provider selection and embedding functionality.
/// </summary>
public static class Program
{
    public static async Task Main(string[] args)
    {
        Console.WriteLine("=== LMSupply Embedder Sample ===\n");

        // Surface the library's provider-selection / fallback diagnostics (Trace warnings) on
        // stderr so a GPU→CPU fallback is visible in this sample's output rather than silent.
        Trace.Listeners.Add(new TextWriterTraceListener(Console.Error));

        var modelAlias = args.Length > 0 ? args[0] : "default";
        var provider = ParseProvider(args.Length > 1 ? args[1] : "auto");

        // Display available models
        Console.WriteLine("Available models:");
        foreach (var model in LocalEmbedder.GetAvailableModels())
        {
            Console.WriteLine($"  - {model}");
        }
        Console.WriteLine();

        await RunEmbeddingTestsAsync(modelAlias, provider);
    }

    private static async Task RunEmbeddingTestsAsync(string modelAlias, ExecutionProvider provider)
    {
        Console.WriteLine($"=== Embedding Test ===\n");
        Console.WriteLine($"Model: {modelAlias}");
        Console.WriteLine($"Provider: {provider}");
        Console.WriteLine();

        var options = new EmbedderOptions
        {
            Provider = provider
        };

        // Load model
        Console.WriteLine("Loading model...");
        var loadStart = Stopwatch.StartNew();
        await using var model = await LocalEmbedder.LoadAsync(modelAlias, options);
        loadStart.Stop();

        Console.WriteLine($"Model loaded in {loadStart.ElapsedMilliseconds}ms");
        Console.WriteLine($"Model ID: {model.ModelId}");
        Console.WriteLine($"Dimensions: {model.Dimensions}");

        var modelInfo = model.GetModelInfo();
        if (modelInfo != null)
        {
            Console.WriteLine($"Repo ID: {modelInfo.RepoId}");
            if (modelInfo.Description != null)
            {
                Console.WriteLine($"Description: {modelInfo.Description}");
            }
        }
        Console.WriteLine();

        // Warmup
        Console.WriteLine("Warming up...");
        await model.WarmupAsync();

        // Test sentences
        var sentences = new[]
        {
            "The quick brown fox jumps over the lazy dog.",
            "A fast auburn fox leaps above a sleepy canine.",
            "Machine learning models require significant computational resources.",
            "인공지능은 현대 기술의 핵심입니다.",
            "Artificial intelligence is at the core of modern technology."
        };

        // Test 1: Single embedding
        Console.WriteLine("\n--- Test 1: Single Embedding ---");
        var sw = Stopwatch.StartNew();
        var embedding1 = await model.EmbedAsync(sentences[0]);
        sw.Stop();

        Console.WriteLine($"Text: \"{sentences[0]}\"");
        Console.WriteLine($"Dimension: {embedding1.Length}");
        Console.WriteLine($"Time: {sw.ElapsedMilliseconds}ms");
        Console.WriteLine($"First 5 values: [{string.Join(", ", embedding1.Take(5).Select(v => v.ToString("F4", CultureInfo.InvariantCulture)))}]");

        // Test 2: Batch embedding
        Console.WriteLine("\n--- Test 2: Batch Embedding ---");
        sw.Restart();
        var embeddings = await model.EmbedAsync(sentences);
        sw.Stop();

        Console.WriteLine($"Batch size: {sentences.Length}");
        Console.WriteLine($"Total time: {sw.ElapsedMilliseconds}ms");
        Console.WriteLine($"Per sentence: {sw.ElapsedMilliseconds / sentences.Length}ms");

        // Test 3: Similarity comparison
        Console.WriteLine("\n--- Test 3: Similarity Comparison ---");
        var similarities = new List<(string, string, float)>();

        for (int i = 0; i < embeddings.Length; i++)
        {
            for (int j = i + 1; j < embeddings.Length; j++)
            {
                var similarity = LocalEmbedder.CosineSimilarity(embeddings[i], embeddings[j]);
                similarities.Add((sentences[i], sentences[j], similarity));
            }
        }

        Console.WriteLine("Pairwise similarities (sorted):");
        foreach (var (s1, s2, sim) in similarities.OrderByDescending(x => x.Item3))
        {
            Console.WriteLine($"  {sim:F4}: \"{Truncate(s1, 40)}\" <-> \"{Truncate(s2, 40)}\"");
        }
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
            "directml" => ExecutionProvider.DirectML,
            "coreml" => ExecutionProvider.CoreML,
            _ => ExecutionProvider.Auto
        };
    }
}
