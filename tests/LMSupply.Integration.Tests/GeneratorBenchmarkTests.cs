using System.Diagnostics;
using LMSupply.Generator;
using LMSupply.Generator.Models;

namespace LMSupply.Integration.Tests;

/// <summary>
/// Benchmark tests for Generator models.
/// Measures tokens per second and time to first token.
/// Run locally only - requires GPU.
/// </summary>
[Trait("Category", "Integration")]
[Trait("Category", "LocalOnly")]
[Trait("Category", "Benchmark")]
public class GeneratorBenchmarkTests
{
    [Fact]
    public async Task GgufFast_Benchmark_MeasuresPerformance()
    {
        System.Console.WriteLine("=== GGUF Generator Benchmark (fast) ===\n");

        // Load model
        var loadSw = Stopwatch.StartNew();
        await using var model = await LocalGenerator.LoadAsync("gguf:gemma4-fast", cancellationToken: TestContext.Current.CancellationToken);
        loadSw.Stop();

        var info = model.GetModelInfo();
        System.Console.WriteLine($"Model: {info.ModelId}");
        System.Console.WriteLine($"Context: {info.MaxContextLength}");
        System.Console.WriteLine($"Load time: {loadSw.ElapsedMilliseconds}ms");
        System.Console.WriteLine();

        // Runtime info from model
        System.Console.WriteLine("=== Runtime ===");
        System.Console.WriteLine($"Runtime: {info.RuntimeVersion ?? "unknown"}");
        System.Console.WriteLine($"Provider: {info.ExecutionProvider}");
        System.Console.WriteLine($"GPU Active: {model.IsGpuActive}");
        System.Console.WriteLine($"Providers: {string.Join(", ", model.ActiveProviders)}");
        System.Console.WriteLine();

        // Warmup
        System.Console.WriteLine("Warming up...");
        for (int i = 0; i < 2; i++)
        {
            await foreach (var _ in model.GenerateAsync("Hello", new GenerationOptions { MaxTokens = 20 }, TestContext.Current.CancellationToken)) { }
        }

        // Benchmark
        System.Console.WriteLine("\n=== Benchmark Results ===\n");
        System.Console.WriteLine($"{"Test",-25} {"Tokens",-8} {"Gen(ms)",-10} {"Tok/s",-10} {"TTFT(ms)",-10}");
        System.Console.WriteLine(new string('=', 70));

        var tests = new (string Name, string Prompt, int MaxTokens)[]
        {
            ("Short generation", "Hello, my name is", 100),
            ("Medium generation", "Explain machine learning:", 200),
            ("Long generation", "Once upon a time,", 500),
        };

        var allResults = new List<(string name, int tokens, double tokPerSec, long ttft)>();

        foreach (var (name, prompt, maxTokens) in tests)
        {
            var runs = new List<(int tokens, long totalMs, long ttft)>();

            for (int run = 0; run < 3; run++)
            {
                var sw = Stopwatch.StartNew();
                var tokenCount = 0;
                long firstTokenTime = 0;

                await foreach (var token in model.GenerateAsync(prompt, new GenerationOptions { MaxTokens = maxTokens }, TestContext.Current.CancellationToken))
                {
                    if (tokenCount == 0)
                        firstTokenTime = sw.ElapsedMilliseconds;
                    tokenCount++;
                }

                sw.Stop();
                runs.Add((tokenCount, sw.ElapsedMilliseconds, firstTokenTime));
            }

            var avgTokens = runs.Average(r => r.tokens);
            var avgTotalMs = runs.Average(r => r.totalMs);
            var avgTtft = runs.Average(r => r.ttft);
            var genTimeMs = avgTotalMs - avgTtft;
            var tokensPerSec = avgTokens / (genTimeMs / 1000.0);

            System.Console.WriteLine($"{name,-25} {avgTokens,-8:F0} {genTimeMs,-10:F0} {tokensPerSec,-10:F1} {avgTtft,-10:F0}");
            allResults.Add((name, (int)avgTokens, tokensPerSec, (long)avgTtft));
        }

        System.Console.WriteLine();

        // Assertions
        var avgTokPerSec = allResults.Average(r => r.tokPerSec);
        avgTokPerSec.Should().BeGreaterThan(50, "Should achieve at least 50 tokens/sec on GPU");

        System.Console.WriteLine($"Average: {avgTokPerSec:F1} tokens/sec");
    }

    [Fact]
    public async Task GgufFast_ChatCompletion_Benchmark()
    {
        System.Console.WriteLine("=== Chat Completion Benchmark ===\n");

        await using var model = await LocalGenerator.LoadAsync("gguf:gemma4-fast", cancellationToken: TestContext.Current.CancellationToken);

        // Warmup
        var warmupMessages = new[] { ChatMessage.User("Hi") };
        await foreach (var _ in model.GenerateChatAsync(warmupMessages, new GenerationOptions { MaxTokens = 10 }, TestContext.Current.CancellationToken)) { }

        // Test different conversation lengths
        var tests = new (string Name, ChatMessage[] Messages)[]
        {
            ("Single turn", new[]
            {
                ChatMessage.System("You are helpful."),
                ChatMessage.User("What is 2+2?")
            }),
            ("Multi-turn", new[]
            {
                ChatMessage.System("You are a math tutor."),
                ChatMessage.User("What is 2+2?"),
                ChatMessage.Assistant("2+2 equals 4."),
                ChatMessage.User("What about 3+3?")
            }),
            ("Long context", new[]
            {
                ChatMessage.System("You are an expert assistant. Provide detailed explanations."),
                ChatMessage.User("Explain quantum computing in simple terms, including superposition and entanglement.")
            }),
        };

        System.Console.WriteLine($"{"Test",-20} {"Tokens",-8} {"Gen(ms)",-10} {"Tok/s",-10} {"TTFT(ms)",-10}");
        System.Console.WriteLine(new string('=', 65));

        foreach (var (name, messages) in tests)
        {
            var runs = new List<(int tokens, long totalMs, long ttft)>();

            for (int run = 0; run < 3; run++)
            {
                var sw = Stopwatch.StartNew();
                var tokenCount = 0;
                long firstTokenTime = 0;

                await foreach (var token in model.GenerateChatAsync(messages, new GenerationOptions { MaxTokens = 150 }, TestContext.Current.CancellationToken))
                {
                    if (tokenCount == 0)
                        firstTokenTime = sw.ElapsedMilliseconds;
                    tokenCount++;
                }

                sw.Stop();
                runs.Add((tokenCount, sw.ElapsedMilliseconds, firstTokenTime));
            }

            var avgTokens = runs.Average(r => r.tokens);
            var avgTotalMs = runs.Average(r => r.totalMs);
            var avgTtft = runs.Average(r => r.ttft);
            var genTimeMs = avgTotalMs - avgTtft;
            var tokensPerSec = avgTokens / (genTimeMs / 1000.0);

            System.Console.WriteLine($"{name,-20} {avgTokens,-8:F0} {genTimeMs,-10:F0} {tokensPerSec,-10:F1} {avgTtft,-10:F0}");
        }
    }
}
