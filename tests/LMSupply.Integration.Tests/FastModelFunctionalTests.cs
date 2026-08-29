using System.Diagnostics;
using LMSupply;
using LMSupply.Captioner;
using LMSupply.Detector;
using LMSupply.Embedder;
using LMSupply.Generator;
using LMSupply.Generator.Models;
using LMSupply.Ocr;
using LMSupply.Reranker;
using LMSupply.Segmenter;
using LMSupply.Synthesizer;
using LMSupply.Transcriber;
using LMSupply.Translator;

namespace LMSupply.Integration.Tests;

/// <summary>
/// Functional tests for the fastest (smallest) model in each domain.
/// These tests download and run actual inference to verify end-to-end functionality.
/// Run locally only - requires GPU and network access.
/// </summary>
[Trait("Category", "Integration")]
[Trait("Category", "LocalOnly")]
[Trait("Category", "Functional")]
public class FastModelFunctionalTests : IAsyncLifetime
{
    private readonly List<string> _testResults = [];

    public ValueTask InitializeAsync() => ValueTask.CompletedTask;

    public ValueTask DisposeAsync()
    {
        // Print summary
        if (_testResults.Count > 0)
        {
            System.Console.WriteLine("\n=== Fast Model Test Summary ===");
            foreach (var result in _testResults)
            {
                System.Console.WriteLine(result);
            }
        }
        GC.SuppressFinalize(this);
        return ValueTask.CompletedTask;
    }

    [Fact]
    public async Task Embedder_Fast_GeneratesEmbeddings()
    {
        var sw = Stopwatch.StartNew();

        await using var model = await LocalEmbedder.LoadAsync("fast", cancellationToken: TestContext.Current.CancellationToken);
        var loadTime = sw.ElapsedMilliseconds;

        var texts = new[] { "Hello world", "Goodbye world" };
        sw.Restart();
        var embeddings = await model.EmbedAsync(texts, TestContext.Current.CancellationToken);
        var inferTime = sw.ElapsedMilliseconds;

        embeddings.Should().HaveCount(2);
        embeddings[0].Length.Should().BeGreaterThan(0);

        var similarity = LocalEmbedder.CosineSimilarity(embeddings[0], embeddings[1]);
        similarity.Should().BeGreaterThan(0.5f, "Similar sentences should have high similarity");

        _testResults.Add($"Embedder (fast): Load={loadTime}ms, Infer={inferTime}ms, Dims={embeddings[0].Length}");
    }

    [Fact]
    public async Task Reranker_Fast_RanksDocuments()
    {
        var sw = Stopwatch.StartNew();

        await using var model = await LocalReranker.LoadAsync("fast", cancellationToken: TestContext.Current.CancellationToken);
        var loadTime = sw.ElapsedMilliseconds;

        var query = "What is machine learning?";
        var documents = new[]
        {
            "Machine learning is a subset of artificial intelligence.",
            "The weather is nice today.",
            "Deep learning uses neural networks.",
        };

        sw.Restart();
        var scores = await model.RerankAsync(query, documents, cancellationToken: TestContext.Current.CancellationToken);
        var inferTime = sw.ElapsedMilliseconds;

        scores.Should().HaveCount(3);
        // ML-related docs should score higher than weather
        scores[0].Should().BeGreaterThan(scores[1]);
        scores[2].Should().BeGreaterThan(scores[1]);

        _testResults.Add($"Reranker (fast): Load={loadTime}ms, Infer={inferTime}ms");
    }

    [Fact]
    public async Task Generator_GgufFast_GeneratesText()
    {
        var sw = Stopwatch.StartNew();

        await using var model = await LocalGenerator.LoadAsync("gguf:gemma4-fast", cancellationToken: TestContext.Current.CancellationToken);
        var loadTime = sw.ElapsedMilliseconds;

        // Warmup
        await foreach (var _ in model.GenerateAsync("Hi", new GenerationOptions { MaxTokens = 5 }, TestContext.Current.CancellationToken)) { }

        // Benchmark
        var tokenCount = 0;
        long ttft = 0;
        sw.Restart();

        await foreach (var token in model.GenerateAsync("Hello, my name is", new GenerationOptions { MaxTokens = 50 }, TestContext.Current.CancellationToken))
        {
            if (tokenCount == 0) ttft = sw.ElapsedMilliseconds;
            tokenCount++;
        }

        var totalTime = sw.ElapsedMilliseconds;
        var genTime = totalTime - ttft;
        var tokensPerSec = tokenCount / (genTime / 1000.0);

        tokenCount.Should().BeGreaterThan(10);

        // Show runtime info from model
        var modelInfo = model.GetModelInfo();
        System.Console.WriteLine($"Runtime: {modelInfo.RuntimeVersion}, Backend: {modelInfo.ExecutionProvider}");

        _testResults.Add($"Generator GGUF (fast): Load={loadTime}ms, TTFT={ttft}ms, Tokens={tokenCount}, Tok/s={tokensPerSec:F1}");
    }

    [Fact]
    public async Task Translator_Default_TranslatesText()
    {
        var sw = Stopwatch.StartNew();

        await using var model = await LocalTranslator.LoadAsync("ko-en", cancellationToken: TestContext.Current.CancellationToken);
        var loadTime = sw.ElapsedMilliseconds;

        var koreanText = "안녕하세요, 만나서 반갑습니다.";
        sw.Restart();
        var result = await model.TranslateAsync(koreanText, TestContext.Current.CancellationToken);
        var inferTime = sw.ElapsedMilliseconds;

        result.Should().NotBeNull();
        result.TranslatedText.Should().NotBeNullOrEmpty();
        result.TranslatedText.ToLowerInvariant().Should().ContainAny("hello", "hi", "nice", "meet", "glad");

        _testResults.Add($"Translator (ko-en): Load={loadTime}ms, Infer={inferTime}ms");
    }

    [Fact]
    public async Task Transcriber_Fast_TranscribesAudio()
    {
        var sw = Stopwatch.StartNew();

        await using var model = await LocalTranscriber.LoadAsync("fast", cancellationToken: TestContext.Current.CancellationToken);
        var loadTime = sw.ElapsedMilliseconds;

        // Generate a simple WAV file with a tone
        var sampleRate = 16000;
        var duration = 1.0f;
        var wavBytes = CreateTestWavFile(sampleRate, duration, 440); // 440Hz tone

        sw.Restart();
        var result = await model.TranscribeAsync(wavBytes, cancellationToken: TestContext.Current.CancellationToken);
        var inferTime = sw.ElapsedMilliseconds;

        // Just verify it runs without error - tone won't produce meaningful text
        result.Should().NotBeNull();

        _testResults.Add($"Transcriber (fast): Load={loadTime}ms, Infer={inferTime}ms");
    }

    [Fact]
    public async Task Synthesizer_Fast_GeneratesSpeech()
    {
        var sw = Stopwatch.StartNew();

        await using var model = await LocalSynthesizer.LoadAsync("fast", cancellationToken: TestContext.Current.CancellationToken);
        var loadTime = sw.ElapsedMilliseconds;

        sw.Restart();
        var audio = await model.SynthesizeAsync("Hello world", cancellationToken: TestContext.Current.CancellationToken);
        var inferTime = sw.ElapsedMilliseconds;

        audio.Should().NotBeNull();
        audio.AudioSamples.Should().NotBeNull();
        audio.AudioSamples.Should().NotBeEmpty();
        audio.SampleRate.Should().BeGreaterThan(0);

        _testResults.Add($"Synthesizer (fast): Load={loadTime}ms, Infer={inferTime}ms, Samples={audio.AudioSamples.Length}");
    }

    [Fact]
    public async Task Captioner_Fast_GeneratesCaption()
    {
        var sw = Stopwatch.StartNew();

        await using var model = await LocalCaptioner.LoadAsync("fast", cancellationToken: TestContext.Current.CancellationToken);
        var loadTime = sw.ElapsedMilliseconds;

        // Create a simple test image (100x100 red square)
        var width = 100;
        var height = 100;
        var imageBytes = CreateSimpleTestImage(width, height);

        sw.Restart();
        var result = await model.CaptionAsync(imageBytes, TestContext.Current.CancellationToken);
        var inferTime = sw.ElapsedMilliseconds;

        result.Should().NotBeNull();
        result.Caption.Should().NotBeNullOrEmpty();

        _testResults.Add($"Captioner (fast): Load={loadTime}ms, Infer={inferTime}ms, Caption={result.Caption}");
    }

    [Fact]
    public async Task Ocr_Fast_ExtractsText()
    {
        var sw = Stopwatch.StartNew();

        await using var model = await LocalOcr.LoadAsync("fast", cancellationToken: TestContext.Current.CancellationToken);
        var loadTime = sw.ElapsedMilliseconds;

        // Create a simple test image
        var imageBytes = CreateSimpleTestImage(200, 50);

        sw.Restart();
        var result = await model.RecognizeAsync(imageBytes, TestContext.Current.CancellationToken);
        var inferTime = sw.ElapsedMilliseconds;

        result.Should().NotBeNull();
        // Image doesn't have text, so just verify it runs

        _testResults.Add($"OCR (fast): Load={loadTime}ms, Infer={inferTime}ms");
    }

    [Fact]
    public async Task Detector_Fast_DetectsObjects()
    {
        var sw = Stopwatch.StartNew();

        await using var model = await LocalDetector.LoadAsync("fast", cancellationToken: TestContext.Current.CancellationToken);
        var loadTime = sw.ElapsedMilliseconds;

        // Create a simple test image
        var imageBytes = CreateSimpleTestImage(640, 480);

        sw.Restart();
        var detections = await model.DetectAsync(imageBytes, TestContext.Current.CancellationToken);
        var inferTime = sw.ElapsedMilliseconds;

        detections.Should().NotBeNull();
        // Simple color image may not have recognizable objects

        _testResults.Add($"Detector (fast): Load={loadTime}ms, Infer={inferTime}ms, Detections={detections.Count}");
    }

    [Fact]
    public async Task Segmenter_Fast_SegmentsImage()
    {
        var sw = Stopwatch.StartNew();

        await using var model = await LocalSegmenter.LoadAsync("fast", cancellationToken: TestContext.Current.CancellationToken);
        var loadTime = sw.ElapsedMilliseconds;

        // Create a simple test image
        var imageBytes = CreateSimpleTestImage(256, 256);

        sw.Restart();
        var mask = await model.SegmentAsync(imageBytes, TestContext.Current.CancellationToken);
        var inferTime = sw.ElapsedMilliseconds;

        mask.Should().NotBeNull();
        mask.Width.Should().BeGreaterThan(0);
        mask.Height.Should().BeGreaterThan(0);

        _testResults.Add($"Segmenter (fast): Load={loadTime}ms, Infer={inferTime}ms, Size={mask.Width}x{mask.Height}");
    }

    #region Helper Methods

    private static byte[] CreateTestWavFile(int sampleRate, float durationSeconds, float frequency)
    {
        var numSamples = (int)(sampleRate * durationSeconds);
        var samples = new float[numSamples];

        // Generate sine wave
        for (int i = 0; i < numSamples; i++)
        {
            samples[i] = 0.1f * MathF.Sin(2 * MathF.PI * frequency * i / sampleRate);
        }

        // Convert to 16-bit PCM
        var pcmData = new byte[numSamples * 2];
        for (int i = 0; i < numSamples; i++)
        {
            var sample = Math.Clamp(samples[i], -1.0f, 1.0f);
            var pcm16 = (short)(sample * 32767);
            pcmData[i * 2] = (byte)(pcm16 & 0xFF);
            pcmData[i * 2 + 1] = (byte)((pcm16 >> 8) & 0xFF);
        }

        // Create WAV file
        using var ms = new MemoryStream();
        using var writer = new BinaryWriter(ms);

        // RIFF header
        writer.Write("RIFF"u8);
        writer.Write(36 + pcmData.Length);
        writer.Write("WAVE"u8);

        // fmt chunk
        writer.Write("fmt "u8);
        writer.Write(16); // Chunk size
        writer.Write((short)1); // PCM format
        writer.Write((short)1); // Channels
        writer.Write(sampleRate);
        writer.Write(sampleRate * 2); // Byte rate
        writer.Write((short)2); // Block align
        writer.Write((short)16); // Bits per sample

        // data chunk
        writer.Write("data"u8);
        writer.Write(pcmData.Length);
        writer.Write(pcmData);

        return ms.ToArray();
    }

    private static byte[] CreateSimpleTestImage(int width, int height)
    {
        // Create a simple BMP image (uncompressed, easy to generate)
        // This is a minimal valid BMP file
        var rowSize = ((width * 3 + 3) / 4) * 4; // Row size padded to 4 bytes
        var imageSize = rowSize * height;
        var fileSize = 54 + imageSize; // Header (54 bytes) + pixel data

        using var ms = new MemoryStream();
        using var bw = new BinaryWriter(ms);

        // BMP Header
        bw.Write((byte)'B');
        bw.Write((byte)'M');
        bw.Write(fileSize);
        bw.Write(0); // Reserved
        bw.Write(54); // Pixel data offset

        // DIB Header (BITMAPINFOHEADER)
        bw.Write(40); // Header size
        bw.Write(width);
        bw.Write(height);
        bw.Write((short)1); // Color planes
        bw.Write((short)24); // Bits per pixel
        bw.Write(0); // Compression (none)
        bw.Write(imageSize);
        bw.Write(2835); // Horizontal resolution (72 DPI)
        bw.Write(2835); // Vertical resolution
        bw.Write(0); // Colors in palette
        bw.Write(0); // Important colors

        // Pixel data (BGR format, bottom-up)
        var rowPadding = rowSize - width * 3;
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                // Create a gradient pattern
                byte r = (byte)(x * 255 / width);
                byte g = (byte)(y * 255 / height);
                byte b = (byte)((x + y) * 127 / (width + height));
                bw.Write(b); // Blue
                bw.Write(g); // Green
                bw.Write(r); // Red
            }
            // Row padding
            for (int p = 0; p < rowPadding; p++)
            {
                bw.Write((byte)0);
            }
        }

        return ms.ToArray();
    }

    #endregion
}
