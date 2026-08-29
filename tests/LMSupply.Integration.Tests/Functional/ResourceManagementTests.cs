using LMSupply.Captioner;
using LMSupply.Detector;
using LMSupply.Embedder;
using LMSupply.Integration.Tests.Helpers;
using LMSupply.Ocr;
using LMSupply.Reranker;
using LMSupply.Segmenter;
using LMSupply.Synthesizer;
using LMSupply.Transcriber;
using LMSupply.Translator;

namespace LMSupply.Integration.Tests.Functional;

/// <summary>
/// Resource management (R axis) tests across all domains.
/// Tests dispose patterns, double-dispose, use-after-dispose, concurrent loading.
/// Requires GPU + network access. Run locally only.
/// </summary>
[Trait("Category", "Functional")]
[Trait("Category", "LocalOnly")]
[Trait("Axis", "Resource")]
public class ResourceManagementTests
{
    // ── Double Dispose ──────────────────────────────────────────────
    // ValueTask DisposeAsync — call directly, assert no exception

    [Fact]
    public async Task R_Embedder_DoubleDispose_NoException()
    {
        var model = await LocalEmbedder.LoadAsync("fast", cancellationToken: TestContext.Current.CancellationToken);
        await model.DisposeAsync();
        await model.DisposeAsync(); // Should not throw
    }

    [Fact]
    public async Task R_Reranker_DoubleDispose_NoException()
    {
        var model = await LocalReranker.LoadAsync("fast", cancellationToken: TestContext.Current.CancellationToken);
        await model.DisposeAsync();
        await model.DisposeAsync();
    }

    [Fact]
    public async Task R_Translator_DoubleDispose_NoException()
    {
        var model = await LocalTranslator.LoadAsync("ko-en", cancellationToken: TestContext.Current.CancellationToken);
        await model.DisposeAsync();
        await model.DisposeAsync();
    }

    [Fact]
    public async Task R_Transcriber_DoubleDispose_NoException()
    {
        var model = await LocalTranscriber.LoadAsync("fast", cancellationToken: TestContext.Current.CancellationToken);
        await model.DisposeAsync();
        await model.DisposeAsync();
    }

    [Fact]
    public async Task R_Synthesizer_DoubleDispose_NoException()
    {
        var model = await LocalSynthesizer.LoadAsync("fast", cancellationToken: TestContext.Current.CancellationToken);
        await model.DisposeAsync();
        await model.DisposeAsync();
    }

    [Fact]
    public async Task R_Detector_DoubleDispose_NoException()
    {
        var model = await LocalDetector.LoadAsync("fast", cancellationToken: TestContext.Current.CancellationToken);
        await model.DisposeAsync();
        await model.DisposeAsync();
    }

    [Fact]
    public async Task R_Segmenter_DoubleDispose_NoException()
    {
        var model = await LocalSegmenter.LoadAsync("fast", cancellationToken: TestContext.Current.CancellationToken);
        await model.DisposeAsync();
        await model.DisposeAsync();
    }

    [Fact]
    public async Task R_Captioner_DoubleDispose_NoException()
    {
        var model = await LocalCaptioner.LoadAsync("fast", cancellationToken: TestContext.Current.CancellationToken);
        await model.DisposeAsync();
        await model.DisposeAsync();
    }

    // ── Use After Dispose ───────────────────────────────────────────

    [Fact]
    public async Task R_Embedder_UseAfterDispose_ThrowsException()
    {
        var model = await LocalEmbedder.LoadAsync("fast", cancellationToken: TestContext.Current.CancellationToken);
        await model.DisposeAsync();

        Func<Task> act = async () => await model.EmbedAsync("test");
        await act.Should().ThrowAsync<Exception>();
    }

    [Fact]
    public async Task R_Reranker_UseAfterDispose_ThrowsException()
    {
        var model = await LocalReranker.LoadAsync("fast", cancellationToken: TestContext.Current.CancellationToken);
        await model.DisposeAsync();

        string[] docs = ["doc1"];
        Func<Task> act = () => model.RerankAsync("query", docs);
        await act.Should().ThrowAsync<Exception>();
    }

    [Fact]
    public async Task R_Translator_UseAfterDispose_ThrowsException()
    {
        var model = await LocalTranslator.LoadAsync("ko-en", cancellationToken: TestContext.Current.CancellationToken);
        await model.DisposeAsync();

        Func<Task> act = () => model.TranslateAsync("테스트");
        await act.Should().ThrowAsync<Exception>();
    }

    // ── Model Swap (Dispose → Load new) ─────────────────────────────

    [Fact]
    public async Task R_Embedder_ModelSwap_WorksCleanly()
    {
        await using (var modelA = await LocalEmbedder.LoadAsync("fast", cancellationToken: TestContext.Current.CancellationToken))
        {
            var embA = await modelA.EmbedAsync("test", TestContext.Current.CancellationToken);
            embA.Length.Should().BeGreaterThan(0);
        }

        await using (var modelB = await LocalEmbedder.LoadAsync("fast", cancellationToken: TestContext.Current.CancellationToken))
        {
            var embB = await modelB.EmbedAsync("test", TestContext.Current.CancellationToken);
            embB.Length.Should().BeGreaterThan(0);
        }
    }

    [Fact]
    public async Task R_Reranker_ModelSwap_WorksCleanly()
    {
        await using (var modelA = await LocalReranker.LoadAsync("fast", cancellationToken: TestContext.Current.CancellationToken))
        {
            string[] docs = ["hello"];
            var result = await modelA.RerankAsync("test", docs, cancellationToken: TestContext.Current.CancellationToken);
            result.Should().HaveCount(1);
        }

        await using (var modelB = await LocalReranker.LoadAsync("fast", cancellationToken: TestContext.Current.CancellationToken))
        {
            string[] docs = ["hello"];
            var result = await modelB.RerankAsync("test", docs, cancellationToken: TestContext.Current.CancellationToken);
            result.Should().HaveCount(1);
        }
    }

    // ── Full Lifecycle ──────────────────────────────────────────────

    [Fact]
    public async Task R_Embedder_FullLifecycle_NoLeak()
    {
        await using var model = await LocalEmbedder.LoadAsync("fast", cancellationToken: TestContext.Current.CancellationToken);

        await model.WarmupAsync(TestContext.Current.CancellationToken);

        for (var i = 0; i < 5; i++)
        {
            var emb = await model.EmbedAsync($"iteration {i}", TestContext.Current.CancellationToken);
            emb.Length.Should().Be(model.Dimensions);
        }
    }

    [Fact]
    public async Task R_Synthesizer_FullLifecycle_NoLeak()
    {
        await using var model = await LocalSynthesizer.LoadAsync("fast", cancellationToken: TestContext.Current.CancellationToken);

        for (var i = 0; i < 3; i++)
        {
            var result = await model.SynthesizeAsync($"Test sentence number {i}", cancellationToken: TestContext.Current.CancellationToken);
            result.AudioSamples.Should().NotBeEmpty();
        }
    }

    // ── Concurrent Inference (thread safety) ────────────────────────

    [Fact]
    public async Task R_Embedder_ConcurrentInference_AllComplete()
    {
        await using var model = await LocalEmbedder.LoadAsync("fast", cancellationToken: TestContext.Current.CancellationToken);

        var tasks = Enumerable.Range(0, 10)
            .Select(i => model.EmbedAsync($"concurrent text {i}").AsTask())
            .ToArray();

        var results = await Task.WhenAll(tasks);

        results.Should().HaveCount(10);
        foreach (var emb in results)
        {
            emb.Length.Should().Be(model.Dimensions);
        }
    }

    [Fact]
    public async Task R_Reranker_ConcurrentInference_AllComplete()
    {
        await using var model = await LocalReranker.LoadAsync("fast", cancellationToken: TestContext.Current.CancellationToken);

        string[] docs = ["alpha", "beta", "gamma"];
        var tasks = Enumerable.Range(0, 5)
            .Select(i => model.RerankAsync($"query {i}", docs))
            .ToArray();

        var results = await Task.WhenAll(tasks);

        results.Should().HaveCount(5);
        foreach (var ranked in results)
        {
            ranked.Should().HaveCount(3);
        }
    }

    // ── Use After Dispose (remaining domains) ─────────────────────

    [Fact]
    public async Task R_Detector_UseAfterDispose_ThrowsException()
    {
        var model = await LocalDetector.LoadAsync("fast", cancellationToken: TestContext.Current.CancellationToken);
        await model.DisposeAsync();

        var imageBytes = TestDataHelper.CreateGradientBmp(64, 64);
        Func<Task> act = () => model.DetectAsync(imageBytes);
        await act.Should().ThrowAsync<Exception>();
    }

    [Fact]
    public async Task R_Captioner_UseAfterDispose_ThrowsException()
    {
        var model = await LocalCaptioner.LoadAsync("fast", cancellationToken: TestContext.Current.CancellationToken);
        await model.DisposeAsync();

        var imageBytes = TestDataHelper.CreateGradientBmp(64, 64);
        Func<Task> act = () => model.CaptionAsync(imageBytes);
        await act.Should().ThrowAsync<Exception>();
    }

    [Fact]
    public async Task R_Segmenter_UseAfterDispose_ThrowsException()
    {
        var model = await LocalSegmenter.LoadAsync("fast", cancellationToken: TestContext.Current.CancellationToken);
        await model.DisposeAsync();

        var imageBytes = TestDataHelper.CreateGradientBmp(64, 64);
        Func<Task> act = () => model.SegmentAsync(imageBytes);
        await act.Should().ThrowAsync<Exception>();
    }

    [Fact]
    public async Task R_Synthesizer_UseAfterDispose_ThrowsException()
    {
        var model = await LocalSynthesizer.LoadAsync("fast", cancellationToken: TestContext.Current.CancellationToken);
        await model.DisposeAsync();

        Func<Task> act = () => model.SynthesizeAsync("test");
        await act.Should().ThrowAsync<Exception>();
    }

    [Fact]
    public async Task R_Transcriber_UseAfterDispose_ThrowsException()
    {
        var model = await LocalTranscriber.LoadAsync("fast", cancellationToken: TestContext.Current.CancellationToken);
        await model.DisposeAsync();

        var wavBytes = TestDataHelper.CreateToneWav(16000, 1.0f, 440);
        Func<Task> act = () => model.TranscribeAsync(wavBytes);
        await act.Should().ThrowAsync<Exception>();
    }

    // ── Full Lifecycle (more domains) ─────────────────────────────

    [Fact]
    public async Task R_Translator_FullLifecycle_NoLeak()
    {
        await using var model = await LocalTranslator.LoadAsync("ko-en", cancellationToken: TestContext.Current.CancellationToken);

        for (var i = 0; i < 3; i++)
        {
            var result = await model.TranslateAsync($"테스트 문장 {i}번", TestContext.Current.CancellationToken);
            result.TranslatedText.Should().NotBeNullOrEmpty();
        }
    }

    [Fact]
    public async Task R_Detector_FullLifecycle_NoLeak()
    {
        await using var model = await LocalDetector.LoadAsync("fast", cancellationToken: TestContext.Current.CancellationToken);

        for (var i = 0; i < 3; i++)
        {
            var imageBytes = TestDataHelper.CreateGradientBmp(200 + i * 50, 200);
            var detections = await model.DetectAsync(imageBytes, TestContext.Current.CancellationToken);
            detections.Should().NotBeNull();
        }
    }

    // ── OCR Resource Management ──────────────────────────────────────

    [Fact]
    public async Task R_Ocr_DoubleDispose_NoException()
    {
        var model = await LocalOcr.LoadAsync("fast", cancellationToken: TestContext.Current.CancellationToken);
        await model.DisposeAsync();
        await model.DisposeAsync();
    }

    [Fact]
    public async Task R_Ocr_UseAfterDispose_ThrowsException()
    {
        var model = await LocalOcr.LoadAsync("fast", cancellationToken: TestContext.Current.CancellationToken);
        await model.DisposeAsync();

        var imageBytes = TestDataHelper.CreateGradientBmp(200, 50);
        Func<Task> act = () => model.RecognizeAsync(imageBytes);
        await act.Should().ThrowAsync<Exception>();
    }

    // ── Concurrent Model Loading ─────────────────────────────────────

    [Fact]
    public async Task R_ConcurrentLoading_TwoDifferentDomains_BothSucceed()
    {
        var embedderTask = LocalEmbedder.LoadAsync("fast", cancellationToken: TestContext.Current.CancellationToken);
        var rerankerTask = LocalReranker.LoadAsync("fast", cancellationToken: TestContext.Current.CancellationToken);

        await Task.WhenAll(embedderTask, rerankerTask);

        await using var embedder = await embedderTask;
        await using var reranker = await rerankerTask;

        var emb = await embedder.EmbedAsync("test", TestContext.Current.CancellationToken);
        emb.Length.Should().BeGreaterThan(0);

        string[] docs = ["hello"];
        var ranked = await reranker.RerankAsync("test", docs, cancellationToken: TestContext.Current.CancellationToken);
        ranked.Should().HaveCount(1);
    }

    // ── Model Swap (additional domains) ──────────────────────────────

    [Fact]
    public async Task R_Captioner_ModelSwap_WorksCleanly()
    {
        await using (var modelA = await LocalCaptioner.LoadAsync("fast", cancellationToken: TestContext.Current.CancellationToken))
        {
            var imageBytes = TestDataHelper.CreateGradientBmp(256, 256);
            var caption = await modelA.CaptionAsync(imageBytes, TestContext.Current.CancellationToken);
            caption.Caption.Should().NotBeNullOrEmpty();
        }

        await using (var modelB = await LocalCaptioner.LoadAsync("fast", cancellationToken: TestContext.Current.CancellationToken))
        {
            var imageBytes = TestDataHelper.CreateGradientBmp(256, 256);
            var caption = await modelB.CaptionAsync(imageBytes, TestContext.Current.CancellationToken);
            caption.Caption.Should().NotBeNullOrEmpty();
        }
    }

    [Fact]
    public async Task R_Segmenter_ModelSwap_WorksCleanly()
    {
        await using (var modelA = await LocalSegmenter.LoadAsync("fast", cancellationToken: TestContext.Current.CancellationToken))
        {
            var imageBytes = TestDataHelper.CreateGradientBmp(256, 256);
            var result = await modelA.SegmentAsync(imageBytes, TestContext.Current.CancellationToken);
            result.ClassMap.Should().NotBeEmpty();
        }

        await using (var modelB = await LocalSegmenter.LoadAsync("fast", cancellationToken: TestContext.Current.CancellationToken))
        {
            var imageBytes = TestDataHelper.CreateGradientBmp(256, 256);
            var result = await modelB.SegmentAsync(imageBytes, TestContext.Current.CancellationToken);
            result.ClassMap.Should().NotBeEmpty();
        }
    }
}
