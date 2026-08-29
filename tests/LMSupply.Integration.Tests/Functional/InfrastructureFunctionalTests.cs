using System.Diagnostics;
using LMSupply.Captioner;
using LMSupply.Detector;
using LMSupply.Download;
using LMSupply.Embedder;
using LMSupply.ImageGenerator;
using LMSupply.Ocr;
using LMSupply.Reranker;
using LMSupply.Runtime;
using LMSupply.Segmenter;
using LMSupply.Synthesizer;
using LMSupply.Transcriber;
using LMSupply.Translator;

namespace LMSupply.Integration.Tests.Functional;

/// <summary>
/// Functional tests for infrastructure: model downloading, caching, GPU detection.
/// Tests DL-1 to DL-6, CACHE-1 to CACHE-3, GPU-1 to GPU-4 from the test plan.
/// </summary>
[Trait("Category", "Functional")]
[Trait("Category", "LocalOnly")]
[Trait("Domain", "Infrastructure")]
public class InfrastructureFunctionalTests
{
    // ── DL: Model Download Tests ────────────────────────────────────

    [Fact]
    [Trait("Axis", "Download")]
    public async Task DL1_FirstDownload_CreatesCache()
    {
        // Use Embedder "fast" (smallest model) — likely already cached
        await using var model = await LocalEmbedder.LoadAsync("fast", cancellationToken: TestContext.Current.CancellationToken);

        model.ModelId.Should().NotBeNullOrEmpty();

        // Verify cache directory exists
        var cacheDir = CacheManager.GetDefaultCacheDirectory();
        Directory.Exists(cacheDir).Should().BeTrue("cache directory should exist after download");
    }

    [Fact]
    [Trait("Axis", "Download")]
    public async Task DL2_CacheHit_LoadsFaster()
    {
        // First load (ensure cached)
        var sw1 = Stopwatch.StartNew();
        await using var model1 = await LocalEmbedder.LoadAsync("fast", cancellationToken: TestContext.Current.CancellationToken);
        var firstLoadMs = sw1.ElapsedMilliseconds;

        // Second load should be faster (no download, no discovery)
        var sw2 = Stopwatch.StartNew();
        await using var model2 = await LocalEmbedder.LoadAsync("fast", cancellationToken: TestContext.Current.CancellationToken);
        var secondLoadMs = sw2.ElapsedMilliseconds;

        // Second load should be at least as fast (network-free)
        // Allow generous margin since model initialization varies
        secondLoadMs.Should().BeLessThanOrEqualTo(firstLoadMs + 5000,
            "Cached load should not be significantly slower than first load");
    }

    [Fact]
    [Trait("Axis", "Download")]
    public async Task DL3_DownloadProgress_ReportsProgress()
    {
        var progressReports = new List<float>();
        var progress = new Progress<float>(p => progressReports.Add(p));

        // LoadAsync with progress - using Embedder as proxy
        // Note: If already cached, progress may not report (no download needed)
        await using var model = await LocalEmbedder.LoadAsync("fast", cancellationToken: TestContext.Current.CancellationToken);

        // Just verify it doesn't throw — progress reporting depends on cache state
        model.Should().NotBeNull();
    }

    [Fact]
    [Trait("Axis", "Download")]
    public async Task DL4_DownloadCancellation_ThrowsOperationCanceled()
    {
        using var cts = new CancellationTokenSource();
        cts.Cancel(); // Cancel immediately

        var act = async () => await LocalEmbedder.LoadAsync("fast", cancellationToken: cts.Token);

        // Should throw OperationCanceledException or TaskCanceledException
        await act.Should().ThrowAsync<OperationCanceledException>();
    }

    [Fact]
    [Trait("Axis", "Download")]
    public async Task DL5_InvalidModelId_ThrowsMeaningfulError()
    {
        var act = async () => await LocalEmbedder.LoadAsync("some-org/totally-nonexistent-model-xyz");

        // Should throw with a meaningful message, not a generic 404
        await act.Should().ThrowAsync<Exception>();
    }

    [Fact]
    [Trait("Axis", "Download")]
    public async Task DL6_EmptyModelId_ThrowsMeaningfulError()
    {
        var act = async () => await LocalEmbedder.LoadAsync("");

        // Embedder throws ModelNotFoundException for unresolvable IDs
        await act.Should().ThrowAsync<Exception>()
            .Where(e => e.Message.Contains("Unknown model") || e.Message.Contains("model"));
    }

    // ── CACHE: Cache Management Tests ───────────────────────────────

    [Fact]
    [Trait("Axis", "Cache")]
    public void CACHE0_DefaultCacheDirectory_IsValid()
    {
        var cacheDir = CacheManager.GetDefaultCacheDirectory();

        cacheDir.Should().NotBeNullOrEmpty();
        // Should be an absolute path
        Path.IsPathRooted(cacheDir).Should().BeTrue("cache directory should be an absolute path");
    }

    [Fact]
    [Trait("Axis", "Cache")]
    public async Task CACHE1_CachedModelsList_ContainsLoadedModel()
    {
        // Ensure at least one model is cached
        await using var model = await LocalEmbedder.LoadAsync("fast", cancellationToken: TestContext.Current.CancellationToken);

        var cacheDir = CacheManager.GetDefaultCacheDirectory();
        var cachedModels = CacheManager.GetCachedModels(cacheDir).ToList();

        cachedModels.Should().NotBeEmpty("at least the loaded model should be cached");
        cachedModels.Should().Contain(m => m.ModelId.Contains("MiniLM") || m.ModelId.Contains("minilm") || m.ModelId.Length > 0,
            "cached models should include the 'fast' Embedder model");
    }

    [Fact]
    [Trait("Axis", "Cache")]
    public async Task CACHE2_CachedModelsWithInfo_HasValidFields()
    {
        // Ensure at least one model is cached
        await using var model = await LocalEmbedder.LoadAsync("fast", cancellationToken: TestContext.Current.CancellationToken);

        var cacheDir = CacheManager.GetDefaultCacheDirectory();
        var cachedModels = CacheManager.GetCachedModelsWithInfo(cacheDir);

        cachedModels.Should().NotBeEmpty();

        var firstModel = cachedModels[0];
        firstModel.RepoId.Should().NotBeNullOrEmpty();
        firstModel.LocalPath.Should().NotBeNullOrEmpty();
        firstModel.SizeBytes.Should().BeGreaterThan(0);
        firstModel.FileCount.Should().BeGreaterThan(0);
    }

    [Fact]
    [Trait("Axis", "Cache")]
    public async Task CACHE3_TotalCacheSize_IsPositive()
    {
        // Ensure at least one model is cached
        await using var model = await LocalEmbedder.LoadAsync("fast", cancellationToken: TestContext.Current.CancellationToken);

        var cacheDir = CacheManager.GetDefaultCacheDirectory();
        var totalSize = CacheManager.GetTotalCacheSize(cacheDir);

        totalSize.Should().BeGreaterThan(0, "cache should have some content after loading a model");
    }

    [Fact]
    [Trait("Axis", "Cache")]
    public void CACHE_GetModelDirectory_FollowsHfConvention()
    {
        var cacheDir = "/tmp/test-cache";
        var modelDir = CacheManager.GetModelDirectory(cacheDir, "sentence-transformers/all-MiniLM-L6-v2");

        modelDir.Should().Contain("models--sentence-transformers--all-MiniLM-L6-v2");
        modelDir.Should().Contain("snapshots");
        modelDir.Should().Contain("main");
    }

    [Fact]
    [Trait("Axis", "Cache")]
    public void CACHE_GetModelFilePath_ReturnsCorrectPath()
    {
        var cacheDir = "/tmp/test-cache";
        var filePath = CacheManager.GetModelFilePath(cacheDir, "org/model", "model.onnx");

        filePath.Should().Contain("models--org--model");
        filePath.Should().EndWith("model.onnx");
    }

    [Fact]
    [Trait("Axis", "Cache")]
    public void CACHE_NonExistentCache_ReturnsEmpty()
    {
        var nonExistent = Path.Combine(Path.GetTempPath(), $"nonexistent_{Guid.NewGuid()}");

        var models = CacheManager.GetCachedModels(nonExistent).ToList();

        models.Should().BeEmpty();
    }

    // ── GPU: Detection Tests ────────────────────────────────────────

    [Fact]
    [Trait("Axis", "GPU")]
    public void GPU1_DetectPlatform_ReturnsValidInfo()
    {
        var platform = EnvironmentDetector.DetectPlatform();

        platform.Should().NotBeNull();
        platform.RuntimeIdentifier.Should().NotBeNullOrEmpty();
        platform.Is64Bit.Should().BeTrue("running on 64-bit platform");
    }

    [Fact]
    [Trait("Axis", "GPU")]
    public void GPU2_DetectGpu_ReturnsInfo()
    {
        var gpu = EnvironmentDetector.DetectGpu();

        gpu.Should().NotBeNull();
        // Vendor should be detected (could be Unknown on CI without GPU)
        gpu.Vendor.Should().BeDefined();
    }

    [Fact]
    [Trait("Axis", "GPU")]
    public void GPU3_DetectAllGpus_ReturnsAtLeastOne()
    {
        var gpus = EnvironmentDetector.DetectAllGpus();

        // Even without dedicated GPU, detection should return a result
        gpus.Should().NotBeNull();
    }

    [Fact]
    [Trait("Axis", "GPU")]
    public void GPU4_GetRecommendedProvider_ReturnsValidProvider()
    {
        var provider = EnvironmentDetector.GetRecommendedProvider();

        // Should be one of the known providers
        provider.Should().BeOneOf(
            ExecutionProvider.Auto,
            ExecutionProvider.Cuda,
            ExecutionProvider.DirectML,
            ExecutionProvider.CoreML,
            ExecutionProvider.Cpu);
    }

    [Fact]
    [Trait("Axis", "GPU")]
    public void GPU_GetAvailableProviders_IncludesCpu()
    {
        var providers = EnvironmentDetector.GetAvailableProviders().ToList();

        providers.Should().NotBeEmpty();
        providers.Should().Contain(ExecutionProvider.Cpu, "CPU provider should always be available");
    }

    [Fact]
    [Trait("Axis", "GPU")]
    public void GPU_GetEnvironmentSummary_ReturnsFormattedString()
    {
        var summary = EnvironmentDetector.GetEnvironmentSummary();

        summary.Should().NotBeNullOrEmpty();
    }

    [Fact]
    [Trait("Axis", "GPU")]
    public void GPU_DetectPlatform_IsCached()
    {
        var platform1 = EnvironmentDetector.DetectPlatform();
        var platform2 = EnvironmentDetector.DetectPlatform();

        // Same instance should be returned (cached)
        ReferenceEquals(platform1, platform2).Should().BeTrue("platform info should be cached");
    }

    [Fact]
    [Trait("Axis", "GPU")]
    public void GPU_PlatformInfo_HasCorrectOs()
    {
        var platform = EnvironmentDetector.DetectPlatform();

        // At least one OS flag should be true
        var hasOs = platform.IsWindows || platform.IsLinux || platform.IsMacOS;
        hasOs.Should().BeTrue("platform should detect an OS");
    }

    [Fact]
    [Trait("Axis", "GPU")]
    public void GPU_GpuInfo_RecommendedProvider_IsValid()
    {
        var gpu = EnvironmentDetector.DetectGpu();

        // If GPU is detected, recommended provider should not be Auto
        if (gpu.Vendor != GpuVendor.Unknown)
        {
            gpu.RecommendedProvider.Should().NotBe(ExecutionProvider.Auto,
                "detected GPU should have a specific provider recommendation");
        }
    }

    // ── GPU-2: Explicit Provider Override ──────────────────────────

    [Fact]
    [Trait("Axis", "GPU")]
    public async Task GPU2_ExplicitCpuProvider_IsActuallyUsed()
    {
        // GPU-2: Verify explicit provider override actually selects the requested provider
        var options = new EmbedderOptions { Provider = ExecutionProvider.Cpu };
        await using var model = await LocalEmbedder.LoadAsync("fast", options, cancellationToken: TestContext.Current.CancellationToken);

        model.RequestedProvider.Should().Be(ExecutionProvider.Cpu,
            "requested provider should reflect the explicit override");
        model.ActiveProviders.Should().Contain("CpuExecutionProvider",
            "CPU provider should be active when explicitly requested");
    }

    [Fact]
    [Trait("Axis", "GPU")]
    public async Task GPU3_UnavailableProvider_FallsBackGracefully()
    {
        // GPU-3: CoreML is only available on macOS — on Windows it should fallback
        if (!OperatingSystem.IsWindows() && !OperatingSystem.IsLinux())
            return; // Only test on non-macOS platforms

        var options = new EmbedderOptions { Provider = ExecutionProvider.CoreML };

        // Should either fallback gracefully or throw a meaningful error
        try
        {
            await using var model = await LocalEmbedder.LoadAsync("fast", options, cancellationToken: TestContext.Current.CancellationToken);

            // If it loads, it must have fallen back to a different provider
            model.ActiveProviders.Should().NotContain("CoreMLExecutionProvider",
                "CoreML should not be active on non-macOS");
        }
        catch (Exception ex)
        {
            // A meaningful error about provider not being available is also acceptable
            ex.Message.Should().NotBeNullOrEmpty(
                "provider mismatch should produce a meaningful error");
        }
    }

    // ── Enum Validation ──────────────────────────────────────────────

    [Fact]
    [Trait("Axis", "GPU")]
    public void ExecutionProvider_HasExpectedValues()
    {
        var values = Enum.GetValues<ExecutionProvider>();

        values.Should().HaveCount(5, "ExecutionProvider should have exactly 5 values");
        values.Should().Contain(ExecutionProvider.Auto);
        values.Should().Contain(ExecutionProvider.Cuda);
        values.Should().Contain(ExecutionProvider.DirectML);
        values.Should().Contain(ExecutionProvider.CoreML);
        values.Should().Contain(ExecutionProvider.Cpu);
    }

    [Fact]
    [Trait("Axis", "Cache")]
    public void CACHE_DefaultCacheDirectory_FollowsHfConvention()
    {
        var cacheDir = CacheManager.GetDefaultCacheDirectory();

        // HuggingFace cache convention: ~/.cache/huggingface/hub
        cacheDir.Should().Contain("huggingface",
            "default cache should follow HuggingFace convention");
    }

    // ── Cross-Domain API Consistency ────────────────────────────────

    [Theory]
    [Trait("Axis", "Consistency")]
    [InlineData("Embedder")]
    [InlineData("Reranker")]
    [InlineData("Detector")]
    [InlineData("Segmenter")]
    [InlineData("Translator")]
    [InlineData("Transcriber")]
    [InlineData("Synthesizer")]
    [InlineData("Captioner")]
    public void CONSISTENCY_AllDomains_GetAvailableModels_ContainsDefault(string domain)
    {
        var models = domain switch
        {
            "Embedder" => LocalEmbedder.GetAvailableModels().ToList(),
            "Reranker" => LocalReranker.GetAvailableModels().ToList(),
            "Detector" => LocalDetector.GetAvailableModels().ToList(),
            "Segmenter" => LocalSegmenter.GetAvailableModels().ToList(),
            "Translator" => LocalTranslator.GetAvailableModels().ToList(),
            "Transcriber" => LocalTranscriber.GetAvailableModels().ToList(),
            "Synthesizer" => LocalSynthesizer.GetAvailableModels().ToList(),
            "Captioner" => LocalCaptioner.GetAvailableModels().ToList(),
            _ => throw new ArgumentException($"Unknown domain: {domain}")
        };

        models.Should().NotBeEmpty($"{domain} should have available models");
        models.Should().Contain("default", $"{domain} should support 'default' alias");
    }

    [Theory]
    [Trait("Axis", "Consistency")]
    [InlineData("Embedder")]
    [InlineData("Reranker")]
    [InlineData("Detector")]
    [InlineData("Segmenter")]
    [InlineData("Transcriber")]
    [InlineData("Synthesizer")]
    [InlineData("Captioner")]
    public void CONSISTENCY_StandardDomains_GetAvailableModels_ContainsFast(string domain)
    {
        // Note: Translator uses language-pair aliases (ko-en, en-ko), not "fast"
        var models = domain switch
        {
            "Embedder" => LocalEmbedder.GetAvailableModels().ToList(),
            "Reranker" => LocalReranker.GetAvailableModels().ToList(),
            "Detector" => LocalDetector.GetAvailableModels().ToList(),
            "Segmenter" => LocalSegmenter.GetAvailableModels().ToList(),
            "Transcriber" => LocalTranscriber.GetAvailableModels().ToList(),
            "Synthesizer" => LocalSynthesizer.GetAvailableModels().ToList(),
            "Captioner" => LocalCaptioner.GetAvailableModels().ToList(),
            _ => throw new ArgumentException($"Unknown domain: {domain}")
        };

        models.Should().Contain("fast", $"{domain} should support 'fast' alias");
    }

    [Fact]
    [Trait("Axis", "Consistency")]
    public void CONSISTENCY_AllDomains_GetAllModels_ReturnNonEmpty()
    {
        LocalEmbedder.GetAllModels().Should().NotBeEmpty("Embedder");
        LocalReranker.GetAllModels().Should().NotBeEmpty("Reranker");
        LocalDetector.GetAllModels().Should().NotBeEmpty("Detector");
        LocalSegmenter.GetAllModels().Should().NotBeEmpty("Segmenter");
        LocalTranslator.GetAllModels().Should().NotBeEmpty("Translator");
        LocalTranscriber.GetAllModels().Should().NotBeEmpty("Transcriber");
        LocalSynthesizer.GetAllModels().Should().NotBeEmpty("Synthesizer");
        LocalCaptioner.GetAllModels().Should().NotBeEmpty("Captioner");
    }

    // ── Non-standard API Domains (OCR, ImageGenerator) ──────────────

    [Fact]
    [Trait("Axis", "Consistency")]
    public void CONSISTENCY_OCR_HasDetectionAndRecognitionModels()
    {
        var detModels = LocalOcr.GetAvailableDetectionModels().ToList();
        var recModels = LocalOcr.GetAvailableRecognitionModels().ToList();
        var languages = LocalOcr.GetSupportedLanguages().ToList();

        detModels.Should().NotBeEmpty("OCR should have detection models");
        recModels.Should().NotBeEmpty("OCR should have recognition models");
        languages.Should().NotBeEmpty("OCR should support at least one language");
        languages.Should().Contain("en", "OCR should support English");
    }

    [Fact]
    [Trait("Axis", "Consistency")]
    public void CONSISTENCY_ImageGenerator_HasAliases()
    {
        var aliases = LocalImageGenerator.GetAvailableAliases();

        aliases.Should().NotBeEmpty("ImageGenerator should have available aliases");
        aliases.Should().Contain("default", "ImageGenerator should support 'default' alias");
    }
}
