using AwesomeAssertions;
using LMSupply.Generator.Gguf;
using LMSupply.Generator.Internal.Llama;
using LMSupply.Hardware;
using LMSupply.Llama.Gguf;
using LMSupply.Llama.Server;
using LMSupply.Runtime;
using Xunit;

namespace LMSupply.Generator.Tests;

/// <summary>
/// Automatic selection, the GPU layer fit and the context bound of a load judge a model's fit with one rule
/// (<see cref="GgufMemoryModel"/>): a model the selector recommends is the model the loader runs on the GPU the
/// selector judged. Shares the "VramBudget" collection because the usable memory honours the budget override.
/// </summary>
[Collection("VramBudget")]
public class GgufMemoryModelTests : IDisposable
{
    private const long MiB = 1024L * 1024;
    private const long GiB = 1024L * MiB;

    private readonly string? _originalBudgetEnv =
        Environment.GetEnvironmentVariable(VramBudget.BudgetOverrideEnvVar);

    public GgufMemoryModelTests()
        => Environment.SetEnvironmentVariable(VramBudget.BudgetOverrideEnvVar, null);

    public void Dispose()
    {
        Environment.SetEnvironmentVariable(VramBudget.BudgetOverrideEnvVar, _originalBudgetEnv);
        GC.SuppressFinalize(this);
    }

    // Qwen3.6-35B-A3B (IQ4_XS): the attention metadata of the real file. 40 layers, every 4th keeps a KV cache
    // (2 KV heads of 256), so the f16 cache grows 20,480 bytes per token — 640 MiB at 32,768 tokens, which is what
    // llama-server allocated when measured on Metal.
    private static readonly GgufMetadata Qwen35MoeMetadata = new()
    {
        Architecture = "qwen35moe",
        LayerCount = 40,
        EmbeddingLength = 2048,
        HeadCount = 16,
        HeadCountKv = 2,
        KeyLength = 256,
        ValueLength = 256,
        FullAttentionInterval = 4,
        ExpertCount = 256,
        ExpertUsedCount = 8,
    };

    private const long Qwen35MoeFileBytes = 17_730_509_792L;

    // Apple M5 with 32 GB, as the Metal backend reports it: 25,559 MiB total, 25,558 MiB free.
    private static readonly GpuInfo AppleSilicon32Gb = new()
    {
        Vendor = GpuVendor.Apple,
        TotalMemoryBytes = 25_559 * MiB,
        FreeMemoryBytes = 25_558 * MiB,
    };

    private static long Usable(GpuInfo gpu) => VramBudget.GetAvailableBytes(gpu);

    // ── the rule ────────────────────────────────────────────────────────────────

    [Fact]
    public void Resident_DenseIsWeightsPlusHeadroom_MixtureOfExpertsAddsItsFixedReserve()
    {
        GgufMemoryModel.ResidentBytes(10 * GiB, mixtureOfExperts: false)
            .Should().Be((long)(10 * GiB * 1.1) + 512 * MiB);
        GgufMemoryModel.ResidentBytes(10 * GiB, mixtureOfExperts: true)
            .Should().Be((long)(10 * GiB * 1.1) + 512 * MiB + GiB,
                because: "the reserve is a fixed amount, not a share of the budget");
    }

    [Fact]
    public void Resident_CountsOnlyTheWeightsOfTheOffloadedLayers()
    {
        var half = GgufMemoryModel.ResidentBytes(10 * GiB, false, gpuShare: 0.5);
        half.Should().Be((long)(10 * GiB * 1.1 * 0.5) + 512 * MiB);
        GgufMemoryModel.GpuShare(20, 40).Should().Be(0.5);
        GgufMemoryModel.GpuShare(-1, 40).Should().Be(1.0, because: "-1 offloads every layer");
        GgufMemoryModel.GpuShare(99, 40).Should().Be(1.0, because: "more layers than the model has is all of them");
        GgufMemoryModel.GpuShare(20, null).Should().Be(1.0, because: "an unknown layer count is judged as the whole model");
    }

    [Fact]
    public void Fits_IsExactlyWhatTheKvBudgetHolds()
    {
        var usable = 20 * GiB;
        var budget = GgufMemoryModel.KvBudgetBytes(usable, 10 * GiB, mixtureOfExperts: false);

        GgufMemoryModel.Fits(usable, 10 * GiB, budget, false).Should().BeTrue();
        GgufMemoryModel.Fits(usable, 10 * GiB, budget + 1, false).Should().BeFalse();
        GgufMemoryModel.Fits(0, 10 * GiB, 0, false).Should().BeFalse(because: "no GPU memory is no fit");
    }

    // ── the reported case: one rule for the selector and the loader ─────────────

    [Fact]
    public void Apple32Gb_TheSelectorsPick_IsAGpuLoadAtTheContextItJudged()
    {
        var usable = Usable(AppleSilicon32Gb);
        var quality = GgufModelRegistry.Resolve("gguf:qwen3-quality")!;

        foreach (var context in new[] { 4096, 8192, 16384, 32768 })
        {
            // Selection: the model fits at this context.
            GgufModelRegistry.Evaluate(quality, usable, 32 * GiB, context)
                .Fits.Should().BeTrue($"the selector judges it at {context} tokens");

            // Load: the same context is held on the GPU — no floor, so Auto does not fall back to CPU.
            var (safe, floored) = LlamaServerGeneratorModel.EstimateSafeContextLengthDetailed(
                Qwen35MoeFileBytes, context, gpuLayerCount: -1, Qwen35MoeMetadata, gpu: AppleSilicon32Gb);
            floored.Should().BeFalse();
            safe.Should().Be(context, $"the loader holds what the selector judged ({context})");
            LlamaServerGeneratorModel.DecideFlooredContextAction(ExecutionProvider.Auto, LlamaServerBackend.Metal, safe, context)
                .Should().Be(LlamaServerGeneratorModel.FlooredContextAction.Proceed);
        }
    }

    [Fact]
    public void Apple32Gb_TheModelAloneNeverBringsTheContextBelowWhatItsKvCacheNeeds()
    {
        // The default request of a model trained for 262,144 tokens is capped at 32,768 (DefaultContextCeiling).
        var (safe, floored) = LlamaServerGeneratorModel.EstimateSafeContextLengthDetailed(
            Qwen35MoeFileBytes, requestedContext: 32768, gpuLayerCount: -1, Qwen35MoeMetadata, gpu: AppleSilicon32Gb);

        floored.Should().BeFalse();
        safe.Should().Be(32768,
            because: "measured on Metal, this model holds a 32,768-token context in about 17.5 GiB of the 21 GiB budget");
    }

    // ── the two rules agree wherever the budget lies ────────────────────────────

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void Selector_FitsAtAContext_ExactlyWhenTheLoaderHoldsThatContext(bool mixtureOfExperts)
    {
        const long fileBytes = 12_000_000_000L;
        var metadata = new GgufMetadata
        {
            LayerCount = 32,
            EmbeddingLength = 4096,
            HeadCount = 32,
            HeadCountKv = 8,
            KeyLength = 128,
            ValueLength = 128,
            ExpertCount = mixtureOfExperts ? 64 : null,
        };
        var perToken = KvCacheGeometry.FromMetadata(metadata)!.BytesPerToken(null, null);
        var model = new GgufModelInfo
        {
            RepoId = "test/model",
            DisplayName = "Test model",
            DefaultFile = "model.gguf",
            ChatFormat = "chatml",
            ContextLength = 131072,
            ParameterCount = 20_000_000_000,
            EstimatedSizeBytes = fileBytes,
            QuantizationType = "Q4_K_M",
            KvCacheBytesPerToken = perToken,
            IsMixtureOfExperts = mixtureOfExperts,
            License = LicenseTier.MIT,
        };

        foreach (var usableGiB in new[] { 8, 12, 14, 15, 16, 17, 20, 24, 32, 48 })
        {
            var gpu = new GpuInfo { Vendor = GpuVendor.Nvidia, TotalMemoryBytes = (long)(usableGiB * GiB / 0.85) + MiB, FreeMemoryBytes = 200 * GiB };
            var usable = Usable(gpu);
            foreach (var context in new[] { 4096, 8192, 16384, 32768, 65536 })
            {
                var selected = GgufModelRegistry.Evaluate(model, usable, 64 * GiB, context).Fits;
                var (safe, _) = LlamaServerGeneratorModel.EstimateSafeContextLengthDetailed(
                    fileBytes, context, gpuLayerCount: -1, metadata, gpu: gpu);

                (safe >= context).Should().Be(selected,
                    $"usable {usable / MiB} MiB, context {context}, mixture of experts {mixtureOfExperts}: the selector and the loader judge one rule");
            }
        }
    }

    [Fact]
    public void LayerFit_TheLayersItOffloads_LeaveRoomForTheContextItWasSizedFor()
    {
        // A model larger than the GPU: the layer fit offloads part of it, and the context bound that follows must
        // find room for that context beside those layers — not floor and send an Auto load to CPU.
        const long fileBytes = 30_000_000_000L;
        const int layers = 60;
        const int context = 8192;
        var metadata = new GgufMetadata
        {
            LayerCount = layers,
            EmbeddingLength = 8192,
            HeadCount = 64,
            HeadCountKv = 8,
            KeyLength = 128,
            ValueLength = 128,
        };
        var geometry = KvCacheGeometry.FromMetadata(metadata)!;
        var gpu = new GpuInfo { Vendor = GpuVendor.Nvidia, TotalMemoryBytes = 24 * GiB, FreeMemoryBytes = 24 * GiB };
        var usable = Usable(gpu);

        var estimate = MemoryEstimator.EstimateForGgufWithKvCache(
            fileBytes, geometry.TotalBytes(context, null, null, 1, KvCacheGeometry.DefaultUBatch),
            usable, availableRamBytes: 64 * GiB, layers, mixtureOfExperts: false);

        estimate.CanFitInVram.Should().BeFalse();
        estimate.RecommendedGpuLayers.Should().BeInRange(1, layers - 1, "the case under test is a partial offload");

        var (safe, floored) = LlamaServerGeneratorModel.EstimateSafeContextLengthDetailed(
            fileBytes, context, estimate.RecommendedGpuLayers, metadata, gpu: gpu);
        floored.Should().BeFalse();
        safe.Should().Be(context, because: "the layers were chosen to leave room for exactly this context");
    }
}
