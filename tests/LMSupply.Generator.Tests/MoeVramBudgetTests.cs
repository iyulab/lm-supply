using AwesomeAssertions;
using LMSupply.Generator.Internal.Llama;
using LMSupply.Hardware;
using Xunit;

namespace LMSupply.Generator.Tests;

/// <summary>
/// Tests for MoE-aware VRAM budget estimation in EstimateSafeContextLength.
/// Placed in "VramBudget" collection to run sequentially with VramAwareSelectionIntegrationTests,
/// preventing env-var race conditions on LMSUPPLY_VRAM_BUDGET_MB.
/// </summary>
[Collection("VramBudget")]
public class MoeVramBudgetTests : IDisposable
{
    private readonly string _tempFile;
    private string? _originalBudgetEnv;

    public MoeVramBudgetTests()
    {
        // 1 KB fake model file — small enough to not dominate the budget calculation.
        _tempFile = Path.GetTempFileName();
        File.WriteAllBytes(_tempFile, new byte[1024]);

        // Save original env var value so tests are isolated.
        _originalBudgetEnv = Environment.GetEnvironmentVariable(VramBudget.BudgetOverrideEnvVar);
    }

    public void Dispose()
    {
        if (File.Exists(_tempFile))
            File.Delete(_tempFile);
        // Restore original env var.
        if (_originalBudgetEnv is null)
            Environment.SetEnvironmentVariable(VramBudget.BudgetOverrideEnvVar, null);
        else
            Environment.SetEnvironmentVariable(VramBudget.BudgetOverrideEnvVar, _originalBudgetEnv);
        GC.SuppressFinalize(this);
    }

    private static void SetVramBudgetMb(int mb)
        => Environment.SetEnvironmentVariable(VramBudget.BudgetOverrideEnvVar, mb.ToString(System.Globalization.CultureInfo.InvariantCulture));

    // ── CPU path ────────────────────────────────────────────────────────────────

    [Fact]
    public void EstimateSafeContextLength_CpuOnly_AlwaysReturnsRequested()
    {
        SetVramBudgetMb(3000);
        var moeMetadata = new GgufMetadata { ExpertCount = 64, ExpertUsedCount = 4 };

        var result = LlamaServerGeneratorModel.EstimateSafeContextLength(
            _tempFile, requestedContext: 16384, gpuLayerCount: 0, ggufMetadata: moeMetadata);

        result.Should().Be(16384);
    }

    // ── Dense vs MoE capping ─────────────────────────────────────────────────

    [Fact]
    public void EstimateSafeContextLength_MoeModel_CapsLowerThanDense()
    {
        // Use a constrained budget so both paths actually cap below requestedContext.
        SetVramBudgetMb(3000);

        var denseCap = LlamaServerGeneratorModel.EstimateSafeContextLength(
            _tempFile, requestedContext: 16384, gpuLayerCount: -1, ggufMetadata: null);

        var moeMetadata = new GgufMetadata { ExpertCount = 64, ExpertUsedCount = 4 };
        var moeCap = LlamaServerGeneratorModel.EstimateSafeContextLength(
            _tempFile, requestedContext: 16384, gpuLayerCount: -1, ggufMetadata: moeMetadata);

        // MoE budget is 80% of dense — cap must be strictly lower.
        moeCap.Should().BeLessThan(denseCap,
            because: "MoE overhead margin (×0.80) must reduce the effective context cap");
        // Both must actually cap below the requested value.
        denseCap.Should().BeLessThan(16384);
        moeCap.Should().BeLessThan(16384);
    }

    [Fact]
    public void EstimateSafeContextLength_DenseModel_NoExpertCount_UseFullBudget()
    {
        SetVramBudgetMb(3000);
        var denseMetadata = new GgufMetadata { ExpertCount = null };

        var result = LlamaServerGeneratorModel.EstimateSafeContextLength(
            _tempFile, requestedContext: 16384, gpuLayerCount: -1, ggufMetadata: denseMetadata);

        var resultNoMetadata = LlamaServerGeneratorModel.EstimateSafeContextLength(
            _tempFile, requestedContext: 16384, gpuLayerCount: -1, ggufMetadata: null);

        // Null metadata and ExpertCount=null must produce identical results.
        result.Should().Be(resultNoMetadata);
    }

    [Fact]
    public void EstimateSafeContextLength_SingleExpert_NotMoe_FullBudget()
    {
        // ExpertCount = 1 is a dense model; margin must NOT apply.
        SetVramBudgetMb(3000);
        var notMoeMetadata = new GgufMetadata { ExpertCount = 1 };

        var result = LlamaServerGeneratorModel.EstimateSafeContextLength(
            _tempFile, requestedContext: 16384, gpuLayerCount: -1, ggufMetadata: notMoeMetadata);

        var resultNoMetadata = LlamaServerGeneratorModel.EstimateSafeContextLength(
            _tempFile, requestedContext: 16384, gpuLayerCount: -1, ggufMetadata: null);

        result.Should().Be(resultNoMetadata);
    }

    [Fact]
    public void EstimateSafeContextLength_RequestedFitsEvenWithMoeMargin_ReturnsRequested()
    {
        // Very large VRAM budget — even the MoE margin should not cause a cap.
        SetVramBudgetMb(200_000);
        var moeMetadata = new GgufMetadata { ExpertCount = 64 };

        var result = LlamaServerGeneratorModel.EstimateSafeContextLength(
            _tempFile, requestedContext: 4096, gpuLayerCount: -1, ggufMetadata: moeMetadata);

        result.Should().Be(4096);
    }

    // ── GgufMetadata ExpertCount properties ──────────────────────────────────

    [Fact]
    public void GgufMetadata_ExpertCount_DefaultsToNull()
    {
        var metadata = new GgufMetadata();

        metadata.ExpertCount.Should().BeNull();
        metadata.ExpertUsedCount.Should().BeNull();
    }

    [Fact]
    public void GgufMetadata_ExpertCount_IsInitialized()
    {
        var metadata = new GgufMetadata { ExpertCount = 64, ExpertUsedCount = 4 };

        metadata.ExpertCount.Should().Be(64);
        metadata.ExpertUsedCount.Should().Be(4);
    }

    [Fact]
    public void GgufMetadata_GetSummary_IncludesExpertInfo()
    {
        var metadata = new GgufMetadata
        {
            Architecture = "gemma4",
            ExpertCount = 64,
            ExpertUsedCount = 4,
        };

        var summary = metadata.GetSummary();

        summary.Should().Contain("Experts: 64");
        summary.Should().Contain("top-4");
    }

    [Fact]
    public void GgufMetadata_GetSummary_SkipsExpertInfoForDenseModels()
    {
        var metadata = new GgufMetadata
        {
            Architecture = "llama",
            ExpertCount = null,
        };

        var summary = metadata.GetSummary();

        summary.Should().NotContain("Experts");
    }
}
