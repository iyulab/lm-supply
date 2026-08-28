using AwesomeAssertions;
using LMSupply.Generator.Internal.Llama;
using LMSupply.Hardware;
using Xunit;

namespace LMSupply.Generator.Tests;

/// <summary>
/// Tests for EstimateSafeContextLengthDetailed — the floored-by-VRAM signal that backs
/// GeneratorModelInfo.ContextFlooredByVram. Floored == the raw VRAM-derived context estimate
/// fell below the 512-token usable floor (VRAM insufficient for a usable context), distinct
/// from a legitimately small request. Uses LMSUPPLY_VRAM_BUDGET_MB override (no special HW),
/// so it shares the "VramBudget" collection with the other env-var-mutating suites.
/// </summary>
[Collection("VramBudget")]
public class EstimateSafeContextLengthDetailedTests : IDisposable
{
    private readonly string _tempFile;
    private readonly string? _originalBudgetEnv;

    public EstimateSafeContextLengthDetailedTests()
    {
        // 1 KB fake model file — small enough to not dominate the budget calculation.
        _tempFile = Path.GetTempFileName();
        File.WriteAllBytes(_tempFile, new byte[1024]);
        _originalBudgetEnv = Environment.GetEnvironmentVariable(VramBudget.BudgetOverrideEnvVar);
    }

    public void Dispose()
    {
        if (File.Exists(_tempFile))
            File.Delete(_tempFile);
        Environment.SetEnvironmentVariable(VramBudget.BudgetOverrideEnvVar, _originalBudgetEnv);
        GC.SuppressFinalize(this);
    }

    private static void SetVramBudgetMb(int mb)
        => Environment.SetEnvironmentVariable(
            VramBudget.BudgetOverrideEnvVar, mb.ToString(System.Globalization.CultureInfo.InvariantCulture));

    [Fact]
    public void TinyBudget_GpuPath_FlooredTrue_AndContextIsFloor()
    {
        // 512MB budget == the safety buffer, so remainingVram collapses to 0 -> raw estimate 0 < 512.
        SetVramBudgetMb(512);

        var (context, floored) = LlamaServerGeneratorModel.EstimateSafeContextLengthDetailed(
            _tempFile, requestedContext: 4096, gpuLayerCount: -1, ggufMetadata: null);

        floored.Should().BeTrue("the VRAM-derived estimate was raised from below 512 to the floor");
        context.Should().Be(LlamaServerGeneratorModel.UnusableContextFloorTokens);
    }

    [Fact]
    public void AmpleBudget_GpuPath_FlooredFalse()
    {
        // Generous budget — the estimate stays well above the 512 floor.
        SetVramBudgetMb(3000);

        var (context, floored) = LlamaServerGeneratorModel.EstimateSafeContextLengthDetailed(
            _tempFile, requestedContext: 4096, gpuLayerCount: -1, ggufMetadata: null);

        floored.Should().BeFalse("the VRAM-derived estimate stayed at or above the 512 floor");
        context.Should().BeGreaterThanOrEqualTo(LlamaServerGeneratorModel.UnusableContextFloorTokens);
    }

    [Fact]
    public void CpuPath_NeverFloored()
    {
        SetVramBudgetMb(512); // would floor a GPU path, but gpuLayerCount==0 short-circuits
        var (context, floored) = LlamaServerGeneratorModel.EstimateSafeContextLengthDetailed(
            _tempFile, requestedContext: 4096, gpuLayerCount: 0, ggufMetadata: null);

        floored.Should().BeFalse("CPU is RAM-bound — no VRAM clamp applies");
        context.Should().Be(4096, "CPU path returns the requested context unchanged");
    }

    [Fact]
    public void PublicEstimate_DelegatesToDetailedContext()
    {
        SetVramBudgetMb(3000);

        var detailedContext = LlamaServerGeneratorModel.EstimateSafeContextLengthDetailed(
            _tempFile, requestedContext: 16384, gpuLayerCount: -1, ggufMetadata: null).Context;
        var publicContext = LlamaServerGeneratorModel.EstimateSafeContextLength(
            _tempFile, requestedContext: 16384, gpuLayerCount: -1, ggufMetadata: null);

        publicContext.Should().Be(detailedContext, "the public estimate must delegate to the detailed Context");
    }
}
