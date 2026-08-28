using AwesomeAssertions;
using LMSupply;
using LMSupply.Generator.Internal.Llama;
using LMSupply.Llama.Server;

namespace LMSupply.Generator.Tests.Internal.Llama;

/// <summary>
/// Tests for DecideFlooredContextAction — the pure decision that recovers from an
/// unusable GPU context clamp (n_ctx floored to 512). Under ExecutionProvider.Auto the
/// loader must transparently fall back to CPU (RAM-bound, no VRAM clamp); under an explicit
/// GPU pin it must fail fast instead of silently loading an unusable 512-token context.
/// HW-free: drives the floor condition by passing a low safeContext directly.
/// </summary>
public class LlamaServerGeneratorModelFlooredContextTests
{
    private const int Floor = 512;     // LlamaServerGeneratorModel.UnusableContextFloorTokens
    private const int Requested = 4096; // typical default request

    // ─── Auto: floored GPU ctx → transparent CPU fallback ───

    [Fact]
    public void Auto_GpuBackend_FlooredCtx_FallsBackToCpu()
    {
        var action = LlamaServerGeneratorModel.DecideFlooredContextAction(
            ExecutionProvider.Auto, LlamaServerBackend.Vulkan, safeContext: Floor, requestedContext: Requested);

        action.Should().Be(LlamaServerGeneratorModel.FlooredContextAction.FallBackToCpu);
    }

    // ─── Explicit GPU pin: floored ctx → fail fast (no silent CPU swap) ───

    [Theory]
    [InlineData(ExecutionProvider.DirectML)]
    [InlineData(ExecutionProvider.Cuda)]
    [InlineData(ExecutionProvider.CoreML)]
    public void ExplicitGpuPin_FlooredCtx_FailsFast(ExecutionProvider pinned)
    {
        var action = LlamaServerGeneratorModel.DecideFlooredContextAction(
            pinned, LlamaServerBackend.Vulkan, safeContext: Floor, requestedContext: Requested);

        action.Should().Be(LlamaServerGeneratorModel.FlooredContextAction.FailFast);
    }

    // ─── Not floored: usable GPU ctx → proceed ───

    [Fact]
    public void Auto_GpuBackend_UsableCtx_Proceeds()
    {
        // safeContext well above the floor and equal to requested → no clamp, no recovery.
        var action = LlamaServerGeneratorModel.DecideFlooredContextAction(
            ExecutionProvider.Auto, LlamaServerBackend.Vulkan, safeContext: Requested, requestedContext: Requested);

        action.Should().Be(LlamaServerGeneratorModel.FlooredContextAction.Proceed);
    }

    [Fact]
    public void Auto_GpuBackend_RequestedAtFloor_Proceeds()
    {
        // Caller asked for a tiny context (512); 512 fits — that is NOT a floored brick.
        var action = LlamaServerGeneratorModel.DecideFlooredContextAction(
            ExecutionProvider.Auto, LlamaServerBackend.Vulkan, safeContext: Floor, requestedContext: Floor);

        action.Should().Be(LlamaServerGeneratorModel.FlooredContextAction.Proceed);
    }

    // ─── CPU backend / CPU provider: never a VRAM floor to recover from ───

    [Fact]
    public void CpuBackend_NeverFloored_Proceeds()
    {
        var action = LlamaServerGeneratorModel.DecideFlooredContextAction(
            ExecutionProvider.Auto, LlamaServerBackend.Cpu, safeContext: Floor, requestedContext: Requested);

        action.Should().Be(LlamaServerGeneratorModel.FlooredContextAction.Proceed);
    }

    [Fact]
    public void ExplicitCpuProvider_FlooredValueButCpuBackend_Proceeds()
    {
        var action = LlamaServerGeneratorModel.DecideFlooredContextAction(
            ExecutionProvider.Cpu, LlamaServerBackend.Cpu, safeContext: Floor, requestedContext: Requested);

        action.Should().Be(LlamaServerGeneratorModel.FlooredContextAction.Proceed);
    }
}
