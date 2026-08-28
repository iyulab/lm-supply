using AwesomeAssertions;
using LMSupply.Exceptions;
using LMSupply.Inference;
using LMSupply.Runtime;

namespace LMSupply.Core.Tests.Inference;

/// <summary>
/// Tests for OnnxSessionFactory fallback chain behavior.
/// These tests verify that Auto mode correctly configures the GPU provider fallback chain.
/// Note: Tests that require native ONNX Runtime binaries are skipped in unit test environments.
/// </summary>
public class OnnxSessionFactoryTests
{
    [Fact]
    public async Task Auto_ShouldUseFallbackChainFromGpuInfo()
    {
        // Arrange
        await RuntimeManager.Instance.InitializeAsync();
        var gpu = RuntimeManager.Instance.Gpu;

        // Act
        var fallbackChain = gpu?.GetFallbackProviders();

        // Assert
        fallbackChain.Should().NotBeNull();
        fallbackChain.Should().Contain(ExecutionProvider.Cpu, "CPU should always be in fallback chain");
    }

    [Fact]
    public async Task GetFallbackProviders_OnNvidiaGpu_ShouldHaveCudaFirst()
    {
        // Arrange
        await RuntimeManager.Instance.InitializeAsync();
        var gpu = RuntimeManager.Instance.Gpu;

        // Skip if not NVIDIA GPU
        if (gpu?.Vendor != GpuVendor.Nvidia || gpu.CudaDriverVersionMajor < 11)
        {
            return; // Skip test - no NVIDIA GPU available
        }

        // Act
        var fallbackChain = gpu.GetFallbackProviders();

        // Assert
        fallbackChain.Should().NotBeNull();
        fallbackChain[0].Should().Be(ExecutionProvider.Cuda, "CUDA should be first for NVIDIA GPUs");
        fallbackChain[^1].Should().Be(ExecutionProvider.Cpu, "CPU should be last");
    }

    [Fact]
    public async Task GetFallbackProviders_OnWindows_ShouldIncludeDirectML()
    {
        // Arrange
        await RuntimeManager.Instance.InitializeAsync();
        var gpu = RuntimeManager.Instance.Gpu;

        // Skip if not Windows or no DirectML support
        if (!OperatingSystem.IsWindows() || gpu?.DirectMLSupported != true)
        {
            return; // Skip test
        }

        // Act
        var fallbackChain = gpu.GetFallbackProviders();

        // Assert
        fallbackChain.Should().Contain(ExecutionProvider.DirectML, "DirectML should be in fallback chain on Windows");
    }

    [Fact]
    public async Task GetFallbackProviders_CpuShouldAlwaysBeLast()
    {
        // Arrange
        await RuntimeManager.Instance.InitializeAsync();
        var gpu = RuntimeManager.Instance.Gpu;

        // Act
        var fallbackChain = gpu?.GetFallbackProviders() ?? new[] { ExecutionProvider.Cpu };

        // Assert
        fallbackChain[^1].Should().Be(ExecutionProvider.Cpu, "CPU should always be the final fallback");
    }

    [Fact]
    public async Task RuntimeManagerChain_ShouldMatchGpuInfoChain()
    {
        // Arrange
        await RuntimeManager.Instance.InitializeAsync();
        var gpu = RuntimeManager.Instance.Gpu;

        // Act
        var runtimeChain = RuntimeManager.Instance.GetProviderFallbackChain();
        var gpuChain = gpu?.GetFallbackProviders() ?? new[] { ExecutionProvider.Cpu };

        // Assert: Both chains should have the same providers (though different string/enum types)
        runtimeChain.Should().NotBeNull();
        runtimeChain.Should().Contain("cpu", "CPU should always be in chain");

        // If GPU has CUDA, RuntimeManager should have cuda11/cuda12
        if (gpuChain.Contains(ExecutionProvider.Cuda))
        {
            runtimeChain.Should().Match(chain =>
                chain.Contains("cuda11") || chain.Contains("cuda12"),
                "CUDA should be in RuntimeManager chain when GPU supports it");
        }

        // If GPU has DirectML, RuntimeManager should have directml
        if (gpuChain.Contains(ExecutionProvider.DirectML))
        {
            runtimeChain.Should().Contain("directml", "DirectML should be in RuntimeManager chain when GPU supports it");
        }
    }

    [Fact]
    public async Task Auto_OnNvidiaWithDirectML_ShouldHaveBothInChain()
    {
        // Arrange
        await RuntimeManager.Instance.InitializeAsync();
        var gpu = RuntimeManager.Instance.Gpu;

        // Skip if not NVIDIA on Windows
        if (gpu?.Vendor != GpuVendor.Nvidia || !gpu.DirectMLSupported)
        {
            return; // Skip test
        }

        // Act
        var fallbackChain = gpu.GetFallbackProviders().ToList();

        // Assert: NVIDIA Windows should have both CUDA and DirectML
        fallbackChain.Should().HaveCountGreaterThanOrEqualTo(3,
            "NVIDIA on Windows should have at least CUDA, DirectML, and CPU");

        // CUDA should come before DirectML
        var cudaIndex = fallbackChain.IndexOf(ExecutionProvider.Cuda);
        var directMLIndex = fallbackChain.IndexOf(ExecutionProvider.DirectML);
        cudaIndex.Should().BeLessThan(directMLIndex,
            "CUDA should be tried before DirectML on NVIDIA GPUs");
    }

    [Fact]
    public async Task FallbackChain_ShouldBeOrdered_GpuBeforeCpu()
    {
        // Arrange
        await RuntimeManager.Instance.InitializeAsync();
        var gpu = RuntimeManager.Instance.Gpu;

        if (gpu is null)
        {
            return; // Skip - no GPU detected
        }

        // Act
        var fallbackChain = gpu.GetFallbackProviders().ToList();
        var cpuIndex = fallbackChain.IndexOf(ExecutionProvider.Cpu);

        // Assert
        cpuIndex.Should().Be(fallbackChain.Count - 1, "CPU should be the last provider in the chain");

        // All GPU providers should come before CPU
        for (int i = 0; i < cpuIndex; i++)
        {
            fallbackChain[i].Should().NotBe(ExecutionProvider.Cpu, $"Position {i} should be a GPU provider");
        }
    }

    [Fact]
    public void CheckOnnxRuntimeAvailability_ShouldReturnWithoutCrash()
    {
        // The key invariant: this method should NEVER crash the process.
        // On dev machines with VC++ Redistributable, returns true.
        // On fresh machines without it, returns false with a helpful message.
        var (available, errorMessage) = OnnxSessionFactory.CheckOnnxRuntimeAvailability();

        if (available)
        {
            errorMessage.Should().BeNull();
        }
        else
        {
            errorMessage.Should().NotBeNullOrEmpty("should provide actionable guidance when unavailable");
        }
    }

    [Fact]
    public void CheckOnnxRuntimeAvailability_ShouldNotCrashProcess_EvenWhenUnavailable()
    {
        // The method should always return a tuple without crashing,
        // regardless of native library availability.
        var action = () => OnnxSessionFactory.CheckOnnxRuntimeAvailability();
        action.Should().NotThrow("pre-check must never crash the process");
    }

    [Fact]
    public async Task CreateWithInfoAsync_WithSkipProviders_ShouldAcceptOverload()
    {
        // The skipProviders overload exists so OnnxInferenceEngine can request a
        // re-creation that excludes a provider that crashed at inference time.
        // Verify the overload is callable with a non-empty skip list and reaches
        // its precheck (which fails on a missing model file as expected).
        var skip = new[] { ExecutionProvider.DirectML };

        Func<Task> action = async () => await OnnxSessionFactory.CreateWithInfoAsync(
            modelPath: "/nonexistent/model.onnx",
            provider: ExecutionProvider.Auto,
            skipProviders: skip);

        // Should fail at the precheck or session creation, not at signature resolution.
        var ex = await action.Should().ThrowAsync<Exception>();

        // Regression guard: the availability precheck must run AFTER RuntimeManager has had a
        // chance to provision the runtime (see CreateWithFallbackChainAsync's <remarks>), not
        // before. A "native library failed to load" message here would mean the precheck ran
        // first-thing again and rejected before ever attempting provisioning.
        ex.Which.Message.Should().NotContain("native library failed to load",
            "the fallback chain must provision the runtime before checking availability, not the reverse");
    }

    [Fact]
    public async Task CreateWithInfoAsync_WithNullSkipProviders_ShouldBeEquivalentToOriginalOverload()
    {
        // Null skipProviders must behave exactly like the original overload — no provider exclusion.
        Func<Task> action = async () => await OnnxSessionFactory.CreateWithInfoAsync(
            modelPath: "/nonexistent/model.onnx",
            provider: ExecutionProvider.Auto,
            skipProviders: null);

        await action.Should().ThrowAsync<Exception>();
    }

    [SkippableFact]
    public void ConfigureExecutionProvider_Cpu_ReturnsFalse_NoGpuEpAppended()
    {
        // D6: ConfigureExecutionProvider now reports whether a GPU EP was appended, so callers can
        // build accurate ActiveProviders instead of a loadability heuristic. CPU appends no GPU EP.
        var (available, _) = OnnxSessionFactory.CheckOnnxRuntimeAvailability();
        Skip.IfNot(available, "ONNX Runtime not available in this environment");

        using var options = new Microsoft.ML.OnnxRuntime.SessionOptions();
        OnnxSessionFactory.ConfigureExecutionProvider(options, ExecutionProvider.Cpu)
            .Should().BeFalse("CPU configuration appends no GPU execution provider");
    }

    [Fact]
    public void WrapProvisioningFailure_ProducesActionableMessage_NamingProviderAndPlatform()
    {
        // Regression test for the provisioning-vs-session-construction distinction (see
        // ISSUE-lm-supply-20260818-explicit-gpu-provisioning-failure-bypasses-cpu-fallback.md).
        // A real end-to-end repro requires a platform where the requested provider genuinely has
        // no native binaries (e.g. DirectML on linux-x64) -- exercising that for real would mean
        // downloading a multi-hundred-MB NuGet package on every CI run just to observe it lacks
        // the current RID's entries. Testing the pure wrapping function directly (no I/O) is the
        // "test double" alternative named in the issue's acceptance criteria.
        var inner = new InvalidOperationException(
            "No native binaries found for linux-x64 in Microsoft.ML.OnnxRuntime.DirectML");

        var wrapped = OnnxSessionFactory.WrapProvisioningFailure(
            ExecutionProvider.DirectML, "model.onnx", inner);

        wrapped.Message.Should().Contain("DirectML",
            "the message must name the provider the caller explicitly requested");
        wrapped.Message.Should().NotContain("No native binaries found",
            "the provisioning layer's raw message must not leak to the caller verbatim");
        wrapped.InnerException.Should().BeSameAs(inner,
            "the original provisioning exception must be preserved for diagnostics");
        wrapped.ModelId.Should().Be("model.onnx");
    }

    [SkippableFact]
    public async Task CreateWithInfoAsync_ExplicitGpuProvider_WhenSessionCreateThrows_FallsBackToCpu()
    {
        // This test simulates a *session-construction*-time GPU failure (e.g. a DX12 device
        // unavailable at runtime) on top of a GPU runtime that DOES provision successfully — it is
        // not exercising (and cannot exercise, by construction) a *provisioning*-time failure, where
        // the requested provider has no native binaries for this platform at all. DirectML only ever
        // provisions on Windows, so gate on that specifically — not on
        // CheckOnnxRuntimeAvailability(), which only reports whether *some* provider (e.g. CPU) is
        // already loaded and says nothing about DirectML. On Linux/macOS CI, the generic check can be
        // true (once anything has provisioned) while DirectML provisioning still always fails with
        // "no native binaries for this platform" — that case is now covered separately: it throws
        // ModelLoadException via WrapProvisioningFailure (see
        // WrapProvisioningFailure_ProducesActionableMessage_NamingProviderAndPlatform above) rather
        // than falling back to CPU, since a caller who explicitly asked for DirectML should learn
        // the platform can't provide it instead of silently running on CPU.
        await RuntimeManager.Instance.InitializeAsync();
        var gpu = RuntimeManager.Instance.Gpu;
        Skip.IfNot(OperatingSystem.IsWindows() && gpu?.DirectMLSupported == true,
            "DirectML is only ever provisionable on Windows with DirectML support");

        // Simulate a session-creation failure on the first attempt (DML) by injecting a
        // configureOptions callback that throws. The factory should catch this and retry
        // with CPU. The CPU attempt then fails on the nonexistent model file, which is the
        // expected final exception (not the simulated DML error).
        var attemptCount = 0;
        Action<Microsoft.ML.OnnxRuntime.SessionOptions> failOnFirstAttempt = _ =>
        {
            if (Interlocked.Increment(ref attemptCount) == 1)
                throw new InvalidOperationException("Simulated DirectML init failure");
        };

        Func<Task> action = async () => await OnnxSessionFactory.CreateWithInfoAsync(
            "nonexistent_model_for_fallback_test.onnx",
            ExecutionProvider.DirectML,
            failOnFirstAttempt);

        // Before the fix: InvalidOperationException propagates directly (no CPU fallback).
        // After the fix: InvalidOperationException is caught; CPU fallback is attempted;
        //   the final exception is from the CPU session attempt (model file not found).
        var ex = await Assert.ThrowsAnyAsync<Exception>(action);
        ex.Should().NotBeOfType<InvalidOperationException>(
            "the simulated GPU failure should be caught and CPU fallback should be attempted");
        attemptCount.Should().Be(2, "configureOptions should have been called twice: once for GPU, once for CPU fallback");
    }
}
