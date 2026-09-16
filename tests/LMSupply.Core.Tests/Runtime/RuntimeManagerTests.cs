using AwesomeAssertions;
using LMSupply.Exceptions;
using LMSupply.Runtime;

namespace LMSupply.Core.Tests.Runtime;

/// <summary>
/// Tests for RuntimeManager GPU fallback chain functionality.
/// </summary>
[Trait("Category", "Unit")]
public class RuntimeManagerTests
{
    [Fact]
    public async Task GetProviderFallbackChain_AlwaysIncludesCpuAsFinalFallback()
    {
        // Arrange
        var manager = new RuntimeManager();
        await manager.InitializeAsync(TestContext.Current.CancellationToken);

        // Act
        var chain = manager.GetProviderFallbackChain();

        // Assert
        chain.Should().NotBeEmpty();
        chain[^1].Should().Be("cpu", "CPU should always be the final fallback");
    }

    [Fact]
    public async Task GetProviderFallbackChain_ReturnsAtLeastOneProvider()
    {
        // Arrange
        var manager = new RuntimeManager();
        await manager.InitializeAsync(TestContext.Current.CancellationToken);

        // Act
        var chain = manager.GetProviderFallbackChain();

        // Assert
        chain.Should().HaveCountGreaterThanOrEqualTo(1, "At least CPU should be available");
    }

    [Fact]
    public async Task GetProviderFallbackChain_NeverIncludesDirectML()
    {
        // 0.67.0: no build can provision the DirectML provider (Microsoft.ML.OnnxRuntime.DirectML ends at
        // 1.24.4; the runtime version comes from the loaded 1.30.0 assembly), so the chain must not name it
        // even on a Direct3D 12 capable Windows GPU — that used to cost a 404 per Auto session.
        var manager = new RuntimeManager();
        await manager.InitializeAsync(TestContext.Current.CancellationToken);

        var chain = manager.GetProviderFallbackChain();

        chain.Should().NotContain("directml");
        manager.GetProviderFallbackChain(RuntimePackageRegistry.PackageTypes.OnnxRuntimeGenAI).Should().NotContain("directml");
    }

    [Fact]
    public async Task EnsureRuntimeAsync_DirectML_ThrowsNotSupported_InsteadOfLandingOnCpu()
    {
        // The registry falls back to the CPU package for a provider it does not know, so without an
        // explicit refusal "directml" would silently provision CPU binaries. The refusal is thrown before
        // any package lookup or network access.
        var manager = new RuntimeManager();
        await manager.InitializeAsync(TestContext.Current.CancellationToken);

        var act = async () => await manager.EnsureRuntimeAsync(
            "onnxruntime", provider: "directml", cancellationToken: TestContext.Current.CancellationToken);

        (await act.Should().ThrowAsync<NotSupportedException>())
            .WithMessage("*DirectML*1.24.4*");
        manager.ActiveProvider.Should().BeNull("nothing was provisioned");
    }

    [Fact]
    public async Task GetProviderFallbackChain_OnNvidiaGpu_ShouldIncludeCuda()
    {
        // Arrange
        var manager = new RuntimeManager();
        await manager.InitializeAsync(TestContext.Current.CancellationToken);

        // Act
        var chain = manager.GetProviderFallbackChain();

        // Assert - Only check if NVIDIA GPU is detected
        if (manager.Gpu.Vendor == GpuVendor.Nvidia && manager.Gpu.CudaDriverVersionMajor >= 11)
        {
            chain.Should().Contain(p => p.StartsWith("cuda", StringComparison.Ordinal), "CUDA should be available on NVIDIA GPU");
        }
    }

    [Fact]
    public async Task GetProviderFallbackChain_PriorityOrder_ShouldBeCudaCoreMLCpu()
    {
        // Arrange
        var manager = new RuntimeManager();
        await manager.InitializeAsync(TestContext.Current.CancellationToken);

        // Act
        var chain = manager.GetProviderFallbackChain().ToList();

        // Assert - Verify order based on what's available
        var cudaIndex = chain.FindIndex(p => p.StartsWith("cuda", StringComparison.Ordinal));
        var coremlIndex = chain.IndexOf("coreml");
        var cpuIndex = chain.IndexOf("cpu");

        // CPU should always be last
        cpuIndex.Should().Be(chain.Count - 1, "CPU should always be the last provider");

        // If CUDA exists, it should come before CoreML
        if (cudaIndex >= 0 && coremlIndex >= 0)
        {
            cudaIndex.Should().BeLessThan(coremlIndex, "CUDA should come before CoreML");
        }

        // If CoreML exists, it should come before CPU
        if (coremlIndex >= 0)
        {
            coremlIndex.Should().BeLessThan(cpuIndex, "CoreML should come before CPU");
        }
    }

    [Fact]
    public async Task GetDefaultProvider_ShouldReturnBestAvailableProvider()
    {
        // Arrange
        var manager = new RuntimeManager();
        await manager.InitializeAsync(TestContext.Current.CancellationToken);

        // Act
        var defaultProvider = manager.GetDefaultProvider();

        // Assert
        defaultProvider.Should().NotBeNullOrEmpty();

        // The default provider should be the first in the fallback chain
        var chain = manager.GetProviderFallbackChain();
        defaultProvider.Should().Be(chain[0], "Default provider should match first in fallback chain");
    }

    [Fact]
    public async Task GetEnvironmentSummary_AfterInitialize_ShouldIncludeProviderInfo()
    {
        // Arrange
        var manager = new RuntimeManager();
        await manager.InitializeAsync(TestContext.Current.CancellationToken);

        // Act
        var summary = manager.GetEnvironmentSummary();

        // Assert
        summary.Should().Contain("Platform:");
        summary.Should().Contain("GPU:");
        summary.Should().Contain("Recommended Provider:");
        summary.Should().Contain("Default Provider String:");
        summary.Should().Contain("Actually Loaded Runtime Path:");
    }

    [Fact]
    public async Task ActuallyLoadedRuntimePath_BeforeEnsureRuntimeCalled_ReturnsNull()
    {
        // ActuallyLoadedRuntimePath (docket iyulab/lm-supply#151) surfaces what NativeLoader
        // actually has resident, distinct from ActiveProvider/CurrentVersion (what was last
        // requested). Before any EnsureRuntimeAsync call this manager has requested nothing,
        // so there is nothing to report yet.
        var manager = new RuntimeManager();
        await manager.InitializeAsync(TestContext.Current.CancellationToken);

        manager.ActuallyLoadedRuntimePath.Should().BeNull();
    }

    [Fact]
    public void RuntimeManagerOptions_FailOnRuntimeConflict_DefaultsFalse()
    {
        // HD-45 (Option A): strict conflict detection must be opt-in so existing consumers'
        // current (lenient) behavior does not change unless they explicitly ask for it.
        var options = new RuntimeManagerOptions();
        options.FailOnRuntimeConflict.Should().BeFalse();
    }

    [Fact]
    public void StopsProviderFallbackChain_NativeLibraryConflictException_ReturnsTrue()
    {
        // cycle-389: without this, EnsureRuntimeAsync's Auto-mode fallback loop would catch a
        // NativeLibraryConflictException like any other per-provider failure and silently try
        // the next provider -- defeating FailOnRuntimeConflict for the common zero-config Auto
        // mode path, since every provider in the chain conflicts identically (same native
        // library name) and the exception a caller opted in to see would either recur pointlessly
        // or, worse, be swallowed entirely if a later provider's attempt happened not to conflict.
        var ex = new NativeLibraryConflictException("onnxruntime", "requested/path", "resident/path");

        RuntimeManager.StopsProviderFallbackChain(ex).Should().BeTrue();
    }

    [Fact]
    public void StopsProviderFallbackChain_OperationCanceledException_ReturnsTrue()
    {
        RuntimeManager.StopsProviderFallbackChain(new OperationCanceledException()).Should().BeTrue();
    }

    [Fact]
    public void StopsProviderFallbackChain_OrdinaryFailure_ReturnsFalse()
    {
        // An ordinary per-provider failure (e.g. this provider's package isn't available) must
        // still fall through to the next provider in the chain -- only cancellation and a native
        // library conflict are chain-terminating.
        RuntimeManager.StopsProviderFallbackChain(new InvalidOperationException("no config")).Should().BeFalse();
    }

    [Fact]
    public async Task GpuInfo_GetFallbackProviders_ShouldMatchRuntimeManagerChain()
    {
        // Arrange
        var manager = new RuntimeManager();
        await manager.InitializeAsync(TestContext.Current.CancellationToken);

        // Act
        var gpuProviders = manager.Gpu.GetFallbackProviders();
        var managerChain = manager.GetProviderFallbackChain();

        // Assert - Both should end with CPU
        gpuProviders[^1].Should().Be(ExecutionProvider.Cpu);
        managerChain[^1].Should().Be("cpu");

        // The number of providers should be similar (manager may have cuda11/cuda12 variants)
        gpuProviders.Count.Should().BeGreaterThanOrEqualTo(1);
        managerChain.Count.Should().BeGreaterThanOrEqualTo(1);
    }
}
