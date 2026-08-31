using AwesomeAssertions;
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
    public async Task GetProviderFallbackChain_OnWindows_ShouldIncludeDirectML()
    {
        // Skip on non-Windows
        if (!OperatingSystem.IsWindows())
        {
            return;
        }

        // Arrange
        var manager = new RuntimeManager();
        await manager.InitializeAsync(TestContext.Current.CancellationToken);

        // Act
        var chain = manager.GetProviderFallbackChain();

        // Assert - On Windows with D3D12 GPU, DirectML should be in the chain
        // Note: This may fail on machines without DirectML support (e.g., very old GPUs)
        if (manager.Gpu.DirectMLSupported)
        {
            chain.Should().Contain("directml", "DirectML should be available on Windows with D3D12 GPU");
        }
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

            // CUDA should come before DirectML in priority
            var cudaIndex = chain.ToList().FindIndex(p => p.StartsWith("cuda", StringComparison.Ordinal));
            var directmlIndex = chain.ToList().IndexOf("directml");

            if (directmlIndex >= 0)
            {
                cudaIndex.Should().BeLessThan(directmlIndex, "CUDA should have higher priority than DirectML");
            }
        }
    }

    [Fact]
    public async Task GetProviderFallbackChain_PriorityOrder_ShouldBeCudaDirectMLCoreMLCpu()
    {
        // Arrange
        var manager = new RuntimeManager();
        await manager.InitializeAsync(TestContext.Current.CancellationToken);

        // Act
        var chain = manager.GetProviderFallbackChain().ToList();

        // Assert - Verify order based on what's available
        var cudaIndex = chain.FindIndex(p => p.StartsWith("cuda", StringComparison.Ordinal));
        var directmlIndex = chain.IndexOf("directml");
        var coremlIndex = chain.IndexOf("coreml");
        var cpuIndex = chain.IndexOf("cpu");

        // CPU should always be last
        cpuIndex.Should().Be(chain.Count - 1, "CPU should always be the last provider");

        // If CUDA exists, it should come before DirectML
        if (cudaIndex >= 0 && directmlIndex >= 0)
        {
            cudaIndex.Should().BeLessThan(directmlIndex, "CUDA should come before DirectML");
        }

        // If DirectML exists, it should come before CoreML
        if (directmlIndex >= 0 && coremlIndex >= 0)
        {
            directmlIndex.Should().BeLessThan(coremlIndex, "DirectML should come before CoreML");
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
