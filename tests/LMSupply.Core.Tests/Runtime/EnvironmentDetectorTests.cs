using System.Runtime.InteropServices;
using AwesomeAssertions;
using LMSupply.Runtime;

namespace LMSupply.Core.Tests.Runtime;

public class EnvironmentDetectorTests
{
    [Fact]
    public void DetectPlatform_ShouldReturnValidPlatformInfo()
    {
        // Act
        var platform = EnvironmentDetector.DetectPlatform();

        // Assert
        platform.Should().NotBeNull();
        platform.OS.Should().NotBe(default(OSPlatform));
        platform.Architecture.Should().BeDefined();
        platform.RuntimeIdentifier.Should().NotBeNullOrEmpty();
    }

    [Fact]
    public void DetectPlatform_ShouldBeCached()
    {
        // Act
        var platform1 = EnvironmentDetector.DetectPlatform();
        var platform2 = EnvironmentDetector.DetectPlatform();

        // Assert
        platform1.Should().BeSameAs(platform2);
    }

    [Fact]
    public void DetectGpu_ShouldReturnValidGpuInfo()
    {
        // Act
        var gpu = EnvironmentDetector.DetectGpu();

        // Assert
        gpu.Should().NotBeNull();
        gpu.Vendor.Should().BeDefined();
    }

    [Fact]
    public void DetectGpu_ShouldBeCached()
    {
        // Act
        var gpu1 = EnvironmentDetector.DetectGpu();
        var gpu2 = EnvironmentDetector.DetectGpu();

        // Assert
        gpu1.Should().BeSameAs(gpu2);
    }

    [Fact]
    public void GetAvailableProviders_ShouldAlwaysIncludeCpu()
    {
        // Act
        var providers = EnvironmentDetector.GetAvailableProviders();

        // Assert
        providers.Should().Contain(ExecutionProvider.Cpu);
    }

    [Fact]
    public void GetAvailableProviders_ShouldReturnDistinctProviders()
    {
        // Act
        var providers = EnvironmentDetector.GetAvailableProviders().ToList();

        // Assert
        providers.Should().OnlyHaveUniqueItems();
    }

    [Fact]
    public void DetectAllGpus_ShouldReturnNonEmptyList()
    {
        var gpus = EnvironmentDetector.DetectAllGpus();

        gpus.Should().NotBeNull();
        gpus.Should().HaveCountGreaterThanOrEqualTo(1, "at least one GPU or fallback entry should exist");
    }

    [Fact]
    public void GetRecommendedProvider_ShouldReturnValidProvider()
    {
        var provider = EnvironmentDetector.GetRecommendedProvider();

        provider.Should().BeDefined();
    }

    [Fact]
    public void GetEnvironmentSummary_ShouldReturnNonEmptyString()
    {
        var summary = EnvironmentDetector.GetEnvironmentSummary();

        summary.Should().NotBeNullOrWhiteSpace();
        summary.Should().Contain("Platform:");
        summary.Should().Contain("GPU:");
        summary.Should().Contain("Recommended Provider:");
        summary.Should().Contain("Available Providers:");
    }

    [Fact]
    public void ClearCache_ShouldAllowRedetection()
    {
        // First detection
        var gpu1 = EnvironmentDetector.DetectGpu();

        // Clear and re-detect
        EnvironmentDetector.ClearCache();
        var gpu2 = EnvironmentDetector.DetectGpu();

        // Both should be valid (may or may not be same instance after clearing)
        gpu1.Should().NotBeNull();
        gpu2.Should().NotBeNull();
        gpu2.Vendor.Should().Be(gpu1.Vendor);
    }
}
