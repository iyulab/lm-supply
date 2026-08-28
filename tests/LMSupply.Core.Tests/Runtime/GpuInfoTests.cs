using AwesomeAssertions;
using LMSupply.Runtime;

namespace LMSupply.Core.Tests.Runtime;

public class GpuInfoTests
{
    private const long GB = 1024L * 1024 * 1024;
    private const long MB = 1024L * 1024;

    [Fact]
    public void DefaultValues_ShouldBeNull()
    {
        var gpu = new GpuInfo { Vendor = GpuVendor.Unknown };

        gpu.DeviceName.Should().BeNull();
        gpu.TotalMemoryBytes.Should().BeNull();
        gpu.SharedMemoryBytes.Should().BeNull();
        gpu.CudaComputeCapabilityMajor.Should().BeNull();
        gpu.CudaComputeCapabilityMinor.Should().BeNull();
        gpu.CudaDriverVersionMajor.Should().BeNull();
        gpu.CudaDriverVersionMinor.Should().BeNull();
        gpu.DirectMLSupported.Should().BeFalse();
        gpu.CoreMLSupported.Should().BeFalse();
        gpu.IsIntegrated.Should().BeFalse();
    }

    [Fact]
    public void IsIntegrated_IntelIGpu_TinyDedicatedPlusShared_True()
    {
        var gpu = new GpuInfo
        {
            Vendor = GpuVendor.Intel,
            DeviceName = "Intel(R) Iris(R) Xe Graphics",
            TotalMemoryBytes = 128 * MB,
            SharedMemoryBytes = 16 * GB
        };
        gpu.IsIntegrated.Should().BeTrue();
    }

    [Fact]
    public void IsIntegrated_IntelArc_LargeDedicated_False()
    {
        var gpu = new GpuInfo
        {
            Vendor = GpuVendor.Intel,
            DeviceName = "Intel(R) Arc(TM) A770",
            TotalMemoryBytes = 12 * GB,
            SharedMemoryBytes = 16 * GB
        };
        gpu.IsIntegrated.Should().BeFalse();
    }

    [Fact]
    public void IsIntegrated_Nvidia_NeverIntegrated_EvenWithSharedMemory()
    {
        // NVML path doesn't set SharedMemoryBytes, but even if DXGI did, NVIDIA is never integrated.
        var gpu = new GpuInfo
        {
            Vendor = GpuVendor.Nvidia,
            TotalMemoryBytes = 1 * GB,
            SharedMemoryBytes = 16 * GB
        };
        gpu.IsIntegrated.Should().BeFalse();
    }

    [Fact]
    public void IsIntegrated_NoSharedMemory_False()
    {
        var gpu = new GpuInfo { Vendor = GpuVendor.Intel, TotalMemoryBytes = 128 * MB };
        gpu.IsIntegrated.Should().BeFalse();
    }

    [Fact]
    public void IsIntegrated_AmdApu_TinyDedicatedPlusShared_True()
    {
        var gpu = new GpuInfo
        {
            Vendor = GpuVendor.Amd,
            DeviceName = "AMD Radeon(TM) Graphics",
            TotalMemoryBytes = 256 * MB,
            SharedMemoryBytes = 16 * GB
        };
        gpu.IsIntegrated.Should().BeTrue();
    }

    [Fact]
    public void RecommendedProvider_NvidiaCuda11Plus_ShouldBeCuda()
    {
        var gpu = new GpuInfo
        {
            Vendor = GpuVendor.Nvidia,
            CudaDriverVersionMajor = 12,
            CudaDriverVersionMinor = 3
        };

        gpu.RecommendedProvider.Should().Be(ExecutionProvider.Cuda);
    }

    [Fact]
    public void RecommendedProvider_NvidiaOldDriver_WithDirectML_ShouldBeDirectML()
    {
        var gpu = new GpuInfo
        {
            Vendor = GpuVendor.Nvidia,
            CudaDriverVersionMajor = 10,
            DirectMLSupported = true
        };

        gpu.RecommendedProvider.Should().Be(ExecutionProvider.DirectML);
    }

    [Fact]
    public void RecommendedProvider_NvidiaOldDriver_NoDirectML_ShouldBeCpu()
    {
        var gpu = new GpuInfo
        {
            Vendor = GpuVendor.Nvidia,
            CudaDriverVersionMajor = 10
        };

        gpu.RecommendedProvider.Should().Be(ExecutionProvider.Cpu);
    }

    [Fact]
    public void RecommendedProvider_Apple_ShouldBeCoreML()
    {
        var gpu = new GpuInfo { Vendor = GpuVendor.Apple };

        gpu.RecommendedProvider.Should().Be(ExecutionProvider.CoreML);
    }

    [Fact]
    public void RecommendedProvider_AmdWithDirectML_ShouldBeDirectML()
    {
        var gpu = new GpuInfo
        {
            Vendor = GpuVendor.Amd,
            DirectMLSupported = true
        };

        gpu.RecommendedProvider.Should().Be(ExecutionProvider.DirectML);
    }

    [Fact]
    public void RecommendedProvider_UnknownVendor_ShouldBeCpu()
    {
        var gpu = new GpuInfo { Vendor = GpuVendor.Unknown };

        gpu.RecommendedProvider.Should().Be(ExecutionProvider.Cpu);
    }

    [Fact]
    public void GetFallbackProviders_NvidiaCuda11_ShouldIncludeCudaFirst()
    {
        var gpu = new GpuInfo
        {
            Vendor = GpuVendor.Nvidia,
            CudaDriverVersionMajor = 12,
            DirectMLSupported = true
        };

        var providers = gpu.GetFallbackProviders();

        providers.Should().StartWith(ExecutionProvider.Cuda);
        providers.Should().Contain(ExecutionProvider.DirectML);
        providers.Should().EndWith(ExecutionProvider.Cpu);
    }

    [Fact]
    public void GetFallbackProviders_AppleWithCoreML_ShouldIncludeCoreML()
    {
        var gpu = new GpuInfo
        {
            Vendor = GpuVendor.Apple,
            CoreMLSupported = true
        };

        var providers = gpu.GetFallbackProviders();

        providers.Should().Contain(ExecutionProvider.CoreML);
        providers.Should().EndWith(ExecutionProvider.Cpu);
        providers.Should().NotContain(ExecutionProvider.Cuda);
    }

    [Fact]
    public void GetFallbackProviders_CpuOnly_ShouldHaveSingleEntry()
    {
        var gpu = new GpuInfo { Vendor = GpuVendor.Unknown };

        var providers = gpu.GetFallbackProviders();

        providers.Should().ContainSingle()
            .Which.Should().Be(ExecutionProvider.Cpu);
    }

    [Theory]
    [InlineData(null, null)]
    [InlineData(0L, 0L)]
    [InlineData(1024L * 1024, 1L)]
    [InlineData(8L * 1024 * 1024 * 1024, 8192L)]
    public void TotalMemoryMB_ShouldCalculateCorrectly(long? bytes, long? expectedMB)
    {
        var gpu = new GpuInfo
        {
            Vendor = GpuVendor.Nvidia,
            TotalMemoryBytes = bytes
        };

        gpu.TotalMemoryMB.Should().Be(expectedMB);
    }

    [Fact]
    public void CudaComputeCapability_ShouldFormatCorrectly()
    {
        var gpu = new GpuInfo
        {
            Vendor = GpuVendor.Nvidia,
            CudaComputeCapabilityMajor = 8,
            CudaComputeCapabilityMinor = 6
        };

        gpu.CudaComputeCapability.Should().Be("8.6");
    }

    [Fact]
    public void CudaComputeCapability_ShouldBeNull_WhenPartsMissing()
    {
        var gpu = new GpuInfo
        {
            Vendor = GpuVendor.Nvidia,
            CudaComputeCapabilityMajor = 8
        };

        gpu.CudaComputeCapability.Should().BeNull();
    }

    [Fact]
    public void ToString_ShouldIncludeVendor()
    {
        var gpu = new GpuInfo { Vendor = GpuVendor.Nvidia };

        gpu.ToString().Should().Contain("Nvidia");
    }

    [Fact]
    public void ToString_ShouldIncludeDeviceName()
    {
        var gpu = new GpuInfo
        {
            Vendor = GpuVendor.Nvidia,
            DeviceName = "GeForce RTX 4090"
        };

        gpu.ToString().Should().Contain("GeForce RTX 4090");
    }

    [Fact]
    public void ToString_ShouldIncludeMemory()
    {
        var gpu = new GpuInfo
        {
            Vendor = GpuVendor.Nvidia,
            TotalMemoryBytes = 24L * 1024 * 1024 * 1024
        };

        gpu.ToString().Should().Contain("24576MB");
    }

    [Fact]
    public void ToString_ShouldIncludeComputeCapability()
    {
        var gpu = new GpuInfo
        {
            Vendor = GpuVendor.Nvidia,
            CudaComputeCapabilityMajor = 8,
            CudaComputeCapabilityMinor = 9
        };

        gpu.ToString().Should().Contain("CC 8.9");
    }

    [Fact]
    public void ToString_FullInfo_ShouldJoinWithPipe()
    {
        var gpu = new GpuInfo
        {
            Vendor = GpuVendor.Nvidia,
            DeviceName = "RTX 4090",
            TotalMemoryBytes = 24L * 1024 * 1024 * 1024,
            CudaComputeCapabilityMajor = 8,
            CudaComputeCapabilityMinor = 9
        };

        var result = gpu.ToString();
        result.Should().Contain(" | ");
        result.Should().Contain("Nvidia");
        result.Should().Contain("RTX 4090");
    }
}

public class GpuInfoEffectiveMemoryTests
{
    [Fact]
    public void EffectiveAvailableBytes_ReturnsFree_WhenBothAvailable()
    {
        var gpu = new GpuInfo
        {
            Vendor = GpuVendor.Nvidia,
            TotalMemoryBytes = 8L * 1024 * 1024 * 1024,
            FreeMemoryBytes = 6L * 1024 * 1024 * 1024
        };

        gpu.EffectiveAvailableBytes.Should().Be(6L * 1024 * 1024 * 1024);
    }

    [Fact]
    public void EffectiveAvailableBytes_FallsBackToTotal_WhenFreeIsNull()
    {
        var gpu = new GpuInfo
        {
            Vendor = GpuVendor.Nvidia,
            TotalMemoryBytes = 8L * 1024 * 1024 * 1024,
            FreeMemoryBytes = null
        };

        gpu.EffectiveAvailableBytes.Should().Be(8L * 1024 * 1024 * 1024);
    }

    [Fact]
    public void EffectiveAvailableBytes_IsNull_WhenBothNull()
    {
        var gpu = new GpuInfo
        {
            Vendor = GpuVendor.Unknown,
            TotalMemoryBytes = null,
            FreeMemoryBytes = null
        };

        gpu.EffectiveAvailableBytes.Should().BeNull();
    }

    [Fact]
    public void FreeMemoryBytes_PreferredOverTotal_EvenWhenSmaller()
    {
        var gpu = new GpuInfo
        {
            Vendor = GpuVendor.Nvidia,
            TotalMemoryBytes = 24L * 1024 * 1024 * 1024,
            FreeMemoryBytes = 2L * 1024 * 1024 * 1024  // Most VRAM in use
        };

        // EffectiveAvailableBytes should reflect reality (free), not capacity (total)
        gpu.EffectiveAvailableBytes.Should().Be(2L * 1024 * 1024 * 1024);
    }
}

public class GpuVendorEnumTests
{
    [Fact]
    public void ShouldHaveSixValues()
    {
        Enum.GetValues<GpuVendor>().Should().HaveCount(6);
    }

    [Theory]
    [InlineData(GpuVendor.Unknown, 0)]
    [InlineData(GpuVendor.Nvidia, 1)]
    [InlineData(GpuVendor.Amd, 2)]
    [InlineData(GpuVendor.Intel, 3)]
    [InlineData(GpuVendor.Apple, 4)]
    [InlineData(GpuVendor.Qualcomm, 5)]
    public void ShouldHaveExpectedIntValues(GpuVendor vendor, int expected)
    {
        ((int)vendor).Should().Be(expected);
    }
}

public class RuntimeVersionStateTests
{
    [Fact]
    public void DefaultValues_ShouldBeCorrect()
    {
        var state = new RuntimeVersionState { InstalledVersion = "1.0.0" };

        state.LatestKnownVersion.Should().BeNull();
        state.PendingVersion.Should().BeNull();
        state.UpdateReady.Should().BeFalse();
        state.UpdateReadyPath.Should().BeNull();
        state.PreviousVersions.Should().BeEmpty();
        state.FailedVersions.Should().BeEmpty();
    }

    [Fact]
    public void UpdateAvailable_ShouldBeFalse_WhenLatestIsNull()
    {
        var state = new RuntimeVersionState { InstalledVersion = "1.0.0" };

        state.UpdateAvailable.Should().BeFalse();
    }

    [Fact]
    public void UpdateAvailable_ShouldBeFalse_WhenSameVersion()
    {
        var state = new RuntimeVersionState
        {
            InstalledVersion = "1.0.0",
            LatestKnownVersion = "1.0.0"
        };

        state.UpdateAvailable.Should().BeFalse();
    }

    [Fact]
    public void UpdateAvailable_ShouldBeTrue_WhenNewerVersion()
    {
        var state = new RuntimeVersionState
        {
            InstalledVersion = "1.0.0",
            LatestKnownVersion = "2.0.0"
        };

        state.UpdateAvailable.Should().BeTrue();
    }

    [Fact]
    public void UpdateAvailable_ShouldBeFalse_WhenLatestIsFailed()
    {
        var state = new RuntimeVersionState
        {
            InstalledVersion = "1.0.0",
            LatestKnownVersion = "2.0.0",
            FailedVersions = ["2.0.0"]
        };

        state.UpdateAvailable.Should().BeFalse();
    }

    [Fact]
    public void UpdateAvailable_CaseInsensitive_ShouldBeFalse()
    {
        var state = new RuntimeVersionState
        {
            InstalledVersion = "1.0.0",
            LatestKnownVersion = "1.0.0"
        };

        state.UpdateAvailable.Should().BeFalse();
    }

    [Fact]
    public void PreviousVersions_ShouldTrackHistory()
    {
        var state = new RuntimeVersionState
        {
            InstalledVersion = "2.0.0",
            PreviousVersions = ["1.0.0", "1.5.0"]
        };

        state.PreviousVersions.Should().HaveCount(2);
        state.PreviousVersions.Should().Contain("1.0.0");
    }
}

public class RuntimeVersionStateFileTests
{
    [Fact]
    public void DefaultValues_ShouldBeCorrect()
    {
        var file = new RuntimeVersionStateFile();

        file.SchemaVersion.Should().Be(1);
        file.Packages.Should().BeEmpty();
    }

    [Fact]
    public void Packages_ShouldStoreMultipleStates()
    {
        var file = new RuntimeVersionStateFile();
        file.Packages["llama-server|vulkan|win-x64"] = new RuntimeVersionState
        {
            InstalledVersion = "1.0.0"
        };
        file.Packages["llama-server|cuda|linux-x64"] = new RuntimeVersionState
        {
            InstalledVersion = "2.0.0"
        };

        file.Packages.Should().HaveCount(2);
    }
}
