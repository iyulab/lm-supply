using System.Runtime.InteropServices;
using AwesomeAssertions;
using LMSupply.Hardware;
using LMSupply.Runtime;

namespace LMSupply.Core.Tests.Runtime;

/// <summary>An Apple Silicon GPU's memory is the system's: it is given the budget Metal states for it.</summary>
public class AppleGpuMemoryTests
{
    private const long GiB = 1024L * 1024 * 1024;

    [Theory]
    [InlineData(8, 8 * GiB / 3 * 2)]
    [InlineData(36, 36 * GiB / 3 * 2)]
    [InlineData(64, 64 * GiB / 4 * 3)]
    [InlineData(96, 96 * GiB / 4 * 3)]
    public void WithoutMetal_TheGpuIsGivenTheShareMacOsWiresForItByDefault(long systemGiB, long expected)
    {
        AppleGpuMemory.DefaultShare(systemGiB * GiB).Should().Be(expected);
    }

    [Fact]
    public void AnUnknownSystemMemory_GivesNoBudget()
    {
        AppleGpuMemory.DefaultShare(0).Should().BeNull();
    }

    [Fact]
    public void OnAppleSilicon_TheGpuHasTheBudgetMetalStates_AndTheTierIsNotThatOfAGpuWithNoMemory()
    {
        if (!OperatingSystem.IsMacOS() || RuntimeInformation.OSArchitecture != Architecture.Arm64)
        {
            Assert.Skip("Apple Silicon only.");
        }

        var budget = AppleGpuMemory.WorkingSetBytes();
        budget.Should().BePositive("a Mac's GPU can use part of the system's memory");
        budget.Should().BeLessThanOrEqualTo(GC.GetGCMemoryInfo().TotalAvailableMemoryBytes);

        var gpu = GpuDetector.DetectPrimaryGpu();
        gpu.Vendor.Should().Be(GpuVendor.Apple);
        gpu.TotalMemoryBytes.Should().Be(budget);

        // ONNX sessions run on the CPU on a Mac (CoreML only when asked for), so the tier is the CPU tier for the
        // system's memory — not Low, as it was while the GPU, reporting no memory at all, decided it.
        HardwareProfile.Current.RecommendedProvider.Should().Be(ExecutionProvider.Cpu);
        var expected = HardwareProfile.Current.SystemMemoryGB switch
        {
            >= 32 => PerformanceTier.High,
            >= 16 => PerformanceTier.Medium,
            _ => PerformanceTier.Low,
        };
        HardwareProfile.Current.Tier.Should().Be(expected);
    }
}
