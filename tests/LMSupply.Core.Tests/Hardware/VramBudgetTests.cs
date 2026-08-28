using AwesomeAssertions;
using LMSupply.Hardware;
using LMSupply.Runtime;
using System.Runtime.InteropServices;

namespace LMSupply.Core.Tests.Hardware;

public class VramBudgetTests
{
    private const long GB = 1024L * 1024 * 1024;

    [Fact]
    public void GetRecommendedSafetyMargin_LowVramNvidiaOnWindows_Returns25Percent()
    {
        if (!RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            return; // Behavior is Windows-specific

        var gpu = new GpuInfo
        {
            Vendor = GpuVendor.Nvidia,
            DeviceName = "NVIDIA RTX 4060 Laptop GPU",
            TotalMemoryBytes = 4 * GB,
            FreeMemoryBytes = 3 * GB,
        };

        var margin = VramBudget.GetRecommendedSafetyMargin(gpu);

        margin.Should().Be(0.25);
    }

    [Fact]
    public void GetRecommendedSafetyMargin_HighVramNvidia_ReturnsDefault()
    {
        var gpu = new GpuInfo
        {
            Vendor = GpuVendor.Nvidia,
            DeviceName = "NVIDIA RTX 4090",
            TotalMemoryBytes = 24 * GB,
            FreeMemoryBytes = 22 * GB,
        };

        var margin = VramBudget.GetRecommendedSafetyMargin(gpu);

        margin.Should().Be(VramBudget.DefaultSafetyMargin);
    }

    [Fact]
    public void GetRecommendedSafetyMargin_NoGpu_ReturnsDefault()
    {
        var gpu = new GpuInfo
        {
            Vendor = GpuVendor.Unknown,
            TotalMemoryBytes = null,
            FreeMemoryBytes = null,
        };

        var margin = VramBudget.GetRecommendedSafetyMargin(gpu);

        margin.Should().Be(VramBudget.DefaultSafetyMargin);
    }

    [Fact]
    public void GetAvailableBytes_NoMarginArg_AppliesRecommendedMargin()
    {
        if (!RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            return;

        var gpu = new GpuInfo
        {
            Vendor = GpuVendor.Nvidia,
            DeviceName = "NVIDIA RTX 4060 Laptop GPU",
            TotalMemoryBytes = 4 * GB,
            FreeMemoryBytes = 3 * GB,
        };

        var available = VramBudget.GetAvailableBytes(gpu);

        // Low-VRAM Windows margin = 0.25 → totalCap = 4GB × 0.75 = 3.0GB
        // freeCap = 3GB × 0.95 = 2.85GB → budget = min(3.0, 2.85) = 2.85GB.
        var expected = (long)(3 * GB * VramBudget.FreeVramSafetyFactor);
        available.Should().Be(expected);
    }

    [Fact]
    public void GetAvailableBytes_FreeIsBindingWhenLower()
    {
        // 16GB total, 12GB free → totalCap = 16 × 0.85 = 13.6GB, freeCap = 12 × 0.95 = 11.4GB
        // Budget = min(13.6, 11.4) = 11.4GB — free is the binding constraint.
        var gpu = new GpuInfo
        {
            Vendor = GpuVendor.Nvidia,
            TotalMemoryBytes = 16 * GB,
            FreeMemoryBytes = 12 * GB,
        };

        var available = VramBudget.GetAvailableBytes(gpu);

        var expected = (long)(12 * GB * VramBudget.FreeVramSafetyFactor);
        available.Should().Be(expected);
    }

    [Fact]
    public void GetAvailableBytes_TotalIsBindingWhenFreeIsHigh()
    {
        // 16GB total, 15.5GB free → totalCap = 16 × 0.85 = 13.6GB, freeCap = 15.5 × 0.95 = 14.7GB
        // Budget = min(13.6, 14.7) = 13.6GB — total cap is the binding constraint (normal case).
        var gpu = new GpuInfo
        {
            Vendor = GpuVendor.Nvidia,
            TotalMemoryBytes = 16 * GB,
            FreeMemoryBytes = (long)(15.5 * GB),
        };

        var available = VramBudget.GetAvailableBytes(gpu);

        available.Should().Be((long)(16 * GB * (1.0 - VramBudget.DefaultSafetyMargin)));
    }

    [Fact]
    public void GetAvailableBytes_WithoutFreeMemory_UsesTotalWithSafetyMargin()
    {
        // Arrange: 8GB total, null free → expect ~6.8GB (8 * 0.85)
        var gpu = new GpuInfo
        {
            Vendor = GpuVendor.Intel,
            TotalMemoryBytes = 8 * GB,
            FreeMemoryBytes = null,
        };

        // Act
        var available = VramBudget.GetAvailableBytes(gpu);

        // Assert: 8GB * 0.85 = 6.8GB
        var expected = (long)(8 * GB * 0.85);
        available.Should().Be(expected);
    }

    [Fact]
    public void GetAvailableBytes_CpuOnly_ReturnsZero()
    {
        // Arrange: Unknown vendor, no memory info
        var gpu = new GpuInfo
        {
            Vendor = GpuVendor.Unknown,
            TotalMemoryBytes = null,
            FreeMemoryBytes = null,
        };

        // Act
        var available = VramBudget.GetAvailableBytes(gpu);

        // Assert
        available.Should().Be(0);
    }

    [Fact]
    public void GetAvailableBytes_CustomSafetyMargin()
    {
        // 16GB total, 10GB free, margin=0.1
        // totalCap = 16 × 0.9 = 14.4GB, freeCap = 10 × 0.95 = 9.5GB → budget = min = 9.5GB
        var gpu = new GpuInfo
        {
            Vendor = GpuVendor.Nvidia,
            TotalMemoryBytes = 16 * GB,
            FreeMemoryBytes = 10 * GB,
        };

        var available = VramBudget.GetAvailableBytes(gpu, safetyMargin: 0.1);

        var expected = (long)(10 * GB * VramBudget.FreeVramSafetyFactor);
        available.Should().Be(expected);
    }

    [Fact]
    public void GetAvailableBytes_EnvOverride_BypassesMargin()
    {
        var gpu = new GpuInfo
        {
            Vendor = GpuVendor.Nvidia,
            TotalMemoryBytes = 16 * GB,
            FreeMemoryBytes = 2 * GB,
        };

        var previous = Environment.GetEnvironmentVariable(VramBudget.BudgetOverrideEnvVar);
        try
        {
            Environment.SetEnvironmentVariable(VramBudget.BudgetOverrideEnvVar, "8000");

            var available = VramBudget.GetAvailableBytes(gpu);

            // 8000 MB exactly, no margin applied
            available.Should().Be(8000L * 1024 * 1024);
        }
        finally
        {
            Environment.SetEnvironmentVariable(VramBudget.BudgetOverrideEnvVar, previous);
        }
    }

    [Fact]
    public void CanFitModel_ModelFits_ReturnsTrue()
    {
        // Arrange: 4GB model, 12GB free
        var gpu = new GpuInfo
        {
            Vendor = GpuVendor.Nvidia,
            TotalMemoryBytes = 16 * GB,
            FreeMemoryBytes = 12 * GB,
        };
        var modelSize = 4 * GB;

        // Act
        var result = VramBudget.CanFitModel(gpu, modelSize);

        // budget = min(16×0.85, 12×0.95) = min(13.6, 11.4) = 11.4GB > 4GB → fits
        result.Should().BeTrue();
    }

    [Fact]
    public void CanFitModel_ModelTooLarge_ReturnsFalse()
    {
        // Arrange: 8GB model on 6GB card (total) → can't fit even with zero-margin.
        var gpu = new GpuInfo
        {
            Vendor = GpuVendor.Amd,
            TotalMemoryBytes = 6 * GB,
            FreeMemoryBytes = 4 * GB,
        };
        var modelSize = 8 * GB;

        // Act
        var result = VramBudget.CanFitModel(gpu, modelSize);

        // budget = min(6×0.85, 4×0.95) = min(5.1, 3.8) = 3.8GB < 8GB → cannot fit
        result.Should().BeFalse();
    }

    [Fact]
    public void GetAvailableBytes_LowFreeVram_FreeCapsSelection()
    {
        // Regression: RTX 3090 (24GB total) with external process consuming most VRAM.
        // Old behavior: budget = 24 × 0.85 = 20890MB → 26B model selected → 0 GPU layers.
        // New behavior: budget = min(20480, 2384) = 2384MB → correct (model won't fit).
        const long totalMb = 24576;
        const long freeMb = 2509;
        var gpu = new GpuInfo
        {
            Vendor = GpuVendor.Nvidia,
            DeviceName = "NVIDIA GeForce RTX 3090",
            TotalMemoryBytes = totalMb * 1024L * 1024,
            FreeMemoryBytes = freeMb * 1024L * 1024,
        };

        var available = VramBudget.GetAvailableBytes(gpu);

        // freeCap = 2509 × 0.95 = ~2383MB — must be the binding constraint
        var freeCap = (long)(freeMb * 1024L * 1024 * VramBudget.FreeVramSafetyFactor);
        var totalCap = (long)(totalMb * 1024L * 1024 * (1.0 - VramBudget.DefaultSafetyMargin));
        freeCap.Should().BeLessThan(totalCap, "free is the binding constraint in this scenario");
        available.Should().Be(freeCap);

        // A 26B model (~18962MB) should not fit in the corrected budget
        const long modelBytes = 18962L * 1024 * 1024;
        VramBudget.CanFitModel(gpu, modelBytes).Should().BeFalse(
            because: "26B model must not be selected when only 2509MB is actually free");
    }
}
