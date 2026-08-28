using AwesomeAssertions;
using LMSupply.Hardware;

namespace LMSupply.Core.Tests.Hardware;

/// <summary>
/// Tests for the LMSUPPLY_SYSTEM_RAM_MB override — lets a CPU/RAM budget be forced below the physical
/// RAM (cgroup-limited containers, low-spec simulation). Mirrors LMSUPPLY_VRAM_BUDGET_MB for VRAM.
/// Serialized via a collection because it mutates a process environment variable.
/// </summary>
[Collection("HardwareEnv")]
public class HardwareProfileRamOverrideTests
{
    private const string Var = "LMSUPPLY_SYSTEM_RAM_MB";

    [Fact]
    public void Refresh_HonorsSystemRamOverride()
    {
        var prev = Environment.GetEnvironmentVariable(Var);
        try
        {
            Environment.SetEnvironmentVariable(Var, "6000");
            var profile = HardwareProfile.Refresh();
            profile.SystemMemoryBytes.Should().Be(6000L * 1024 * 1024);
        }
        finally
        {
            Environment.SetEnvironmentVariable(Var, prev);
        }
    }

    [Fact]
    public void Refresh_NoOverride_UsesDetectedRam()
    {
        var prev = Environment.GetEnvironmentVariable(Var);
        try
        {
            Environment.SetEnvironmentVariable(Var, null);
            var profile = HardwareProfile.Refresh();
            profile.SystemMemoryBytes.Should().BeGreaterThan(0);
        }
        finally
        {
            Environment.SetEnvironmentVariable(Var, prev);
        }
    }

    [Fact]
    public void Refresh_InvalidOverride_FallsBackToDetectedRam()
    {
        var prev = Environment.GetEnvironmentVariable(Var);
        try
        {
            Environment.SetEnvironmentVariable(Var, "not-a-number");
            var profile = HardwareProfile.Refresh();
            profile.SystemMemoryBytes.Should().BeGreaterThan(6000L * 1024 * 1024,
                "an unparseable override is ignored, so detected physical RAM is used");
        }
        finally
        {
            Environment.SetEnvironmentVariable(Var, prev);
        }
    }
}
