using AwesomeAssertions;
using LMSupply.Generator.Abstractions;
using Xunit;

namespace LMSupply.Generator.Tests;

/// <summary>
/// Tests for the VRAM-budget telemetry fields added to GeneratorModelInfo so downstream
/// consumers can classify "(a) accurately-small VRAM" vs "(b) under-reported budget" from
/// GetModelInfo() instead of inferring brick state from a magic-number 512 in the logs.
/// </summary>
public class GeneratorModelInfoTelemetryTests
{
    private static GeneratorModelInfo Sample() =>
        new("m", "/path", 4096, "chatml", "llama-server-Cpu");

    [Fact]
    public void TelemetryFields_DefaultToNullAndFalse()
    {
        var info = Sample();

        info.VramBudgetBytes.Should().BeNull();
        info.VramFreeBytes.Should().BeNull();
        info.VramTotalBytes.Should().BeNull();
        info.ContextFlooredByVram.Should().BeFalse();
    }

    [Fact]
    public void TelemetryFields_RoundTripThroughInitializers()
    {
        var info = Sample() with
        {
            VramBudgetBytes = 2_100L * 1024 * 1024,
            VramFreeBytes = 7_957L * 1024 * 1024,
            VramTotalBytes = 8_188L * 1024 * 1024,
            ContextFlooredByVram = true,
        };

        info.VramBudgetBytes.Should().Be(2_100L * 1024 * 1024);
        info.VramFreeBytes.Should().Be(7_957L * 1024 * 1024);
        info.VramTotalBytes.Should().Be(8_188L * 1024 * 1024);
        info.ContextFlooredByVram.Should().BeTrue();
    }
}
