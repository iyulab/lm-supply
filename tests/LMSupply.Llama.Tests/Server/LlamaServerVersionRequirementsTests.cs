using AwesomeAssertions;
using LMSupply.Llama.Server;

namespace LMSupply.Llama.Tests.Server;

public class LlamaServerVersionRequirementsTests
{
    // --- ParseBuildNumber ---

    [Theory]
    [InlineData("b8672", 8672)]
    [InlineData("b1234", 1234)]
    [InlineData("b99999", 99999)]
    [InlineData("B8672", 8672)] // case insensitive
    public void ParseBuildNumber_ValidVersion_ReturnsNumber(string version, int expected)
    {
        LlamaServerVersionRequirements.ParseBuildNumber(version).Should().Be(expected);
    }

    [Theory]
    [InlineData(null)]
    [InlineData("")]
    [InlineData("  ")]
    [InlineData("unknown")]
    [InlineData("v1.0.0")]
    public void ParseBuildNumber_InvalidVersion_ReturnsNull(string? version)
    {
        LlamaServerVersionRequirements.ParseBuildNumber(version).Should().BeNull();
    }

    // --- GetMinimumBuild ---

    [Fact]
    public void GetMinimumBuild_Gemma4_Returns8672()
    {
        LlamaServerVersionRequirements.GetMinimumBuild("gemma4").Should().Be(8672);
    }

    [Fact]
    public void GetMinimumBuild_UnknownFormat_ReturnsNull()
    {
        LlamaServerVersionRequirements.GetMinimumBuild("chatml").Should().BeNull();
        LlamaServerVersionRequirements.GetMinimumBuild("llama3").Should().BeNull();
    }

    // --- Validate ---

    [Fact]
    public void Validate_SufficientVersion_DoesNotThrow()
    {
        var act = () => LlamaServerVersionRequirements.Validate("b9000", "gemma4");
        act.Should().NotThrow();
    }

    [Fact]
    public void Validate_ExactMinimumVersion_DoesNotThrow()
    {
        var act = () => LlamaServerVersionRequirements.Validate("b8672", "gemma4");
        act.Should().NotThrow();
    }

    [Fact]
    public void Validate_OldVersion_ThrowsWithMessage()
    {
        var act = () => LlamaServerVersionRequirements.Validate("b7898", "gemma4");

        act.Should().Throw<InvalidOperationException>()
            .WithMessage("*too old*gemma4*b8672*");
    }

    [Fact]
    public void Validate_NoRequirement_DoesNotThrow()
    {
        var act = () => LlamaServerVersionRequirements.Validate("b1000", "chatml");
        act.Should().NotThrow();
    }

    [Fact]
    public void Validate_UnparsableVersion_DoesNotThrow()
    {
        // Graceful degradation — unknown version format should not block model loading
        var act = () => LlamaServerVersionRequirements.Validate("unknown", "gemma4");
        act.Should().NotThrow();
    }

    [Fact]
    public void Validate_NullVersion_DoesNotThrow()
    {
        var act = () => LlamaServerVersionRequirements.Validate(null, "gemma4");
        act.Should().NotThrow();
    }

    // --- MeetsMinimum ---

    [Fact]
    public void MeetsMinimum_SufficientVersion_ReturnsTrue()
    {
        LlamaServerVersionRequirements.MeetsMinimum("b9000", "gemma4").Should().BeTrue();
    }

    [Fact]
    public void MeetsMinimum_ExactMinimumVersion_ReturnsTrue()
    {
        LlamaServerVersionRequirements.MeetsMinimum("b8672", "gemma4").Should().BeTrue();
    }

    [Fact]
    public void MeetsMinimum_OldVersion_ReturnsFalse()
    {
        LlamaServerVersionRequirements.MeetsMinimum("b7898", "gemma4").Should().BeFalse();
    }

    [Fact]
    public void MeetsMinimum_NoRequirement_ReturnsTrue()
    {
        LlamaServerVersionRequirements.MeetsMinimum("b1000", "chatml").Should().BeTrue();
    }

    [Fact]
    public void MeetsMinimum_UnparsableVersion_ReturnsTrue()
    {
        // Graceful degradation — unknown version format should not block model loading
        LlamaServerVersionRequirements.MeetsMinimum("unknown", "gemma4").Should().BeTrue();
    }

    [Fact]
    public void MeetsMinimum_NullVersion_ReturnsTrue()
    {
        LlamaServerVersionRequirements.MeetsMinimum(null, "gemma4").Should().BeTrue();
    }
}
