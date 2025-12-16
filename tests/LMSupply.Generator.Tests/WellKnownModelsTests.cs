using FluentAssertions;

namespace LMSupply.Generator.Tests;

public class WellKnownModelsTests
{
    [Fact]
    public void Generator_Default_IsPhi4Mini()
    {
        // Assert
        WellKnownModels.Generator.Default.Should().Be("microsoft/Phi-4-mini-instruct-onnx");
    }

    [Fact]
    public void Generator_Quality_IsPhi4()
    {
        // Assert
        WellKnownModels.Generator.Quality.Should().Be("microsoft/phi-4-onnx");
    }

    [Fact]
    public void Generator_Fast_IsLlama1B()
    {
        // Assert
        WellKnownModels.Generator.Fast.Should().Contain("Llama-3.2-1B");
    }

    [Fact]
    public void Generator_Small_IsSameAsFast()
    {
        // Assert
        WellKnownModels.Generator.Small.Should().Be(WellKnownModels.Generator.Fast);
    }

    [Fact]
    public void GetLicenseTier_DefaultModel_ReturnsMIT()
    {
        // Act
        var tier = WellKnownModels.GetLicenseTier(WellKnownModels.Generator.Default);

        // Assert
        tier.Should().Be(LicenseTier.MIT);
    }

    [Fact]
    public void GetLicenseTier_Llama_ReturnsConditional()
    {
        // Act
        var tier = WellKnownModels.GetLicenseTier(WellKnownModels.Generator.Fast);

        // Assert
        tier.Should().Be(LicenseTier.Conditional);
    }

    [Fact]
    public void HasRestrictions_MITModel_ReturnsFalse()
    {
        // Act
        var hasRestrictions = WellKnownModels.HasRestrictions(WellKnownModels.Generator.Default);

        // Assert
        hasRestrictions.Should().BeFalse();
    }

    [Fact]
    public void HasRestrictions_ConditionalModel_ReturnsTrue()
    {
        // Act
        var hasRestrictions = WellKnownModels.HasRestrictions(WellKnownModels.Generator.Fast);

        // Assert
        hasRestrictions.Should().BeTrue();
    }

    [Fact]
    public void GetUnrestrictedModels_ContainsMITModelsOnly()
    {
        // Act
        var models = WellKnownModels.GetUnrestrictedModels();

        // Assert
        models.Should().Contain(WellKnownModels.Generator.Default);
        models.Should().Contain(WellKnownModels.Generator.Quality);
        models.Should().NotContain(WellKnownModels.Generator.Fast);
    }

    [Fact]
    public void Embedder_Default_IsNotEmpty()
    {
        // Assert
        WellKnownModels.Embedder.Default.Should().NotBeNullOrEmpty();
    }

    [Fact]
    public void Reranker_Default_IsNotEmpty()
    {
        // Assert
        WellKnownModels.Reranker.Default.Should().NotBeNullOrEmpty();
    }
}
