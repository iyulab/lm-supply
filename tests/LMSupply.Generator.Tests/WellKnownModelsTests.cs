using AwesomeAssertions;

namespace LMSupply.Generator.Tests;

public class WellKnownModelsTests
{
    [Fact]
    public void Generator_Default_IsAutoAlias()
    {
        // Default now routes through hardware-aware auto selection.
        WellKnownModels.Generator.Default.Should().Be("default");
    }

    [Fact]
    public void Generator_Quality_IsPhi4()
    {
        // Assert
        WellKnownModels.Generator.Quality.Should().Be("microsoft/phi-4-onnx");
    }

    [Fact]
    public void Generator_Fast_IsPhi4MiniAlias()
    {
        // Fast pins the ONNX path explicitly (Phi-4 Mini alias).
        WellKnownModels.Generator.Fast.Should().Be("phi-4-mini");
    }

    [Fact]
    public void Generator_Small_IsSameAsFast()
    {
        // Assert
        WellKnownModels.Generator.Small.Should().Be(WellKnownModels.Generator.Fast);
    }

    [Fact]
    public void GetLicenseTier_FastModel_ReturnsMIT()
    {
        // Fast = "phi-4-mini" resolves to the MIT-licensed Phi-4 Mini.
        var tier = WellKnownModels.GetLicenseTier(WellKnownModels.Generator.Fast);

        tier.Should().Be(LicenseTier.MIT);
    }

    [Fact]
    public void HasRestrictions_FastModel_ReturnsFalse()
    {
        // Fast is "phi-4-mini" (MIT, no restrictions)
        var hasRestrictions = WellKnownModels.HasRestrictions(WellKnownModels.Generator.Fast);

        hasRestrictions.Should().BeFalse();
    }

    [Fact]
    public void GetUnrestrictedModels_ContainsFastAndQuality()
    {
        var models = WellKnownModels.GetUnrestrictedModels();

        // Fast ("phi-4-mini" alias) and Quality are both MIT-licensed Phi-4 models.
        // The platform-aware Default alias is intentionally excluded because it
        // routes through auto selection and has no single license tier.
        models.Should().Contain(WellKnownModels.Generator.Fast);
        models.Should().Contain(WellKnownModels.Generator.Quality);
    }

    [Fact]
    public void GetUnrestrictedModels_AllEntriesAreMit()
    {
        foreach (var modelId in WellKnownModels.GetUnrestrictedModels())
        {
            WellKnownModels.HasRestrictions(modelId).Should().BeFalse(
                $"GetUnrestrictedModels should only contain MIT-licensed entries but '{modelId}' was flagged as restricted");
        }
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
