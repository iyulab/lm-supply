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
    public void Generator_Advanced_IsPhi4()
    {
        // Assert
        WellKnownModels.Generator.Advanced.Should().Be("microsoft/phi-4-onnx");
    }

    [Fact]
    public void Generator_Entry_IsPhi4Mini()
    {
        // Entry pins the ONNX path explicitly (Phi-4 Mini).
        WellKnownModels.Generator.Entry.Should().Be("phi-4-mini");
    }

    [Fact]
    public void GetLicenseTier_EntryModel_ReturnsMIT()
    {
        // Entry = "phi-4-mini" resolves to the MIT-licensed Phi-4 Mini.
        var tier = WellKnownModels.GetLicenseTier(WellKnownModels.Generator.Entry);

        tier.Should().Be(LicenseTier.MIT);
    }

    [Fact]
    public void HasRestrictions_EntryModel_ReturnsFalse()
    {
        // Entry is "phi-4-mini" (MIT, no restrictions)
        var hasRestrictions = WellKnownModels.HasRestrictions(WellKnownModels.Generator.Entry);

        hasRestrictions.Should().BeFalse();
    }

    [Fact]
    public void GetUnrestrictedModels_ContainsEntryAndAdvanced()
    {
        var models = WellKnownModels.GetUnrestrictedModels();

        // Entry ("phi-4-mini") and Advanced are both MIT-licensed Phi-4 models.
        // The platform-aware Default alias is intentionally excluded because it
        // routes through auto selection and has no single license tier.
        models.Should().Contain(WellKnownModels.Generator.Entry);
        models.Should().Contain(WellKnownModels.Generator.Advanced);
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
}
