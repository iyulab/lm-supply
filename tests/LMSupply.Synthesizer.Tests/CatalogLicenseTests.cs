using AwesomeAssertions;
using LMSupply.Synthesizer.Models;

namespace LMSupply.Synthesizer.Tests;

/// <summary>
/// Every built-in model carries the licence of its weights. The entry type has no default licence, so a model the
/// catalog does not curate reports none instead of one nobody declared — and a built-in entry that forgot its licence
/// fails here instead of reading as "none" on a consent screen.
/// </summary>
public class CatalogLicenseTests
{
    [Fact]
    public void EveryCatalogModel_DeclaresItsLicense()
    {
        // "chinese" (huayan): the voice's own model card states no licence, so the catalog says none rather than one.
        SynthesizerModelRegistry.Default.GetAvailableModels().Should().NotBeEmpty()
            .And.OnlyContain(m => !string.IsNullOrWhiteSpace(m.License) || m.AliasName == "chinese",
                "an uncurated licence reads as \"none\" on a consent screen");
        SynthesizerModelRegistry.Default.Resolve("chinese").License.Should().BeNull();
    }

    [Fact]
    public void AModelTheCatalogDoesNotKnow_ReportsNoLicense_NotAGuess()
    {
        var model = SynthesizerModelRegistry.Default.Resolve("someone/not-in-the-catalog");

        model.License.Should().BeNull("nobody declared one; before 0.104.0 the entry type defaulted to a licence of its own");
    }
}
