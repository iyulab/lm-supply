using AwesomeAssertions;
using LMSupply.Translator.Models;

namespace LMSupply.Translator.Tests;

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
        TranslatorModelRegistry.Default.GetAvailableModels().Should().NotBeEmpty()
            .And.OnlyContain(m => !string.IsNullOrWhiteSpace(m.License), "an uncurated licence reads as \"none\" on a consent screen");
    }

    [Fact]
    public void AModelTheCatalogDoesNotKnow_ReportsNoLicense_NotAGuess()
    {
        var model = TranslatorModelRegistry.Default.Resolve("someone/not-in-the-catalog");

        model.License.Should().BeNull("nobody declared one; before 0.104.0 the entry type defaulted to a licence of its own");
    }
}
