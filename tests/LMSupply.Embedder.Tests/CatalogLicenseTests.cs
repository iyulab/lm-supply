using AwesomeAssertions;
using LMSupply.Embedder.Utils;

namespace LMSupply.Embedder.Tests;

/// <summary>
/// Every built-in model carries the licence of its weights, so a consent screen can state it without reading a
/// conversion repository's card (which often states none).
/// </summary>
public class CatalogLicenseTests
{
    [Fact]
    public void EveryCatalogModel_DeclaresItsLicense()
    {
        EmbedderModelRegistry.Default.GetAvailableModels().Should().NotBeEmpty()
            .And.OnlyContain(m => !string.IsNullOrWhiteSpace(m.License), "embedder: an uncurated licence reads as \"none\" on a consent screen");
    }
}
