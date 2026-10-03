using AwesomeAssertions;
using LMSupply.Captioner.Models;

namespace LMSupply.Captioner.Tests;

/// <summary>
/// Every built-in model carries the licence of its weights, so a consent screen can state it without reading a
/// conversion repository's card (which often states none).
/// </summary>
public class CatalogLicenseTests
{
    [Fact]
    public void EveryCatalogModel_DeclaresItsLicense()
    {
        CaptionerModelRegistry.Default.GetAvailableModels().Should().NotBeEmpty()
            .And.OnlyContain(m => !string.IsNullOrWhiteSpace(m.License), "captioner: an uncurated licence reads as \"none\" on a consent screen");
    }

    [Theory]
    [InlineData("default", "Apache-2.0")] // ViT-GPT2: the conversion's card states none; the model it converts is Apache-2.0
    [InlineData("quality", "MIT")]
    public void Alias_ReportsTheConvertedModelsLicense(string alias, string license)
        => CaptionerModelRegistry.Default.Resolve(alias).License.Should().Be(license);
}
