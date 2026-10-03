using AwesomeAssertions;
using LMSupply.Ocr.Models;

namespace LMSupply.Ocr.Tests;

/// <summary>
/// Every built-in model carries the licence of its weights, so a consent screen can state it without reading a
/// conversion repository's card (which often states none).
/// </summary>
public class CatalogLicenseTests
{
    [Fact]
    public void EveryCatalogModel_DeclaresItsLicense()
    {
        OcrDetectionModelRegistry.Default.GetAvailableModels().Should().NotBeEmpty()
            .And.OnlyContain(m => !string.IsNullOrWhiteSpace(m.License), "detection: an uncurated licence reads as \"none\" on a consent screen");
        OcrRecognitionModelRegistry.Default.GetAvailableModels().Should().NotBeEmpty()
            .And.OnlyContain(m => !string.IsNullOrWhiteSpace(m.License), "recognition: an uncurated licence reads as \"none\" on a consent screen");
    }

    [Fact]
    public void Pipeline_ReportsTheSharedLicense()
        => new OcrModelInfo(OcrDetectionModelRegistry.Default.Resolve("default"), OcrRecognitionModelRegistry.Default.Resolve("ko"))
            .License.Should().Be("Apache-2.0");
}
