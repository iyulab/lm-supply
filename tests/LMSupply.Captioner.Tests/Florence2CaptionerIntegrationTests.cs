using AwesomeAssertions;
using LMSupply.Captioner.Inference;
using LMSupply.Captioner.Models;
using LMSupply.Text;

namespace LMSupply.Captioner.Tests;

/// <summary>
/// The <c>quality</c> alias against the real model (downloads Florence-2 base at the variant this machine's tier picks,
/// about 275 MB at int8).
/// </summary>
[Trait("Category", "Integration")]
public sealed class Florence2CaptionerIntegrationTests
{
    private static CancellationToken Ct => TestContext.Current.CancellationToken;

    private static string Fixture(string name) => Path.Combine(AppContext.BaseDirectory, "Fixtures", name);

    [Fact]
    public async Task Quality_NamesTheFruitInABowlOfFruit()
    {
        await using var captioner = await LocalCaptioner.LoadAsync("quality", cancellationToken: Ct);

        var result = await captioner.CaptionAsync(Fixture("fruit-bowl.jpg"), Ct);

        result.Caption.Should().MatchRegex("(?i)(banana|apple|fruit|tangerine|orange)");
        result.Caption.Should().NotContain("<").And.NotContain("</s>");
    }

    [Fact]
    public async Task Quality_DetailedCaptionIsLongerThanTheBriefOne()
    {
        await using var brief = await LocalCaptioner.LoadAsync("quality", cancellationToken: Ct);
        await using var detailed = await LocalCaptioner.LoadAsync(
            "quality", new CaptionerOptions { Detail = CaptionDetail.Paragraph, MaxLength = 150 }, cancellationToken: Ct);

        var b = await brief.CaptionAsync(Fixture("fruit-bowl.jpg"), Ct);
        var d = await detailed.CaptionAsync(Fixture("fruit-bowl.jpg"), Ct);

        d.Caption.Length.Should().BeGreaterThan(b.Caption.Length);
    }

    [Theory]
    [InlineData("quality")]
    [InlineData("fast")]
    public async Task BeamSearch_ReturnsTheBestCaption_AndTheRunnersUp(string alias)
    {
        await using var captioner = await LocalCaptioner.LoadAsync(alias, new CaptionerOptions { NumBeams = 3 }, cancellationToken: Ct);

        var result = await captioner.CaptionAsync(Fixture("fruit-bowl.jpg"), Ct);

        result.Caption.Should().MatchRegex("(?i)(banana|apple|fruit|tangerine|orange)");
        result.AlternativeCaptions.Should().NotContain(result.Caption);
        result.Confidence.Should().BeInRange(0f, 1f);
    }

    [Fact]
    public async Task PromptTokens_MatchTheReferenceTokenizer()
    {
        // The ids Hugging Face's Florence-2 processor produces for the <CAPTION> task (tokenizers 0.23).
        await using var captioner = await LocalCaptioner.LoadAsync("quality", cancellationToken: Ct);
        var modelDir = Path.GetDirectoryName(Directory.GetFiles(
            Path.Combine(LMSupply.Download.CacheManager.GetDefaultCacheDirectory()), "vision_encoder*.onnx", SearchOption.AllDirectories)
            .First(p => p.Contains("Florence-2-base-ft", StringComparison.Ordinal)))!;

        var ids = Florence2Captioner.BuildPromptIds(TokenizerFactory.CreateGpt2(modelDir), DefaultModels.Florence2Base, CaptionDetail.Brief);

        ids.Should().Equal(0, 2264, 473, 5, 2274, 6190, 116, 2);
    }

    [Fact]
    public async Task DownloadSize_MatchesTheVariantTheLoadPicks()
    {
        var fp16 = await LocalCaptioner.GetDownloadSizeBytesAsync("quality:fp16", cancellationToken: Ct);
        var int8 = await LocalCaptioner.GetDownloadSizeBytesAsync("quality", new CaptionerOptions { QuantizationHint = "int8" }, Ct);

        int8.Should().BeInRange(250_000_000, 300_000_000);
        fp16.Should().BeGreaterThan(int8);
    }
}
