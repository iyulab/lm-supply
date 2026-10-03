using AwesomeAssertions;
using LMSupply.Captioner.Inference;
using LMSupply.Captioner.Models;

namespace LMSupply.Captioner.Tests;

/// <summary>
/// The <c>quality</c> alias (Florence-2 base) beside ViT-GPT2: which files a load picks, which options it honours or
/// refuses, and how generation avoids repeats. None of these download anything; the real-model facts are in
/// <see cref="Florence2CaptionerIntegrationTests"/>.
/// </summary>
public sealed class Florence2CaptionerTests : IDisposable
{
    private readonly string _dir = Path.Combine(Path.GetTempPath(), "lmsupply-florence-" + Guid.NewGuid().ToString("N"));

    private static CancellationToken Ct => TestContext.Current.CancellationToken;

    public void Dispose()
    {
        if (Directory.Exists(_dir))
            Directory.Delete(_dir, recursive: true);
    }

    [Fact]
    public void QualityAlias_ResolvesToFlorence2()
    {
        var model = LocalCaptioner.Registry.Resolve("quality");

        model.Architecture.Should().Be(CaptionerArchitecture.Florence2);
        model.RepoId.Should().Be("onnx-community/Florence-2-base-ft");
        model.HasQuantizationVariants.Should().BeTrue();
        LocalCaptioner.GetAvailableModels().Should().Contain("quality");
    }

    [Fact]
    public void DefaultAlias_StaysViTGpt2()
    {
        LocalCaptioner.Registry.Resolve("default").Architecture.Should().Be(CaptionerArchitecture.VitGpt2);
        LocalCaptioner.Registry.Resolve("auto").Architecture.Should().Be(CaptionerArchitecture.VitGpt2);
    }

    [Theory]
    [InlineData("fp16", "_fp16")]
    [InlineData("int8", "_quantized")]
    [InlineData("q4", "_q4")]
    [InlineData("fp32", "")]
    public void QuantizationHint_PicksTheVariantFiles(string hint, string suffix)
    {
        var model = DefaultModels.Florence2Base;

        LocalCaptioner.SelectVariantSuffix(model, new CaptionerOptions { QuantizationHint = hint }).Should().Be(suffix);
        LocalCaptioner.OnnxFiles(model, suffix).Should().Equal(
            $"vision_encoder{suffix}.onnx", $"embed_tokens{suffix}.onnx", $"encoder_model{suffix}.onnx", $"decoder_model_merged{suffix}.onnx");
    }

    [Fact]
    public void ModelPublishedAtOnePrecision_IgnoresTheHint()
    {
        // ViT-GPT2's registry entry names its files outright; a hint must not invent names the repository lacks.
        LocalCaptioner.SelectVariantSuffix(DefaultModels.VitGpt2, new CaptionerOptions { QuantizationHint = "q4" })
            .Should().BeEmpty();
    }

    [Fact]
    public void RequiredFiles_IncludeTheTokenizer()
    {
        LocalCaptioner.GetRequiredFiles(DefaultModels.Florence2Base, "_quantized")
            .Should().Contain(["vision_encoder_quantized.onnx", "vocab.json", "merges.txt"]);
    }

    [Fact]
    public async Task DetailOnViTGpt2_IsRefusedBeforeAnythingIsDownloaded()
    {
        var options = new CaptionerOptions { Detail = CaptionDetail.Detailed, CacheDirectory = _dir, DisableAutoDownload = true };

        var load = () => LocalCaptioner.LoadAsync("default", options, cancellationToken: Ct);

        // NotSupported, not ModelNotFound: the refusal comes before the (offline) cache lookup.
        await load.Should().ThrowAsync<NotSupportedException>().WithMessage("*quality*");
    }

    [Fact]
    public async Task PromptOnFlorence2_IsRefusedBeforeAnythingIsDownloaded()
    {
        var options = new CaptionerOptions { Prompt = "a photo of", CacheDirectory = _dir, DisableAutoDownload = true };

        var load = () => LocalCaptioner.LoadAsync("quality", options, cancellationToken: Ct);

        await load.Should().ThrowAsync<NotSupportedException>().WithMessage("*Detail*");
    }

    [Fact]
    public void LocalDirectory_WithTheFlorenceGraphs_IsRecognised_AtTheVariantPresent()
    {
        Directory.CreateDirectory(_dir);
        File.WriteAllText(Path.Combine(_dir, "vocab.json"), "{}");
        foreach (var file in LocalCaptioner.OnnxFiles(DefaultModels.Florence2Base, "_q4"))
            File.WriteAllBytes(Path.Combine(_dir, file), []);

        LocalCaptioner.TryInferModelInfo(_dir, _dir, new CaptionerOptions { QuantizationHint = "fp16" }, out var model, out var suffix)
            .Should().BeTrue();

        model!.Architecture.Should().Be(CaptionerArchitecture.Florence2);
        suffix.Should().Be("_q4", "only that variant is on disk, whatever the options prefer");
    }

    [Fact]
    public void LocalDirectory_MissingOneFlorenceGraph_IsNotFlorence()
    {
        Directory.CreateDirectory(_dir);
        File.WriteAllText(Path.Combine(_dir, "vocab.json"), "{}");
        foreach (var file in LocalCaptioner.OnnxFiles(DefaultModels.Florence2Base, "").Skip(1))
            File.WriteAllBytes(Path.Combine(_dir, file), []);

        LocalCaptioner.TryInferModelInfo(_dir, _dir, new CaptionerOptions(), out var model, out _).Should().BeFalse();
        model.Should().BeNull();
    }

    [Theory]
    [InlineData(CaptionDetail.Brief, "What does the image describe?")]
    [InlineData(CaptionDetail.Detailed, "Describe in detail what is shown in the image.")]
    [InlineData(CaptionDetail.Paragraph, "Describe with a paragraph what is shown in the image.")]
    public void Detail_SelectsTheTaskPromptFlorenceWasTrainedOn(CaptionDetail detail, string prompt)
        => Florence2Captioner.TaskPrompt(detail).Should().Be(prompt);

    [Fact]
    public void NoRepeatNgram_BansTheTokenThatWouldRepeatATrigram()
    {
        // ... 7 8 9 ... 7 8 → 9 would repeat "7 8 9".
        var logits = new float[16];
        NextToken.BanRepeatedNgrams(logits, [2, 0, 7, 8, 9, 5, 7, 8], size: 3);

        logits[9].Should().Be(float.NegativeInfinity);
        logits.Count(float.IsNegativeInfinity).Should().Be(1);
    }

    [Fact]
    public void NoRepeatNgram_LeavesANewTrigramAlone()
    {
        var logits = new float[16];
        NextToken.BanRepeatedNgrams(logits, [2, 0, 7, 8, 9, 5], size: 3);

        logits.Should().NotContain(float.NegativeInfinity);
    }

    [Fact]
    public async Task DownloadSize_OfALocalDirectory_IsZero()
    {
        Directory.CreateDirectory(_dir);
        (await LocalCaptioner.GetDownloadSizeBytesAsync(_dir, cancellationToken: Ct)).Should().Be(0);
    }
}
