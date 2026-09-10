using AwesomeAssertions;
using LMSupply.Core.Download;
using LMSupply.Download;
using LMSupply.Hardware;
using Xunit;

namespace LMSupply.Core.Tests.Download;

/// <summary>
/// A caller that names an exact model file must get that file.
///
/// <para>
/// It did not. <c>ResolveModelAsync</c>'s <c>expectedOnnxFile</c> was passed no further than the local
/// lookup: the download was steered by the quantization preference derived from the machine's hardware
/// tier, and on a repository that publishes int8 builds alongside the plain weights, a mid-tier machine
/// fetched the int8 and never the file that was named. The resolver then found the named file missing and
/// handed over the first ONNX file it could see, without a word.
/// </para>
/// <para>
/// Nothing failed. The wrong model loaded, ran, and returned plausible detections — measured on one such
/// model, 8.5 times slower than the weights that were asked for (20.0 ms against 2.34 ms), and firing on a
/// synthetic gradient the real weights ignore. Both symptoms read as facts about the model rather than as
/// evidence that a different model was loaded, which is what makes this the worst shape of failure: the
/// only trace it left was in the timings.
/// </para>
/// </summary>
public class RequestedOnnxFileTests
{
    private static readonly string[] RepoWithQuantizedVariants =
    [
        "face_detection_yunet_2023mar.onnx",
        "face_detection_yunet_2023mar_int8.onnx",
        "face_detection_yunet_2023mar_int8bq.onnx"
    ];

    [Fact]
    public void NamingAFileAsksTheDownloaderForThatFile()
    {
        var preferences = ModelPreferences.ForTier(PerformanceTier.Medium);

        var withRequest = ModelPathResolver.WithRequestedFile(preferences, "face_detection_yunet_2023mar.onnx");

        withRequest.PreferredOnnxFiles.Should().Equal(["face_detection_yunet_2023mar.onnx"]);
    }

    [Fact]
    public void TheRestOfThePreferencesSurvive()
    {
        var preferences = ModelPreferences.ForTier(PerformanceTier.Low);

        var withRequest = ModelPathResolver.WithRequestedFile(preferences, "model.onnx");

        withRequest.PreferLowMemory.Should().Be(preferences.PreferLowMemory);
        withRequest.QuantizationPriority.Should().Equal(preferences.QuantizationPriority);
        withRequest.DecoderVariantPriority.Should().Equal(preferences.DecoderVariantPriority);
    }

    [Fact]
    public void AnExplicitFileListFromTheCallerIsLeftAlone()
    {
        var preferences = new ModelPreferences { PreferredOnnxFiles = ["encoder_model.onnx", "decoder_model.onnx"] };

        var withRequest = ModelPathResolver.WithRequestedFile(preferences, "model.onnx");

        withRequest.Should().BeSameAs(preferences);
    }

    [Theory]
    [InlineData(null)]
    [InlineData("")]
    [InlineData("   ")]
    public void NoNamedFileMeansNoChange(string? requested)
    {
        var preferences = ModelPreferences.ForTier(PerformanceTier.Medium);

        ModelPathResolver.WithRequestedFile(preferences, requested!).Should().BeSameAs(preferences);
    }

    [Theory]
    [InlineData("model.onnx")]
    [InlineData("onnx/model.onnx")]
    [InlineData("MODEL.ONNX")]
    public void ThePlaceholderNameStillLetsTheMachineChooseAPrecision(string requested)
    {
        // Both fallbacks that describe an arbitrary repository emit this name because they have no idea
        // what the repository calls its weights. Treating that guess as a decision would switch off variant
        // selection for every consumer that never named anything - a repository publishing model.onnx beside
        // model_quantized.onnx expects the machine to pick, and a low-memory machine that used to get the
        // small build would suddenly be handed the large one.
        var preferences = ModelPreferences.ForTier(PerformanceTier.Low);

        ModelPathResolver.WithRequestedFile(preferences, requested).Should().BeSameAs(preferences);
    }

    [Fact]
    public void ADistinctiveNameIsTreatedAsADecision()
    {
        var preferences = ModelPreferences.ForTier(PerformanceTier.Low);

        ModelPathResolver.WithRequestedFile(preferences, "rt-detrv2-s.onnx")
            .PreferredOnnxFiles.Should().Equal(["rt-detrv2-s.onnx"]);
    }

    [Fact]
    public void WithoutTheRequest_AMidTierMachineWouldTakeTheInt8Build()
    {
        // Not a defect in itself — this preference is right for a repository that publishes one model in
        // several precisions and lets the machine choose. It becomes one only when the caller already named
        // a file, which is the case this fix covers. Pinned because it is the mechanism, and because a
        // future edit to the tier table would otherwise quietly change which of the two cases applies.
        var preferences = ModelPreferences.ForTier(PerformanceTier.Medium);

        var candidates = RepoWithQuantizedVariants
            .Select(f => new RepoFile { Path = f, Type = "file", Size = 100_000L })
            .ToList();

        var selected = ModelDiscoveryService.SelectBestVariantsForTest(candidates, preferences);

        selected.Should().NotContain("face_detection_yunet_2023mar.onnx",
            "the plain weights are exactly the file the caller named, and without the request they are the one file not fetched");
        selected.Should().OnlyContain(f => f.Contains("int8", StringComparison.Ordinal));
    }
}
