using AwesomeAssertions;
using LMSupply.Detector.Core;
using LMSupply.Detector.Models;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using Xunit;

namespace LMSupply.Detector.Tests;

/// <summary>
/// Which bytes go into the input tensor, in which order.
///
/// <para>
/// This is worth pinning because getting it wrong is invisible. Preprocessing was fixed for every model —
/// RGB, scaled, then shifted by the ImageNet statistics — and a model wanting anything else does not fail
/// loudly when fed the wrong thing: it returns an empty result, which is indistinguishable from an image
/// containing nothing to find. Measured on YuNet with a street photograph: BGR finds seven faces, RGB finds
/// none, and neither raises anything.
/// </para>
/// </summary>
public class DetectorPreprocessingTests
{
    private const byte R = 10;
    private const byte G = 20;
    private const byte B = 30;

    private static Image<Rgb24> SolidColour(int size = 4) =>
        new(size, size, new Rgb24(R, G, B));

    [Fact]
    public void ScaledRgb_PutsRedFirstAndOnlyDividesBy255()
    {
        using var image = SolidColour();

        var tensor = OnnxDetectorModel.PreprocessImage(image, 4, DetectorInputFormat.ScaledRgb);

        // Exact division, not the ImageNet shift that used to be applied unconditionally: the RT-DETR
        // reference preprocessing ships those statistics with normalisation switched off.
        tensor[0, 0, 0, 0].Should().BeApproximately(R / 255f, 1e-6f);
        tensor[0, 1, 0, 0].Should().BeApproximately(G / 255f, 1e-6f);
        tensor[0, 2, 0, 0].Should().BeApproximately(B / 255f, 1e-6f);
    }

    [Fact]
    public void RawBgr_PutsBlueFirstAndLeavesTheBytesAlone()
    {
        using var image = SolidColour();

        var tensor = OnnxDetectorModel.PreprocessImage(image, 4, DetectorInputFormat.RawBgr);

        tensor[0, 0, 0, 0].Should().Be(B, "YuNet was exported against OpenCV's default blob, which is BGR");
        tensor[0, 1, 0, 0].Should().Be(G);
        tensor[0, 2, 0, 0].Should().Be(R);
    }

    [Fact]
    public void TheTwoFormatsAreNotInterchangeable()
    {
        using var rgbImage = SolidColour();
        using var bgrImage = SolidColour();

        var rgb = OnnxDetectorModel.PreprocessImage(rgbImage, 4, DetectorInputFormat.ScaledRgb);
        var bgr = OnnxDetectorModel.PreprocessImage(bgrImage, 4, DetectorInputFormat.RawBgr);

        rgb[0, 0, 0, 0].Should().NotBe(bgr[0, 0, 0, 0]);
    }

    [Fact]
    public void EveryFormatProducesAnNchwTensorOfTheRequestedSize()
    {
        foreach (var format in Enum.GetValues<DetectorInputFormat>())
        {
            using var image = SolidColour(9);

            var tensor = OnnxDetectorModel.PreprocessImage(image, 8, format);

            tensor.Dimensions.ToArray().Should().Equal([1, 3, 8, 8]);
        }
    }

    [Fact]
    public void AnUndeclaredFormatIsRefusedRatherThanGuessed()
    {
        using var image = SolidColour();

        var act = () => OnnxDetectorModel.PreprocessImage(image, 4, (DetectorInputFormat)0);

        act.Should().Throw<NotSupportedException>();
    }
}
