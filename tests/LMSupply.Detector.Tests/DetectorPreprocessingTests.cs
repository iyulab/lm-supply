using AwesomeAssertions;
using LMSupply.Detector.Core;
using LMSupply.Detector.Models;
using Xunit;
using LMSupply.Vision;

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

    private static RgbImage SolidColour(int size = 4) => RgbImage.Filled(size, size, R, G, B);

    [Fact]
    public void ScaledRgb_PutsRedFirstAndOnlyDividesBy255()
    {
        var image = SolidColour();

        var tensor = OnnxDetectorModel.PreprocessImage(image, 4, 4, DetectorInputFormat.ScaledRgb);

        // Exact division, not the ImageNet shift that used to be applied unconditionally: the RT-DETR
        // reference preprocessing ships those statistics with normalisation switched off.
        tensor[0, 0, 0, 0].Should().BeApproximately(R / 255f, 1e-6f);
        tensor[0, 1, 0, 0].Should().BeApproximately(G / 255f, 1e-6f);
        tensor[0, 2, 0, 0].Should().BeApproximately(B / 255f, 1e-6f);
    }

    [Fact]
    public void RawBgr_PutsBlueFirstAndLeavesTheBytesAlone()
    {
        var image = SolidColour();

        var tensor = OnnxDetectorModel.PreprocessImage(image, 4, 4, DetectorInputFormat.RawBgr);

        tensor[0, 0, 0, 0].Should().Be(B, "YuNet was exported against OpenCV's default blob, which is BGR");
        tensor[0, 1, 0, 0].Should().Be(G);
        tensor[0, 2, 0, 0].Should().Be(R);
    }

    [Fact]
    public void TheTwoFormatsAreNotInterchangeable()
    {
        var rgbImage = SolidColour();
        var bgrImage = SolidColour();

        var rgb = OnnxDetectorModel.PreprocessImage(rgbImage, 4, 4, DetectorInputFormat.ScaledRgb);
        var bgr = OnnxDetectorModel.PreprocessImage(bgrImage, 4, 4, DetectorInputFormat.RawBgr);

        rgb[0, 0, 0, 0].Should().NotBe(bgr[0, 0, 0, 0]);
    }

    [Fact]
    public void EveryFloatFormatProducesAnNchwTensorOfTheRequestedSize()
    {
        // The int32 format is NHWC and built by its own method; it is pinned below.
        foreach (var format in Enum.GetValues<DetectorInputFormat>().Where(f => f != DetectorInputFormat.PaddedRgbInt32))
        {
            var image = SolidColour(9);

            var tensor = OnnxDetectorModel.PreprocessImage(image, 8, 8, format);

            tensor.Dimensions.ToArray().Should().Equal([1, 3, 8, 8]);
        }
    }

    [Fact]
    public void AnUndeclaredFormatIsRefusedRatherThanGuessed()
    {
        var image = SolidColour();

        var act = () => OnnxDetectorModel.PreprocessImage(image, 4, 4, (DetectorInputFormat)0);

        act.Should().Throw<NotSupportedException>();
    }

    [Fact]
    public void ANonSquareInputKeepsWidthAndHeightApart()
    {
        // The licence-plate model takes 320x240. A single square size could not describe it, and a tensor
        // built with the two swapped is not a formatting detail - the model reads its offsets against priors
        // derived from these numbers, so every box would land somewhere else.
        var image = SolidColour(64);

        var tensor = OnnxDetectorModel.PreprocessImage(image, 320, 240, DetectorInputFormat.RawBgr);

        tensor.Dimensions.ToArray().Should().Equal([1, 3, 240, 320]);
    }

    [Fact]
    public void PixelsLandAtTheRightOffsetInANonSquareTensor()
    {
        var image = new RgbImage(8, 4);
        image[7, 3] = (1, 2, 3);

        var tensor = OnnxDetectorModel.PreprocessImage(image, 8, 4, DetectorInputFormat.RawBgr);

        // Last pixel of the last row, on each of the three planes.
        tensor[0, 0, 3, 7].Should().Be(3);
        tensor[0, 1, 3, 7].Should().Be(2);
        tensor[0, 2, 3, 7].Should().Be(1);
    }

    [Fact]
    public void PaddedRgbInt32_IsNotAFloatFormat()
    {
        var image = SolidColour();

        var act = () => OnnxDetectorModel.PreprocessImage(image, 4, 4, DetectorInputFormat.PaddedRgbInt32);

        act.Should().Throw<NotSupportedException>();
    }

    [Fact]
    public void PaddedRgbInt32_IsNhwcRawRgb()
    {
        var image = SolidColour(8);

        var (tensor, _) = OnnxDetectorModel.PreprocessPaddedInt32(image, 8, 8);

        tensor.Dimensions.ToArray().Should().Equal([1, 8, 8, 3]);
        tensor[0, 0, 0, 0].Should().Be(R, "MoveNet takes RGB bytes, unscaled");
        tensor[0, 0, 0, 1].Should().Be(G);
        tensor[0, 0, 0, 2].Should().Be(B);
    }

    [Fact]
    public void PaddedRgbInt32_KeepsTheAspectRatio_AndCentresTheImageOnBlack()
    {
        // A 2:1 image on a square input fills the width and half the height, a quarter of black above and below.
        var image = RgbImage.Filled(16, 8, R, G, B);

        var (tensor, frame) = OnnxDetectorModel.PreprocessPaddedInt32(image, 8, 8);

        frame.Should().Be(new DetectorInputFrame(8, 8, 8, 4, 0, 2, 16, 8));
        for (var x = 0; x < 8; x++)
        {
            tensor[0, 0, x, 0].Should().Be(0, "row 0 is padding");
            tensor[0, 1, x, 0].Should().Be(0, "row 1 is padding");
            tensor[0, 2, x, 0].Should().Be(R, "row 2 is the first row of the image");
            tensor[0, 5, x, 0].Should().Be(R, "row 5 is the last row of the image");
            tensor[0, 6, x, 0].Should().Be(0, "row 6 is padding");
        }
    }

    [Fact]
    public void PaddedFrame_PutsTheOddPixelOfPaddingOnTheFarSide()
    {
        // 8x3 on 8x8: no resize, 5 rows of padding - 2 above, 3 below.
        DetectorInputFrame.Padded(8, 8, 8, 3).Should().Be(new DetectorInputFrame(8, 8, 8, 3, 0, 2, 8, 3));
        // 3x8 on 8x8: 5 columns of padding - 2 left, 3 right.
        DetectorInputFrame.Padded(8, 8, 3, 8).Should().Be(new DetectorInputFrame(8, 8, 3, 8, 2, 0, 3, 8));
        // 10x3 on 8x8: resized to 8x2 (2.4 rounds down), 6 rows of padding split evenly.
        DetectorInputFrame.Padded(8, 8, 10, 3).Should().Be(new DetectorInputFrame(8, 8, 8, 2, 0, 3, 10, 3));
    }
}
