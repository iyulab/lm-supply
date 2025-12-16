using FluentAssertions;
using LMSupply.Segmenter.Interactive;
using LMSupply.Segmenter.Models;
using LMSupply.Vision;

namespace LMSupply.Segmenter.Tests;

public class InteractiveSegmentationTests
{
    #region PointPrompt Tests

    [Theory]
    [InlineData(100f, 200f, PointLabel.Foreground)]
    [InlineData(0f, 0f, PointLabel.Background)]
    [InlineData(512.5f, 512.5f, PointLabel.Foreground)]
    public void PointPrompt_ShouldStoreCoordinatesAndLabel(float x, float y, PointLabel label)
    {
        var prompt = new PointPrompt(x, y, label);

        prompt.X.Should().Be(x);
        prompt.Y.Should().Be(y);
        prompt.Label.Should().Be(label);
    }

    [Fact]
    public void PointPrompt_DefaultLabel_ShouldBeForeground()
    {
        var prompt = new PointPrompt(100, 200);

        prompt.Label.Should().Be(PointLabel.Foreground);
    }

    [Fact]
    public void PointPrompt_Type_ShouldBePoint()
    {
        var prompt = new PointPrompt(100, 200);

        prompt.Type.Should().Be(SamPromptType.Point);
    }

    [Fact]
    public void PointPrompt_Foreground_ShouldCreateForegroundPrompt()
    {
        var prompt = PointPrompt.Foreground(100, 200);

        prompt.Label.Should().Be(PointLabel.Foreground);
        prompt.X.Should().Be(100);
        prompt.Y.Should().Be(200);
    }

    [Fact]
    public void PointPrompt_Background_ShouldCreateBackgroundPrompt()
    {
        var prompt = PointPrompt.Background(100, 200);

        prompt.Label.Should().Be(PointLabel.Background);
    }

    #endregion

    #region BoxPrompt Tests

    [Theory]
    [InlineData(0f, 0f, 100f, 100f)]
    [InlineData(50.5f, 50.5f, 200f, 150f)]
    [InlineData(100f, 200f, 50f, 50f)]
    public void BoxPrompt_ShouldStoreCoordinatesAndSize(float x, float y, float width, float height)
    {
        var prompt = new BoxPrompt(x, y, width, height);

        prompt.X.Should().Be(x);
        prompt.Y.Should().Be(y);
        prompt.Width.Should().Be(width);
        prompt.Height.Should().Be(height);
    }

    [Fact]
    public void BoxPrompt_Right_ShouldCalculateCorrectly()
    {
        var prompt = new BoxPrompt(100, 50, 200, 150);

        prompt.Right.Should().Be(300);
    }

    [Fact]
    public void BoxPrompt_Bottom_ShouldCalculateCorrectly()
    {
        var prompt = new BoxPrompt(100, 50, 200, 150);

        prompt.Bottom.Should().Be(200);
    }

    [Fact]
    public void BoxPrompt_Type_ShouldBeBox()
    {
        var prompt = new BoxPrompt(0, 0, 100, 100);

        prompt.Type.Should().Be(SamPromptType.Box);
    }

    [Fact]
    public void BoxPrompt_FromCorners_ShouldCreateCorrectBox()
    {
        var prompt = BoxPrompt.FromCorners(10, 20, 110, 170);

        prompt.X.Should().Be(10);
        prompt.Y.Should().Be(20);
        prompt.Width.Should().Be(100);
        prompt.Height.Should().Be(150);
    }

    [Fact]
    public void BoxPrompt_FromCorners_WithReversedCoords_ShouldNormalize()
    {
        var prompt = BoxPrompt.FromCorners(110, 170, 10, 20);

        prompt.X.Should().Be(10);
        prompt.Y.Should().Be(20);
        prompt.Width.Should().Be(100);
        prompt.Height.Should().Be(150);
    }

    #endregion

    #region MaskPrompt Tests

    [Fact]
    public void MaskPrompt_ShouldStoreMaskData()
    {
        var mask = new bool[100, 100];
        mask[50, 50] = true;

        var prompt = new MaskPrompt(mask);

        prompt.Mask.Should().BeSameAs(mask);
        prompt.Mask.GetLength(0).Should().Be(100);
        prompt.Mask.GetLength(1).Should().Be(100);
    }

    [Fact]
    public void MaskPrompt_Type_ShouldBeMask()
    {
        var mask = new bool[10, 10];
        var prompt = new MaskPrompt(mask);

        prompt.Type.Should().Be(SamPromptType.Mask);
    }

    [Fact]
    public void MaskPrompt_NullMask_ShouldThrow()
    {
        var act = () => new MaskPrompt(null!);

        act.Should().Throw<ArgumentNullException>();
    }

    #endregion

    #region InteractiveSegmentationResult Tests

    [Fact]
    public void InteractiveSegmentationResult_BestMask_ShouldReturnFirstMask()
    {
        var masks = new List<SegmentationMask>
        {
            CreateTestMask(100, 100),
            CreateTestMask(100, 100),
            CreateTestMask(100, 100)
        };
        var scores = new List<float> { 0.95f, 0.85f, 0.75f };

        var result = new InteractiveSegmentationResult
        {
            Masks = masks,
            IoUScores = scores,
            ImageWidth = 512,
            ImageHeight = 512,
            InferenceTimeMs = 100
        };

        result.BestMask.Should().BeSameAs(masks[0]);
        result.BestScore.Should().Be(0.95f);
    }

    [Fact]
    public void InteractiveSegmentationResult_EmptyMasks_ShouldThrowOnBestMask()
    {
        var result = new InteractiveSegmentationResult
        {
            Masks = new List<SegmentationMask>(),
            IoUScores = new List<float>(),
            ImageWidth = 512,
            ImageHeight = 512,
            InferenceTimeMs = 100
        };

        var act = () => result.BestMask;

        act.Should().Throw<InvalidOperationException>();
    }

    [Fact]
    public void InteractiveSegmentationResult_EmptyScores_ShouldReturnZeroBestScore()
    {
        var result = new InteractiveSegmentationResult
        {
            Masks = new List<SegmentationMask>(),
            IoUScores = new List<float>(),
            ImageWidth = 512,
            ImageHeight = 512
        };

        result.BestScore.Should().Be(0);
    }

    [Fact]
    public void InteractiveSegmentationResult_ShouldStoreImageDimensions()
    {
        var result = new InteractiveSegmentationResult
        {
            Masks = new List<SegmentationMask>(),
            IoUScores = new List<float>(),
            ImageWidth = 1920,
            ImageHeight = 1080
        };

        result.ImageWidth.Should().Be(1920);
        result.ImageHeight.Should().Be(1080);
    }

    #endregion

    #region SegmenterModelInfo Interactive Properties Tests

    [Fact]
    public void SegmenterModelInfo_WithEncoderDecoder_ShouldBeInteractive()
    {
        var model = new SegmenterModelInfo
        {
            Id = "test/model",
            Alias = "test",
            DisplayName = "Test Model",
            Architecture = "SAM",
            EncoderFile = "encoder.onnx",
            DecoderFile = "decoder.onnx"
        };

        model.IsInteractive.Should().BeTrue();
    }

    [Fact]
    public void SegmenterModelInfo_WithOnlyEncoder_ShouldNotBeInteractive()
    {
        var model = new SegmenterModelInfo
        {
            Id = "test/model",
            Alias = "test",
            DisplayName = "Test Model",
            Architecture = "Test",
            EncoderFile = "encoder.onnx"
        };

        model.IsInteractive.Should().BeFalse();
    }

    [Fact]
    public void SegmenterModelInfo_WithOnlyDecoder_ShouldNotBeInteractive()
    {
        var model = new SegmenterModelInfo
        {
            Id = "test/model",
            Alias = "test",
            DisplayName = "Test Model",
            Architecture = "Test",
            DecoderFile = "decoder.onnx"
        };

        model.IsInteractive.Should().BeFalse();
    }

    [Fact]
    public void SegmenterModelInfo_WithoutEncoderDecoder_ShouldNotBeInteractive()
    {
        var model = new SegmenterModelInfo
        {
            Id = "test/model",
            Alias = "test",
            DisplayName = "Test Model",
            Architecture = "Test"
        };

        model.IsInteractive.Should().BeFalse();
    }

    #endregion

    #region DefaultModels MobileSAM Tests

    [Fact]
    public void MobileSAM_ShouldBeInteractive()
    {
        DefaultModels.MobileSAM.IsInteractive.Should().BeTrue();
    }

    [Fact]
    public void MobileSAM_ShouldHaveCorrectInputSize()
    {
        DefaultModels.MobileSAM.InputSize.Should().Be(1024);
    }

    [Fact]
    public void MobileSAM_ShouldHaveBinarySegmentation()
    {
        DefaultModels.MobileSAM.NumClasses.Should().Be(1);
    }

    [Fact]
    public void MobileSAM_ShouldHaveSA1BDataset()
    {
        DefaultModels.MobileSAM.Dataset.Should().Be("SA-1B");
    }

    [Fact]
    public void SegFormerModels_ShouldNotBeInteractive()
    {
        DefaultModels.SegFormerB0.IsInteractive.Should().BeFalse();
        DefaultModels.SegFormerB1.IsInteractive.Should().BeFalse();
        DefaultModels.SegFormerB2.IsInteractive.Should().BeFalse();
        DefaultModels.SegFormerB5.IsInteractive.Should().BeFalse();
    }

    #endregion

    #region SamPromptType Enum Tests

    [Fact]
    public void SamPromptType_ShouldHaveCorrectValues()
    {
        SamPromptType.Point.Should().Be(SamPromptType.Point);
        SamPromptType.Box.Should().Be(SamPromptType.Box);
        SamPromptType.Mask.Should().Be(SamPromptType.Mask);
    }

    #endregion

    #region PointLabel Enum Tests

    [Fact]
    public void PointLabel_ShouldHaveCorrectValues()
    {
        ((int)PointLabel.Background).Should().Be(0);
        ((int)PointLabel.Foreground).Should().Be(1);
    }

    #endregion

    #region Helper Methods

    private static SegmentationMask CreateTestMask(int width, int height)
    {
        var mask = new bool[height, width];
        // Set some pixels to true
        for (int y = height / 4; y < height * 3 / 4; y++)
        {
            for (int x = width / 4; x < width * 3 / 4; x++)
            {
                mask[y, x] = true;
            }
        }
        return new SegmentationMask(mask, 1);
    }

    #endregion
}
