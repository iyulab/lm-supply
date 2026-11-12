using AwesomeAssertions;
using LMSupply.Segmenter.Core;
using LMSupply.Segmenter.Models;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace LMSupply.Segmenter.Tests;

public sealed class SegmentationOutputParserTests
{
    [Fact]
    public void ClassLogits_GiveEachPixelItsHighestScoringClass()
    {
        // 3 classes over a 1x2 image: pixel 0 scores highest for class 2, pixel 1 for class 0.
        float[] values = [0f, 5f, 1f, 0f, 9f, 0f];
        var scores = new DenseTensor<float>(values, [1, 3, 1, 2]);

        var (classMap, confidence, width, height) = SegmentationOutputParser.Parse(SegmenterOutputKind.ClassLogits, scores, null);

        (width, height).Should().Be((2, 1));
        classMap.Should().Equal(2, 0);
        confidence[0].Should().BeApproximately(1f / (1f + MathF.Exp(-9f)), 1e-6f);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void ForegroundAlpha_IsClassOneAboveOneHalfAndTheProbabilityOfTheChosenClass(bool channelsLast)
    {
        float[] alpha = [0.9f, 0.1f, 0.6f, 0.4f];
        var tensor = channelsLast
            ? new DenseTensor<float>(alpha, [1, 2, 2, 1])
            : new DenseTensor<float>(alpha, [1, 1, 2, 2]);

        var (classMap, confidence, width, height) = SegmentationOutputParser.Parse(SegmenterOutputKind.ForegroundAlpha, tensor, null);

        (width, height).Should().Be((2, 2));
        classMap.Should().Equal(1, 0, 1, 0);
        float[] expected = [0.9f, 0.9f, 0.6f, 0.6f];
        for (var i = 0; i < expected.Length; i++)
        {
            confidence[i].Should().BeApproximately(expected[i], 1e-6f);
        }
    }

    [Fact]
    public void QueryMasks_GiveEachPixelTheClassOfTheQueryWhoseMaskCoversIt()
    {
        // Three queries over 3 classes (plus "no object" as the last class), a 1x4 image.
        // Query 0 says class 1 and covers the two left pixels; query 1 says class 2 and covers the two right pixels;
        // query 2 says "no object" and covers everything — it must not make any pixel anything.
        float[] classLogits =
        [
            -9f, 9f, -9f, -9f,   // query 0: class 1
            -9f, -9f, 9f, -9f,   // query 1: class 2
            -9f, -9f, -9f, 9f,   // query 2: no object
        ];
        float[] maskLogits =
        [
            9f, 9f, -9f, -9f,    // query 0
            -9f, -9f, 9f, 9f,    // query 1
            9f, 9f, 9f, 9f,      // query 2
        ];
        var classes = new DenseTensor<float>(classLogits, [1, 3, 4]);
        var masks = new DenseTensor<float>(maskLogits, [1, 3, 1, 4]);

        var (classMap, confidence, width, height) = SegmentationOutputParser.Parse(SegmenterOutputKind.QueryMasks, classes, masks);

        (width, height).Should().Be((4, 1));
        classMap.Should().Equal(1, 1, 2, 2);
        confidence.Should().OnlyContain(c => c > 0.99f && c <= 1f);
    }

    [Fact]
    public void QueryMasks_WithOnlyUndecidedQueries_GiveAnEmptyClassMapWithoutConfidence()
    {
        // Every class probability is below the threshold (a uniform spread over 200 classes plus "no object").
        var classes = new DenseTensor<float>(new float[201], [1, 1, 201]);
        float[] maskLogits = [5f, 5f];
        var masks = new DenseTensor<float>(maskLogits, [1, 1, 1, 2]);

        var (classMap, confidence, _, _) = SegmentationOutputParser.Parse(SegmenterOutputKind.QueryMasks, classes, masks);

        classMap.Should().Equal(0, 0);
        confidence.Should().OnlyContain(c => c == 0f);
    }

    [Fact]
    public void QueryMasks_WithoutTheMaskOutput_Throws()
    {
        var classes = new DenseTensor<float>(new float[4], [1, 1, 4]);

        var act = () => SegmentationOutputParser.Parse(SegmenterOutputKind.QueryMasks, classes, null);

        act.Should().Throw<InvalidOperationException>();
    }

    [Fact]
    public void Registry_DescribesEachModelByWhatItOutputsAndWhatItNames()
    {
        var registry = SegmenterModelRegistry.Default;

        var selfie = registry.Resolve("fast");
        selfie.OutputKind.Should().Be(SegmenterOutputKind.ForegroundAlpha);
        selfie.InputNormalization.Should().Be(SegmenterInputNormalization.UnitRange);
        selfie.ClassLabels.Should().Equal("background", "person");

        var maskFormer = registry.Resolve("quality");
        maskFormer.OutputKind.Should().Be(SegmenterOutputKind.QueryMasks);
        maskFormer.NumClasses.Should().Be(847);
        maskFormer.ClassLabels.Should().BeNull("its names come from the model's own configuration");

        var segFormer = registry.Resolve("default");
        segFormer.OutputKind.Should().Be(SegmenterOutputKind.ClassLogits);
        segFormer.InputNormalization.Should().Be(SegmenterInputNormalization.ImageNet);
    }
}
