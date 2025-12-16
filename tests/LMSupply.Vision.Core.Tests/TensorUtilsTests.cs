using FluentAssertions;
using LMSupply.Vision;

namespace LMSupply.Vision.Core.Tests;

public class TensorUtilsTests
{
    [Fact]
    public void CreateImageTensor_WithValidData_ShouldCreateCorrectShape()
    {
        // Arrange
        int width = 224;
        int height = 224;
        int channels = 3;
        var data = new float[channels * height * width];

        // Act
        var tensor = TensorUtils.CreateImageTensor(data, width, height, channels);

        // Assert
        tensor.Dimensions.ToArray().Should().Equal([1, channels, height, width]);
        tensor.Length.Should().Be(data.Length);
    }

    [Fact]
    public void CreateImageTensor_WithProfile_ShouldUseProfileDimensions()
    {
        // Arrange
        var profile = PreprocessProfile.ImageNet;
        var data = new float[3 * profile.Height * profile.Width];

        // Act
        var tensor = TensorUtils.CreateImageTensor(data, profile);

        // Assert
        tensor.Dimensions.ToArray().Should().Equal([1, 3, profile.Height, profile.Width]);
    }

    [Fact]
    public void CreateImageTensor_WithBatchSize_ShouldCreateBatchedTensor()
    {
        // Arrange
        int batchSize = 4;
        int width = 32;
        int height = 32;
        int channels = 3;
        var data = new float[batchSize * channels * height * width];

        // Act
        var tensor = TensorUtils.CreateImageTensor(data, width, height, channels, batchSize);

        // Assert
        tensor.Dimensions.ToArray().Should().Equal([batchSize, channels, height, width]);
    }

    [Fact]
    public void CreateImageTensor_WithWrongDataLength_ShouldThrow()
    {
        // Arrange
        var data = new float[100]; // Wrong size

        // Act
        var act = () => TensorUtils.CreateImageTensor(data, 224, 224);

        // Assert
        act.Should().Throw<ArgumentException>()
            .WithMessage("*Data length*does not match expected length*");
    }

    [Fact]
    public void CreateImageTensor_WithNullData_ShouldThrow()
    {
        // Act
        var act = () => TensorUtils.CreateImageTensor(null!, 224, 224);

        // Assert
        act.Should().Throw<ArgumentNullException>();
    }

    [Fact]
    public void CreateTokenTensor_FromIntArray_ShouldCreateCorrectShape()
    {
        // Arrange
        var tokenIds = new int[] { 1, 2, 3, 4, 5 };

        // Act
        var tensor = TensorUtils.CreateTokenTensor(tokenIds);

        // Assert
        tensor.Dimensions.ToArray().Should().Equal([1, 5]);
        tensor.Length.Should().Be(5);
    }

    [Fact]
    public void CreateTokenTensor_FromLongArray_ShouldCreateCorrectShape()
    {
        // Arrange
        var tokenIds = new long[] { 1, 2, 3, 4, 5 };

        // Act
        var tensor = TensorUtils.CreateTokenTensor(tokenIds);

        // Assert
        tensor.Dimensions.ToArray().Should().Equal([1, 5]);
    }

    [Fact]
    public void CreateTokenTensor_WithBatchSize_ShouldSetBatchDimension()
    {
        // Arrange - for batch processing, data must be provided for all batches
        // tokenIds length = seqLength (3) for single batch with batchSize metadata
        var tokenIds = new int[] { 1, 2, 3 };
        int batchSize = 1; // batchSize here represents the batch dimension value

        // Act
        var tensor = TensorUtils.CreateTokenTensor(tokenIds, batchSize);

        // Assert
        tensor.Dimensions.ToArray().Should().Equal([batchSize, 3]);
    }

    [Fact]
    public void CreateAttentionMask_ShouldCreateAllOnes()
    {
        // Arrange
        int seqLength = 10;

        // Act
        var tensor = TensorUtils.CreateAttentionMask(seqLength);

        // Assert
        tensor.Dimensions.ToArray().Should().Equal([1, seqLength]);
        tensor.ToArray().Should().AllSatisfy(v => v.Should().Be(1));
    }

    [Fact]
    public void CreateAttentionMask_WithBatchSize_ShouldWork()
    {
        // Arrange
        int seqLength = 5;
        int batchSize = 3;

        // Act
        var tensor = TensorUtils.CreateAttentionMask(seqLength, batchSize);

        // Assert
        tensor.Dimensions.ToArray().Should().Equal([batchSize, seqLength]);
        tensor.Length.Should().Be(batchSize * seqLength);
    }

    [Fact]
    public void ArgMax_ShouldReturnIndexOfMaxValue()
    {
        // Arrange
        var logits = new float[] { 0.1f, 0.5f, 0.3f, 0.9f, 0.2f };

        // Act
        var result = TensorUtils.ArgMax(logits);

        // Assert
        result.Should().Be(3);
    }

    [Fact]
    public void ArgMax_WithFirstElementMax_ShouldReturnZero()
    {
        // Arrange
        var logits = new float[] { 1.0f, 0.5f, 0.3f };

        // Act
        var result = TensorUtils.ArgMax(logits);

        // Assert
        result.Should().Be(0);
    }

    [Fact]
    public void ArgMax_WithLastElementMax_ShouldReturnLastIndex()
    {
        // Arrange
        var logits = new float[] { 0.1f, 0.5f, 0.3f, 0.9f };

        // Act
        var result = TensorUtils.ArgMax(logits);

        // Assert
        result.Should().Be(3);
    }

    [Fact]
    public void ArgMax_WithEmptySpan_ShouldThrow()
    {
        // Act
        var act = () => TensorUtils.ArgMax(ReadOnlySpan<float>.Empty);

        // Assert
        act.Should().Throw<ArgumentException>();
    }

    [Fact]
    public void ArgMax_WithNegativeValues_ShouldWork()
    {
        // Arrange
        var logits = new float[] { -5f, -2f, -10f, -1f };

        // Act
        var result = TensorUtils.ArgMax(logits);

        // Assert
        result.Should().Be(3); // -1 is the max
    }

    [Fact]
    public void Softmax_ShouldSumToOne()
    {
        // Arrange
        var logits = new float[] { 1.0f, 2.0f, 3.0f };

        // Act
        var probs = TensorUtils.Softmax(logits);

        // Assert
        probs.Sum().Should().BeApproximately(1.0f, 0.001f);
    }

    [Fact]
    public void Softmax_ShouldPreserveOrder()
    {
        // Arrange
        var logits = new float[] { 1.0f, 2.0f, 3.0f };

        // Act
        var probs = TensorUtils.Softmax(logits);

        // Assert
        probs[2].Should().BeGreaterThan(probs[1]);
        probs[1].Should().BeGreaterThan(probs[0]);
    }

    [Fact]
    public void Softmax_WithHighTemperature_ShouldFlattenDistribution()
    {
        // Arrange
        var logits = new float[] { 1.0f, 2.0f, 3.0f };

        // Act
        var normalProbs = TensorUtils.Softmax(logits, temperature: 1.0f);
        var highTempProbs = TensorUtils.Softmax(logits, temperature: 10.0f);

        // Assert - High temperature makes distribution more uniform
        var normalRange = normalProbs.Max() - normalProbs.Min();
        var highTempRange = highTempProbs.Max() - highTempProbs.Min();
        highTempRange.Should().BeLessThan(normalRange);
    }

    [Fact]
    public void Softmax_WithLowTemperature_ShouldSharpenDistribution()
    {
        // Arrange
        var logits = new float[] { 1.0f, 2.0f, 3.0f };

        // Act
        var normalProbs = TensorUtils.Softmax(logits, temperature: 1.0f);
        var lowTempProbs = TensorUtils.Softmax(logits, temperature: 0.1f);

        // Assert - Low temperature makes distribution more peaked
        lowTempProbs[2].Should().BeGreaterThan(normalProbs[2]);
    }

    [Fact]
    public void Softmax_WithEmptySpan_ShouldThrow()
    {
        // Act
        var act = () => TensorUtils.Softmax(ReadOnlySpan<float>.Empty);

        // Assert
        act.Should().Throw<ArgumentException>();
    }

    [Fact]
    public void SampleFromDistribution_ShouldReturnValidIndex()
    {
        // Arrange
        var probs = new float[] { 0.1f, 0.2f, 0.3f, 0.4f };

        // Act
        var result = TensorUtils.SampleFromDistribution(probs);

        // Assert
        result.Should().BeInRange(0, probs.Length - 1);
    }

    [Fact]
    public void SampleFromDistribution_WithDeterministicDistribution_ShouldReturnExpectedIndex()
    {
        // Arrange - All probability on one token
        var probs = new float[] { 0f, 0f, 1f, 0f };

        // Act
        var result = TensorUtils.SampleFromDistribution(probs);

        // Assert
        result.Should().Be(2);
    }

    [Fact]
    public void SampleFromDistribution_WithNullProbabilities_ShouldThrow()
    {
        // Act
        var act = () => TensorUtils.SampleFromDistribution(null!);

        // Assert
        act.Should().Throw<ArgumentNullException>();
    }

    [Fact]
    public void SampleFromDistribution_WithFixedSeed_ShouldBeReproducible()
    {
        // Arrange
        var probs = new float[] { 0.25f, 0.25f, 0.25f, 0.25f };
        var random1 = new Random(42);
        var random2 = new Random(42);

        // Act
        var results1 = Enumerable.Range(0, 100)
            .Select(_ => TensorUtils.SampleFromDistribution(probs, random1))
            .ToList();
        var results2 = Enumerable.Range(0, 100)
            .Select(_ => TensorUtils.SampleFromDistribution(probs, random2))
            .ToList();

        // Assert
        results1.Should().Equal(results2);
    }

    [Fact]
    public void CreateImageInput_ShouldCreateNamedOnnxValue()
    {
        // Arrange
        var profile = PreprocessProfile.ImageNet;
        var data = new float[3 * profile.Height * profile.Width];

        // Act
        var input = TensorUtils.CreateImageInput("pixel_values", data, profile);

        // Assert
        input.Name.Should().Be("pixel_values");
        input.Value.Should().NotBeNull();
    }
}
