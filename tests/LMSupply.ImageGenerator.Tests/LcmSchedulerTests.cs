using FluentAssertions;
using LMSupply.ImageGenerator.Schedulers;

namespace LMSupply.ImageGenerator.Tests;

public class LcmSchedulerTests
{
    [Fact]
    public void SetTimesteps_WithValidSteps_SetsTimesteps()
    {
        // Arrange
        var scheduler = new LcmScheduler();

        // Act
        scheduler.SetTimesteps(4);

        // Assert
        scheduler.Timesteps.Length.Should().Be(4);
        scheduler.StepIndex.Should().Be(0);
    }

    [Fact]
    public void SetTimesteps_WithOneStep_SetsTimesteps()
    {
        // Arrange
        var scheduler = new LcmScheduler();

        // Act
        scheduler.SetTimesteps(1);

        // Assert
        scheduler.Timesteps.Length.Should().Be(1);
    }

    [Fact]
    public void SetTimesteps_WithMaxSteps_SetsTimesteps()
    {
        // Arrange
        var config = LcmSchedulerConfig.ForLcm();
        var scheduler = new LcmScheduler(config);

        // Act
        scheduler.SetTimesteps(config.OriginalInferenceSteps);

        // Assert
        scheduler.Timesteps.Length.Should().Be(config.OriginalInferenceSteps);
    }

    [Fact]
    public void SetTimesteps_WithZeroSteps_ThrowsException()
    {
        // Arrange
        var scheduler = new LcmScheduler();

        // Act & Assert
        var act = () => scheduler.SetTimesteps(0);
        act.Should().Throw<ArgumentOutOfRangeException>();
    }

    [Fact]
    public void SetTimesteps_ExceedingOriginalSteps_ThrowsException()
    {
        // Arrange
        var config = LcmSchedulerConfig.ForLcm();
        var scheduler = new LcmScheduler(config);

        // Act & Assert
        var act = () => scheduler.SetTimesteps(config.OriginalInferenceSteps + 1);
        act.Should().Throw<ArgumentOutOfRangeException>();
    }

    [Fact]
    public void Timesteps_WithoutSetTimesteps_ThrowsException()
    {
        // Arrange
        var scheduler = new LcmScheduler();

        // Act & Assert
        Action act = () => _ = scheduler.Timesteps;
        act.Should().Throw<InvalidOperationException>();
    }

    [Fact]
    public void Timesteps_AreDescending()
    {
        // Arrange
        var scheduler = new LcmScheduler();
        scheduler.SetTimesteps(4);

        // Act
        var timesteps = scheduler.Timesteps.ToArray();

        // Assert - timesteps should start high and decrease
        for (int i = 0; i < timesteps.Length - 1; i++)
        {
            timesteps[i].Should().BeGreaterThan(timesteps[i + 1]);
        }
    }

    [Fact]
    public void CreateNoise_CreatesCorrectShape()
    {
        // Arrange
        var shape = new[] { 1, 4, 64, 64 };
        var expectedLength = 1 * 4 * 64 * 64;

        // Act
        var noise = LcmScheduler.CreateNoise(shape);

        // Assert
        noise.Length.Should().Be(expectedLength);
    }

    [Fact]
    public void CreateNoise_WithSeed_IsReproducible()
    {
        // Arrange
        var shape = new[] { 1, 4, 64, 64 };
        var random1 = new Random(42);
        var random2 = new Random(42);

        // Act
        var noise1 = LcmScheduler.CreateNoise(shape, random1);
        var noise2 = LcmScheduler.CreateNoise(shape, random2);

        // Assert
        noise1.Should().BeEquivalentTo(noise2);
    }

    [Fact]
    public void CreateNoise_HasApproximatelyZeroMean()
    {
        // Arrange
        var shape = new[] { 1, 4, 64, 64 };

        // Act
        var noise = LcmScheduler.CreateNoise(shape, new Random(42));
        var mean = (double)noise.Average();

        // Assert - Gaussian noise should have mean close to 0
        mean.Should().BeApproximately(0, 0.1);
    }

    [Fact]
    public void CreateNoise_HasApproximatelyUnitVariance()
    {
        // Arrange
        var shape = new[] { 1, 4, 64, 64 };

        // Act
        var noise = LcmScheduler.CreateNoise(shape, new Random(42));
        var mean = noise.Average();
        var variance = (double)noise.Select(x => (x - mean) * (x - mean)).Average();

        // Assert - Gaussian noise should have variance close to 1
        variance.Should().BeApproximately(1, 0.1);
    }

    [Fact]
    public void Step_UpdatesStepIndex()
    {
        // Arrange
        var scheduler = new LcmScheduler();
        scheduler.SetTimesteps(4);
        var sampleSize = 1 * 4 * 64 * 64;
        var sample = new float[sampleSize];
        var modelOutput = new float[sampleSize];

        // Act
        var timestep = scheduler.Timesteps[0];
        scheduler.Step(modelOutput, timestep, sample);

        // Assert
        scheduler.StepIndex.Should().Be(1);
    }

    [Fact]
    public void Step_ReturnsCorrectShape()
    {
        // Arrange
        var scheduler = new LcmScheduler();
        scheduler.SetTimesteps(4);
        var sampleSize = 1 * 4 * 64 * 64;
        var sample = new float[sampleSize];
        var modelOutput = new float[sampleSize];

        // Act
        var timestep = scheduler.Timesteps[0];
        var result = scheduler.Step(modelOutput, timestep, sample);

        // Assert
        result.Length.Should().Be(sampleSize);
    }

    [Fact]
    public void ScaleModelInput_ReturnsCorrectShape()
    {
        // Arrange
        var scheduler = new LcmScheduler();
        scheduler.SetTimesteps(4);
        var sampleSize = 1 * 4 * 64 * 64;
        var sample = new float[sampleSize];

        // Act
        var result = scheduler.ScaleModelInput(sample, scheduler.Timesteps[0]);

        // Assert
        result.Length.Should().Be(sampleSize);
    }
}
