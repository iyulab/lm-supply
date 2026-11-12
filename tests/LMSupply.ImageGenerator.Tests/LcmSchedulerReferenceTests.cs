using AwesomeAssertions;
using LMSupply.ImageGenerator.Schedulers;

namespace LMSupply.ImageGenerator.Tests;

/// <summary>
/// Pins the scheduler to the values of the reference latent consistency scheduler (the original implementation with the
/// model's own configuration): its timesteps, its skip and output coefficients, and the noise it adds between steps.
/// </summary>
public sealed class LcmSchedulerReferenceTests
{
    public static TheoryData<int, int[]> ReferenceTimesteps => new()
    {
        { 1, [999] },
        { 2, [999, 499] },
        { 3, [999, 679, 339] },
        { 4, [999, 759, 499, 259] },
        { 5, [999, 799, 599, 399, 199] },
        { 8, [999, 879, 759, 639, 499, 379, 259, 139] },
    };

    [Theory]
    [MemberData(nameof(ReferenceTimesteps))]
    public void SetTimesteps_GivesTheReferenceSchedule(int steps, int[] expected)
    {
        var scheduler = new LcmScheduler();

        scheduler.SetTimesteps(steps);

        scheduler.Timesteps.ToArray().Should().Equal(expected);
    }

    [Fact]
    public void SetTimesteps_WithAllOriginalSteps_WalksEveryDistillationTimestep()
    {
        var scheduler = new LcmScheduler();

        scheduler.SetTimesteps(50);

        var timesteps = scheduler.Timesteps.ToArray();
        timesteps[0].Should().Be(999);
        timesteps[^1].Should().Be(19);
        timesteps.Zip(timesteps.Skip(1), (a, b) => a - b).Should().OnlyContain(gap => gap == 20);
    }

    [Fact]
    public void Step_OfTheOnlyStep_ReturnsTheDenoisedSampleOfTheReference()
    {
        // One step from t = 999: the consistency function's output is almost entirely the predicted clean sample.
        var scheduler = new LcmScheduler();
        scheduler.SetTimesteps(1);
        float[] sample = [0.5f, -1.0f, 2.0f];
        float[] noisePrediction = [0.1f, 0.2f, -0.3f];

        var result = scheduler.Step(noisePrediction, 999, sample);

        result.Should().HaveCount(3);
        float[] expected = [5.862943f, -17.571742f, 33.682019f];
        for (var i = 0; i < expected.Length; i++)
        {
            result[i].Should().BeApproximately(expected[i], 1e-3f * Math.Abs(expected[i]) + 1e-3f);
        }
    }

    [Fact]
    public void Step_BeforeTheLastStep_MixesTheDenoisedSampleWithFreshNoiseForTheNextTimestep()
    {
        var scheduler = new LcmScheduler();
        scheduler.SetTimesteps(2);
        float[] sample = [0.5f, -1.0f, 2.0f];
        float[] noisePrediction = [0.1f, 0.2f, -0.3f];
        float[] denoised = [5.862943f, -17.571742f, 33.682019f];
        var noise = LcmScheduler.CreateNoise([3], new Random(5));

        var result = scheduler.Step(noisePrediction, 999, sample, new Random(5));

        // The next timestep is 499, whose cumulative alpha is 0.27767 for this schedule.
        const float alphaNext = 0.27767f;
        var keep = MathF.Sqrt(alphaNext);
        var fresh = MathF.Sqrt(1.0f - alphaNext);
        for (var i = 0; i < denoised.Length; i++)
        {
            var expected = keep * denoised[i] + fresh * noise[i];
            result[i].Should().BeApproximately(expected, 5e-3f * Math.Max(1.0f, Math.Abs(expected)));
        }
    }
}
