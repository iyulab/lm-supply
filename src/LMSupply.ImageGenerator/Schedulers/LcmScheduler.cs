using System.Numerics.Tensors;

namespace LMSupply.ImageGenerator.Schedulers;

/// <summary>
/// Latent Consistency Model scheduler for fast image generation with 2-8 steps.
/// Implements Algorithm 3 from the LCM paper.
/// </summary>
public sealed class LcmScheduler
{
    private readonly LcmSchedulerConfig _config;
    private readonly float[] _alphasCumprod;
    private readonly float _finalAlphaCumprod;

    private int[]? _timesteps;
    private int _numInferenceSteps;
    private int _stepIndex;

    /// <summary>
    /// Creates a new LCM scheduler with the specified configuration.
    /// </summary>
    public LcmScheduler(LcmSchedulerConfig? config = null)
    {
        _config = config ?? new LcmSchedulerConfig();

        // Compute betas
        var betas = ComputeBetas(_config);

        // Compute alphas = 1 - betas
        var alphas = new float[betas.Length];
        for (int i = 0; i < betas.Length; i++)
        {
            alphas[i] = 1.0f - betas[i];
        }

        // Compute cumulative product of alphas
        _alphasCumprod = new float[alphas.Length];
        _alphasCumprod[0] = alphas[0];
        for (int i = 1; i < alphas.Length; i++)
        {
            _alphasCumprod[i] = _alphasCumprod[i - 1] * alphas[i];
        }

        _finalAlphaCumprod = _config.SetAlphaToOne ? 1.0f : _alphasCumprod[0];
    }

    /// <summary>
    /// Gets the current step index.
    /// </summary>
    public int StepIndex => _stepIndex;

    /// <summary>
    /// Gets the configured timesteps.
    /// </summary>
    public ReadOnlySpan<int> Timesteps => _timesteps ?? throw new InvalidOperationException("Call SetTimesteps first");

    /// <summary>
    /// Configures the scheduler for the given number of inference steps.
    /// </summary>
    /// <param name="numInferenceSteps">Number of denoising steps (typically 2-8 for LCM).</param>
    public void SetTimesteps(int numInferenceSteps)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(numInferenceSteps, 1);
        ArgumentOutOfRangeException.ThrowIfGreaterThan(numInferenceSteps, _config.OriginalInferenceSteps);

        _numInferenceSteps = numInferenceSteps;
        _stepIndex = 0;

        // LCM uses specific timestep spacing based on original training steps
        // Formula: timesteps at evenly spaced intervals from training schedule
        var originalSteps = _config.OriginalInferenceSteps;
        var trainTimesteps = _config.NumTrainTimesteps;

        // Calculate step ratio and timesteps
        var lcmOriginSteps = Math.Min(originalSteps, trainTimesteps);
        var stepRatio = trainTimesteps / lcmOriginSteps;

        // Calculate evenly spaced timesteps
        var allTimesteps = new int[lcmOriginSteps];
        for (int i = 0; i < lcmOriginSteps; i++)
        {
            allTimesteps[i] = (int)((lcmOriginSteps - 1 - i) * stepRatio) + _config.StepsOffset;
        }

        // Select subset of timesteps for actual inference
        var skippingStep = lcmOriginSteps / numInferenceSteps;
        _timesteps = new int[numInferenceSteps];
        for (int i = 0; i < numInferenceSteps; i++)
        {
            var idx = Math.Min(lcmOriginSteps - 1 - (i * skippingStep), lcmOriginSteps - 1);
            _timesteps[numInferenceSteps - 1 - i] = allTimesteps[idx];
        }

        // Reverse to go from high noise to low noise
        Array.Reverse(_timesteps);
    }

    /// <summary>
    /// Performs a single denoising step.
    /// </summary>
    /// <param name="modelOutput">The output from the UNet model (noise prediction).</param>
    /// <param name="timestep">Current timestep.</param>
    /// <param name="sample">Current noisy latent sample.</param>
    /// <param name="random">Random number generator for noise injection.</param>
    /// <returns>Denoised sample for the next step.</returns>
    public float[] Step(
        ReadOnlySpan<float> modelOutput,
        int timestep,
        ReadOnlySpan<float> sample,
        Random? random = null)
    {
        if (_timesteps is null)
            throw new InvalidOperationException("Call SetTimesteps first");

        random ??= Random.Shared;

        // Get alpha values
        var alphaT = _alphasCumprod[timestep];
        var alphaTNext = _stepIndex < _numInferenceSteps - 1
            ? _alphasCumprod[_timesteps[_stepIndex + 1]]
            : _finalAlphaCumprod;

        var betaT = 1.0f - alphaT;

        // Compute predicted original sample from noise prediction
        // x0 = (sample - sqrt(1-alpha) * noise) / sqrt(alpha)
        var sqrtAlphaT = MathF.Sqrt(alphaT);
        var sqrtBetaT = MathF.Sqrt(betaT);

        var predictedOriginal = new float[sample.Length];
        for (int i = 0; i < sample.Length; i++)
        {
            predictedOriginal[i] = (sample[i] - sqrtBetaT * modelOutput[i]) / sqrtAlphaT;
        }

        // Apply clipping if configured
        if (_config.ClipSample)
        {
            var clipRange = _config.ClipSampleRange;
            for (int i = 0; i < predictedOriginal.Length; i++)
            {
                predictedOriginal[i] = Math.Clamp(predictedOriginal[i], -clipRange, clipRange);
            }
        }

        // Apply thresholding if configured
        if (_config.Thresholding)
        {
            ApplyDynamicThresholding(predictedOriginal);
        }

        // LCM denoising: apply skip/out coefficients
        // c_skip = sigma_data^2 / (scaled_t^2 + sigma_data^2)
        // c_out = scaled_t * sigma_data / sqrt(scaled_t^2 + sigma_data^2)
        var sigmaData = 0.5f; // Standard value for LCM
        var scaledT = GetScaledTimestep(timestep);

        var cSkip = (sigmaData * sigmaData) / (scaledT * scaledT + sigmaData * sigmaData);
        var cOut = scaledT * sigmaData / MathF.Sqrt(scaledT * scaledT + sigmaData * sigmaData);

        var denoised = new float[sample.Length];
        for (int i = 0; i < sample.Length; i++)
        {
            denoised[i] = cOut * predictedOriginal[i] + cSkip * sample[i];
        }

        // Compute previous sample
        var prevSample = new float[sample.Length];

        if (_stepIndex < _numInferenceSteps - 1)
        {
            // Not final step: add noise for next iteration
            var sqrtAlphaTNext = MathF.Sqrt(alphaTNext);
            var sqrtBetaTNext = MathF.Sqrt(1.0f - alphaTNext);

            // Sample gaussian noise
            var noise = SampleNoise(sample.Length, random);

            for (int i = 0; i < sample.Length; i++)
            {
                prevSample[i] = sqrtAlphaTNext * denoised[i] + sqrtBetaTNext * noise[i];
            }
        }
        else
        {
            // Final step: just return denoised
            Array.Copy(denoised, prevSample, sample.Length);
        }

        _stepIndex++;
        return prevSample;
    }

    /// <summary>
    /// Scales the model input based on the current timestep.
    /// </summary>
    public float[] ScaleModelInput(ReadOnlySpan<float> sample, int timestep)
    {
        // LCM doesn't require input scaling
        var result = new float[sample.Length];
        sample.CopyTo(result);
        return result;
    }

    /// <summary>
    /// Generates initial random noise for the latent space.
    /// </summary>
    /// <param name="shape">Shape of the latent tensor [batch, channels, height, width].</param>
    /// <param name="random">Random number generator.</param>
    /// <returns>Random noise tensor.</returns>
    public static float[] CreateNoise(int[] shape, Random? random = null)
    {
        random ??= Random.Shared;
        var length = shape.Aggregate(1, (a, b) => a * b);
        return SampleNoise(length, random);
    }

    private static float[] SampleNoise(int length, Random random)
    {
        var noise = new float[length];

        // Box-Muller transform for gaussian noise
        for (int i = 0; i < length; i += 2)
        {
            var u1 = random.NextDouble();
            var u2 = random.NextDouble();

            // Avoid log(0)
            if (u1 < 1e-10) u1 = 1e-10;

            var r = MathF.Sqrt(-2.0f * MathF.Log((float)u1));
            var theta = 2.0f * MathF.PI * (float)u2;

            noise[i] = r * MathF.Cos(theta);
            if (i + 1 < length)
            {
                noise[i + 1] = r * MathF.Sin(theta);
            }
        }

        return noise;
    }

    private float GetScaledTimestep(int timestep)
    {
        // Convert timestep to sigma-like scale
        var alpha = _alphasCumprod[timestep];
        return MathF.Sqrt((1.0f - alpha) / alpha);
    }

    private void ApplyDynamicThresholding(float[] sample)
    {
        // Dynamic thresholding from Imagen
        var ratio = _config.DynamicThresholdingRatio;
        var maxValue = _config.SampleMaxValue;

        // Compute percentile
        var sorted = sample.OrderBy(x => MathF.Abs(x)).ToArray();
        var percentileIdx = (int)(ratio * sorted.Length);
        var threshold = MathF.Max(sorted[percentileIdx], maxValue);

        // Clip and rescale
        for (int i = 0; i < sample.Length; i++)
        {
            sample[i] = Math.Clamp(sample[i], -threshold, threshold) / threshold * maxValue;
        }
    }

    private static float[] ComputeBetas(LcmSchedulerConfig config)
    {
        var numSteps = config.NumTrainTimesteps;
        var betas = new float[numSteps];

        switch (config.BetaSchedule)
        {
            case BetaSchedule.Linear:
                for (int i = 0; i < numSteps; i++)
                {
                    betas[i] = config.BetaStart + (config.BetaEnd - config.BetaStart) * i / (numSteps - 1);
                }
                break;

            case BetaSchedule.ScaledLinear:
                // Scaled linear schedule used by Stable Diffusion
                var sqrtStart = MathF.Sqrt(config.BetaStart);
                var sqrtEnd = MathF.Sqrt(config.BetaEnd);
                for (int i = 0; i < numSteps; i++)
                {
                    var sqrtBeta = sqrtStart + (sqrtEnd - sqrtStart) * i / (numSteps - 1);
                    betas[i] = sqrtBeta * sqrtBeta;
                }
                break;

            case BetaSchedule.SquaredCosCapV2:
                // Cosine schedule from Nichol et al.
                for (int i = 0; i < numSteps; i++)
                {
                    var t = (float)i / numSteps;
                    var alpha = MathF.Cos((t + 0.008f) / 1.008f * MathF.PI / 2);
                    betas[i] = 1.0f - (alpha * alpha);
                    betas[i] = Math.Clamp(betas[i], 0.0f, 0.999f);
                }
                break;

            default:
                throw new ArgumentException($"Unknown beta schedule: {config.BetaSchedule}");
        }

        return betas;
    }
}
