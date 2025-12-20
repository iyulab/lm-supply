namespace LMSupply.ImageGenerator.Schedulers;

/// <summary>
/// Configuration for the LCM scheduler.
/// </summary>
public sealed class LcmSchedulerConfig
{
    /// <summary>
    /// Number of training timesteps used for the scheduler.
    /// Default: 1000
    /// </summary>
    public int NumTrainTimesteps { get; set; } = 1000;

    /// <summary>
    /// Starting beta value for the noise schedule.
    /// Default: 0.00085 (Stable Diffusion default)
    /// </summary>
    public float BetaStart { get; set; } = 0.00085f;

    /// <summary>
    /// Ending beta value for the noise schedule.
    /// Default: 0.012 (Stable Diffusion default)
    /// </summary>
    public float BetaEnd { get; set; } = 0.012f;

    /// <summary>
    /// Beta schedule type.
    /// Default: ScaledLinear (recommended for Stable Diffusion)
    /// </summary>
    public BetaSchedule BetaSchedule { get; set; } = BetaSchedule.ScaledLinear;

    /// <summary>
    /// Number of inference steps the model was originally trained for.
    /// Default: 50
    /// </summary>
    public int OriginalInferenceSteps { get; set; } = 50;

    /// <summary>
    /// Whether to clip the predicted sample.
    /// Default: false
    /// </summary>
    public bool ClipSample { get; set; } = false;

    /// <summary>
    /// Range for sample clipping.
    /// Default: 1.0
    /// </summary>
    public float ClipSampleRange { get; set; } = 1.0f;

    /// <summary>
    /// Whether to set the final alpha to 1.0.
    /// Default: true
    /// </summary>
    public bool SetAlphaToOne { get; set; } = true;

    /// <summary>
    /// Offset added to the computed timesteps.
    /// Default: 0
    /// </summary>
    public int StepsOffset { get; set; } = 0;

    /// <summary>
    /// Type of prediction the model makes.
    /// Default: Epsilon (noise prediction)
    /// </summary>
    public PredictionType PredictionType { get; set; } = PredictionType.Epsilon;

    /// <summary>
    /// Whether to apply dynamic thresholding.
    /// Default: false
    /// </summary>
    public bool Thresholding { get; set; } = false;

    /// <summary>
    /// Ratio for dynamic thresholding percentile calculation.
    /// Default: 0.995
    /// </summary>
    public float DynamicThresholdingRatio { get; set; } = 0.995f;

    /// <summary>
    /// Maximum sample value after dynamic thresholding.
    /// Default: 1.0
    /// </summary>
    public float SampleMaxValue { get; set; } = 1.0f;

    /// <summary>
    /// Creates default configuration for LCM-based models.
    /// </summary>
    public static LcmSchedulerConfig ForLcm() => new()
    {
        NumTrainTimesteps = 1000,
        BetaStart = 0.00085f,
        BetaEnd = 0.012f,
        BetaSchedule = BetaSchedule.ScaledLinear,
        OriginalInferenceSteps = 50,
        SetAlphaToOne = true,
        PredictionType = PredictionType.Epsilon
    };
}

/// <summary>
/// Beta schedule types for noise scheduling.
/// </summary>
public enum BetaSchedule
{
    /// <summary>
    /// Linear interpolation between start and end.
    /// </summary>
    Linear,

    /// <summary>
    /// Scaled linear schedule (sqrt interpolation, then squared).
    /// Used by Stable Diffusion.
    /// </summary>
    ScaledLinear,

    /// <summary>
    /// Squared cosine schedule from Nichol et al.
    /// </summary>
    SquaredCosCapV2
}

/// <summary>
/// Types of predictions the model can make.
/// </summary>
public enum PredictionType
{
    /// <summary>
    /// Model predicts the noise (epsilon).
    /// </summary>
    Epsilon,

    /// <summary>
    /// Model predicts the denoised sample directly.
    /// </summary>
    Sample,

    /// <summary>
    /// Model predicts the velocity (v-prediction).
    /// </summary>
    VPrediction
}
