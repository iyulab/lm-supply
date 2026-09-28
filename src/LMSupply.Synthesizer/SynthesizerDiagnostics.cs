namespace LMSupply.Synthesizer;

/// <summary>
/// Diagnostic ids of the Synthesizer package.
/// </summary>
public static class SynthesizerDiagnostics
{
    /// <summary>
    /// Reported on every use of <see cref="LocalSynthesizer"/> and <see cref="ISynthesizerModel"/>: the synthesizer has no
    /// text-to-phoneme step yet, so its output is not intelligible speech. Suppress it (<c>&lt;NoWarn&gt;LMSUPPLY001&lt;/NoWarn&gt;</c>
    /// or <c>#pragma warning disable LMSUPPLY001</c>) to use the package anyway — loading, voice selection and the audio API work.
    /// </summary>
    public const string ExperimentalId = "LMSUPPLY001";

    /// <summary>Where the diagnostic points: the package README's known issue.</summary>
    public const string ExperimentalUrl = "https://github.com/iyulab/lm-supply/blob/main/docs/synthesizer.md";
}
