namespace LMSupply.Synthesizer;

/// <summary>
/// Diagnostic ids of the Synthesizer package.
/// </summary>
public static class SynthesizerDiagnostics
{
    /// <summary>
    /// Reported on every use of <see cref="LocalSynthesizer"/> and <see cref="ISynthesizerModel"/>: the package has no
    /// phonemizer for the voices in its catalog, so it cannot speak text on its own — synthesizing text throws
    /// <see cref="NotSupportedException"/> unless the caller supplies an <see cref="IPhonemizer"/>. Given phonemes, or a
    /// phonemizer, the output is real speech. Suppress it (<c>&lt;NoWarn&gt;LMSUPPLY001&lt;/NoWarn&gt;</c> or
    /// <c>#pragma warning disable LMSUPPLY001</c>) to use the package on those terms.
    /// </summary>
    public const string ExperimentalId = "LMSUPPLY001";

    /// <summary>Where the diagnostic points: the synthesizer guide's known issue.</summary>
    public const string ExperimentalUrl = "https://github.com/iyulab/lm-supply/blob/main/docs/synthesizer.md";
}
