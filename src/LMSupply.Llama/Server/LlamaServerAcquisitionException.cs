namespace LMSupply.Llama.Server;

/// <summary>
/// Thrown when a llama-server binary could not be acquired, carrying the whole search rather than a
/// sentence about it.
/// </summary>
/// <remarks>
/// <para>
/// Acquisition used to throw a plain <see cref="InvalidOperationException"/> whose message was the
/// only thing a caller had. A consumer that has to answer "can this device do local inference at
/// all?" therefore matched on a substring of that message — there was nothing else to key on. When
/// the message was rewritten to be more accurate, the two wordings shared no substring, their
/// classifier went dead against the exact case it was built for, and nothing anywhere went red.
/// </para>
/// <para>
/// The information was never missing; it was only ever rendered. <see cref="Resolution"/> exposes
/// the same fields the message is built from, and <see cref="Reason"/> is the one to branch on:
/// <see cref="LlamaServerAcquisitionFailure.NoAssetForPlatform"/> says the machine has no build in
/// that release, while the other reasons are moments that can pass.
/// </para>
/// </remarks>
public sealed class LlamaServerAcquisitionException : InvalidOperationException
{
    /// <summary>
    /// Creates the exception for a failed resolution.
    /// </summary>
    /// <param name="resolution">The failed search. Its <c>Describe()</c> becomes the message.</param>
    /// <param name="innerException">The underlying failure, when there was one.</param>
    public LlamaServerAcquisitionException(
        LlamaServerAssetResolution resolution,
        Exception? innerException = null)
        : base(resolution.Describe(), innerException)
    {
        Resolution = resolution;
    }

    /// <summary>The search that failed: platform, architecture, requested backend, the fallback chain actually
    /// walked, the release tag searched and the asset names it published.</summary>
    public LlamaServerAssetResolution Resolution { get; }

    /// <summary>
    /// Which kind of failure this was. Null only if a resolution without a reason reaches here, which
    /// a successful resolution would — this exception is not constructed for those.
    /// </summary>
    public LlamaServerAcquisitionFailure? Reason => Resolution.Reason;

    /// <summary>
    /// Whether this machine has no llama-server build in the release that was searched, as opposed to
    /// a condition that may pass. This is the question a consumer deciding whether to offer local
    /// inference is actually asking.
    /// </summary>
    public bool IsPlatformUnsupported => Reason == LlamaServerAcquisitionFailure.NoAssetForPlatform;
}
