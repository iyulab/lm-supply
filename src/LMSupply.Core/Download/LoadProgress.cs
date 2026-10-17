namespace LMSupply.Download;

/// <summary>
/// The reporter one load hands down to everything it calls: it forwards every report, except a second
/// <see cref="DownloadPhase.Complete"/> for a file or runtime this load has already completed.
/// </summary>
/// <remarks>
/// The pieces of a load report on their own — each runtime session a load creates provisions the runtime and reports
/// it, so a load with three sessions completed the native ONNX Runtime three times. Only the load knows that the three
/// are one load, so the rule «a file is reported <see cref="DownloadPhase.Complete"/> once» is kept here, at the load's
/// entry point, and not by any one piece. A reporter the caller reuses across loads is wrapped again by the next load,
/// which therefore reports its runtime step too.
/// </remarks>
internal sealed class LoadProgress : IProgress<DownloadProgress>
{
    private readonly IProgress<DownloadProgress> _inner;
    private readonly HashSet<(DownloadKind Kind, string FileName)> _completed = [];

    private LoadProgress(IProgress<DownloadProgress> inner) => _inner = inner;

    /// <summary>
    /// Wraps the reporter a load was given, or returns it when it already belongs to a load (a load that delegates to
    /// another entry point stays one load), or <c>null</c> when there is nothing to report to.
    /// </summary>
    public static IProgress<DownloadProgress>? Begin(IProgress<DownloadProgress>? progress) =>
        progress is null or LoadProgress ? progress : new LoadProgress(progress);

    /// <inheritdoc />
    public void Report(DownloadProgress value)
    {
        if (value.Phase == DownloadPhase.Complete)
        {
            // Sessions may be created concurrently.
            lock (_completed)
            {
                if (!_completed.Add((value.Kind, value.FileName)))
                    return;
            }
        }

        _inner.Report(value);
    }
}
