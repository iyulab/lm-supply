using System.Diagnostics;

namespace LMSupply.Download;

/// <summary>
/// Forwards <see cref="DownloadProgress"/> reports at a density a consumer can bind to directly:
/// the first report, every report that advances the file by at least one percent, at least one
/// report every <see cref="MinInterval"/>, a change of file, and the report that completes a file.
/// Everything in between is dropped.
/// </summary>
/// <remarks>
/// The download loop reports once per read — every 16 KB — so a 470 MB model produced 30,006
/// callbacks. <see cref="IProgress{T}"/> callbacks are synchronous, and the <c>Progress&lt;T&gt;</c>
/// most consumers use marshals each one onto a UI thread; thirty thousand posts for one file is not
/// a progress bar, it is a queue. Coalescing belongs here, once, rather than in every consumer.
/// </remarks>
internal sealed class CoalescingProgress : IProgress<DownloadProgress>
{
    /// <summary>Minimum percent advance (per file) that earns a forwarded report.</summary>
    public const double MinPercentStep = 1.0;

    /// <summary>Longest silence allowed while bytes keep arriving.</summary>
    public static readonly TimeSpan MinInterval = TimeSpan.FromMilliseconds(250);

    private readonly IProgress<DownloadProgress> _inner;
    private readonly Func<TimeSpan> _clock;

    private string? _lastFile;
    private double _lastPercent;
    private TimeSpan _lastAt;
    private bool _any;

    /// <param name="inner">The consumer's reporter.</param>
    /// <param name="clock">Monotonic clock, injectable for tests; defaults to a <see cref="Stopwatch"/> started now.</param>
    public CoalescingProgress(IProgress<DownloadProgress> inner, Func<TimeSpan>? clock = null)
    {
        _inner = inner ?? throw new ArgumentNullException(nameof(inner));
        if (clock is null)
        {
            var sw = Stopwatch.StartNew();
            clock = () => sw.Elapsed;
        }
        _clock = clock;
    }

    /// <summary>Wraps <paramref name="progress"/>, or returns <c>null</c> when there is nothing to report to.</summary>
    public static IProgress<DownloadProgress>? Wrap(IProgress<DownloadProgress>? progress) =>
        progress is null ? null : new CoalescingProgress(progress);

    /// <inheritdoc />
    public void Report(DownloadProgress value)
    {
        var now = _clock();
        var percent = value.PercentComplete;
        var fileChanged = !string.Equals(value.FileName, _lastFile, StringComparison.Ordinal);
        var complete = value.TotalBytes > 0 && value.BytesDownloaded >= value.TotalBytes;

        var forward = !_any
            || fileChanged
            || complete
            || percent - _lastPercent >= MinPercentStep
            || now - _lastAt >= MinInterval;

        if (!forward)
            return;

        _any = true;
        _lastFile = value.FileName;
        _lastPercent = percent;
        _lastAt = now;
        _inner.Report(value);
    }
}
