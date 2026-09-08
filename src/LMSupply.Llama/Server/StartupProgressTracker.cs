using System.Globalization;

namespace LMSupply.Llama.Server;

/// <summary>
/// What the startup wait concluded from the latest sample.
/// </summary>
internal enum StartupWaitVerdict
{
    /// <summary>The process is alive and has shown activity recently — keep waiting.</summary>
    Waiting,

    /// <summary>No activity of any kind for <see cref="LlamaServerConfig.StartupStallTimeout"/>.</summary>
    Stalled,

    /// <summary>
    /// <see cref="LlamaServerConfig.StartupTimeout"/> elapsed even though the process kept showing
    /// activity — the guard against a start that never converges.
    /// </summary>
    TimedOut,
}

/// <summary>
/// Decides whether a starting llama-server is still making progress, from periodic samples of the
/// process rather than from the wall clock alone.
///
/// <para>
/// A fixed deadline cannot tell "slow" from "stuck": a 4 GB model read through mmap from a cold page
/// cache legitimately takes minutes on a shared CI runner or a laptop HDD, and the process is busy the
/// whole time. stderr is not a usable liveness signal for that phase either — the tensor load emits
/// no line-delimited output, so the captured stream is silent for the entire read. What does move is
/// the process itself: its working set climbs as pages are faulted in, its CPU time climbs as the
/// context is built and warmed up, and the occasional log line arrives in between. Any of those
/// counts as progress here; the stall window restarts on each, and only a process that shows none of
/// them for the whole window is declared stuck. The absolute limit stays as a far-out cap so a process
/// that keeps allocating without ever answering <c>/health</c> is still bounded.
/// </para>
/// </summary>
internal sealed class StartupProgressTracker
{
    private readonly DateTime _startedAt;
    private readonly TimeSpan _stallTimeout;
    private readonly TimeSpan _timeout;

    private DateTime _lastProgressAt;
    private int _stderrLength;
    private long _workingSetHigh;
    private TimeSpan _cpuTime;
    private long _lastWorkingSet;
    private StartupWaitVerdict _verdict = StartupWaitVerdict.Waiting;

    /// <param name="startedAt">When the process was launched.</param>
    /// <param name="stallTimeout">How long the process may go without any observable activity.</param>
    /// <param name="timeout">The absolute limit on the whole start, regardless of activity.</param>
    public StartupProgressTracker(DateTime startedAt, TimeSpan stallTimeout, TimeSpan timeout)
    {
        if (stallTimeout <= TimeSpan.Zero)
            throw new ArgumentOutOfRangeException(nameof(stallTimeout), stallTimeout, "The stall limit must be positive.");
        if (timeout <= TimeSpan.Zero)
            throw new ArgumentOutOfRangeException(nameof(timeout), timeout, "The startup limit must be positive.");
        if (stallTimeout > timeout)
            throw new ArgumentOutOfRangeException(nameof(stallTimeout), stallTimeout,
                $"The stall limit ({stallTimeout}) must not exceed the startup limit ({timeout}); the cap would always fire first and name the wrong limit.");

        _startedAt = startedAt;
        _stallTimeout = stallTimeout;
        _timeout = timeout;
        _lastProgressAt = startedAt;
    }

    /// <summary>
    /// Records one sample of the process and returns the verdict as of <paramref name="now"/>.
    /// </summary>
    /// <param name="now">Sample time.</param>
    /// <param name="stderrLength">Characters captured from stderr so far (any growth is a new line).</param>
    /// <param name="workingSet">Resident set in bytes; only a new high-water mark counts as progress.</param>
    /// <param name="cpuTime">Total processor time consumed so far.</param>
    public StartupWaitVerdict Observe(DateTime now, int stderrLength, long workingSet, TimeSpan cpuTime)
    {
        var progressed = false;

        if (stderrLength > _stderrLength)
        {
            _stderrLength = stderrLength;
            progressed = true;
        }

        if (workingSet > _workingSetHigh)
        {
            _workingSetHigh = workingSet;
            progressed = true;
        }

        if (cpuTime > _cpuTime)
        {
            _cpuTime = cpuTime;
            progressed = true;
        }

        _lastWorkingSet = workingSet;

        if (progressed)
        {
            _lastProgressAt = now;
        }

        if (now - _startedAt >= _timeout)
        {
            _verdict = StartupWaitVerdict.TimedOut;
        }
        else if (now - _lastProgressAt >= _stallTimeout)
        {
            _verdict = StartupWaitVerdict.Stalled;
        }
        else
        {
            _verdict = StartupWaitVerdict.Waiting;
        }

        return _verdict;
    }

    /// <summary>
    /// One line naming which limit fired and what the process looked like when it did — the only
    /// diagnostic a CI log gets, so it carries the numbers an operator would otherwise have to guess.
    /// </summary>
    public string Describe(DateTime now)
    {
        var limit = _verdict switch
        {
            StartupWaitVerdict.Stalled => "no observable activity for the stall limit",
            StartupWaitVerdict.TimedOut => "the absolute startup limit elapsed while the process was still active",
            _ => "still waiting",
        };

        return string.Create(CultureInfo.InvariantCulture,
            $"{limit}: elapsed {Seconds(now - _startedAt)}s, no progress for {Seconds(now - _lastProgressAt)}s " +
            $"(stall limit {Seconds(_stallTimeout)}s, absolute limit {Seconds(_timeout)}s), " +
            $"working set {Math.Round(_lastWorkingSet / 1048576.0)} MB, cpu time {_cpuTime.TotalSeconds:F1}s, " +
            $"stderr {_stderrLength} chars");
    }

    private static long Seconds(TimeSpan span) => (long)Math.Round(span.TotalSeconds);
}
