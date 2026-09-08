using AwesomeAssertions;
using LMSupply.Llama.Server;

namespace LMSupply.Llama.Tests.Server;

/// <summary>
/// Pins the startup-wait policy that decides when llama-server has stopped making progress.
///
/// <para>
/// Why this exists: the wait used to be a single fixed deadline (120 s at every call site). A
/// 4.2 GB Q4_0 model read through mmap from a cold page cache took ~80 s to load and then warmed up
/// on a single thread, so the deadline fired while the process was alive and still working
/// (<c>Exit code: still running</c>). During that load the server's stderr was silent for the whole
/// 80 s — the load phase emits no line-delimited output — so "reset on a new stderr line" would have
/// killed the very same start. The tracker therefore treats <em>any</em> observable activity as
/// progress: a new stderr line, a working-set high-water mark, or CPU time consumed.
/// </para>
///
/// <para>
/// The tests feed synthetic samples with an explicit clock so no process is launched.
/// </para>
/// </summary>
public class StartupProgressTrackerTests
{
    private static readonly DateTime T0 = new(2026, 9, 8, 4, 0, 0, DateTimeKind.Utc);

    private static StartupProgressTracker NewTracker(int stallSeconds = 120, int capMinutes = 10) =>
        new(T0, stallTimeout: TimeSpan.FromSeconds(stallSeconds), timeout: TimeSpan.FromMinutes(capMinutes));

    [Fact]
    public void WorkingSetGrowth_AlonePastTheOldFixedBudget_IsStillWaiting()
    {
        // The exact shape of the CI failure: stderr flat, RSS climbing for well over 120 s.
        var tracker = NewTracker();
        var verdict = StartupWaitVerdict.Waiting;
        for (var s = 1; s <= 200; s++)
        {
            verdict = tracker.Observe(T0.AddSeconds(s), stderrLength: 11, workingSet: 100L * 1024 * 1024 * s, cpuTime: TimeSpan.Zero);
            verdict.Should().Be(StartupWaitVerdict.Waiting, $"working set was still growing at {s}s");
        }

        verdict.Should().Be(StartupWaitVerdict.Waiting);
    }

    [Fact]
    public void CpuTimeGrowth_Alone_IsProgress()
    {
        // Warm-up after the load: RSS plateaus, stderr silent, only CPU time moves.
        var tracker = NewTracker();
        for (var s = 1; s <= 200; s++)
        {
            tracker.Observe(T0.AddSeconds(s), stderrLength: 11, workingSet: 5_000_000_000, cpuTime: TimeSpan.FromMilliseconds(250 * s))
                .Should().Be(StartupWaitVerdict.Waiting, $"cpu time was still growing at {s}s");
        }
    }

    [Fact]
    public void NewStderrLine_Alone_IsProgress()
    {
        var tracker = NewTracker(stallSeconds: 30);
        tracker.Observe(T0.AddSeconds(29), 10, 100, TimeSpan.Zero).Should().Be(StartupWaitVerdict.Waiting);
        // A single new line at 29 s resets the stall window; 31 s later we are still inside it.
        tracker.Observe(T0.AddSeconds(29), 11, 100, TimeSpan.Zero).Should().Be(StartupWaitVerdict.Waiting);
        tracker.Observe(T0.AddSeconds(58), 11, 100, TimeSpan.Zero).Should().Be(StartupWaitVerdict.Waiting);
        tracker.Observe(T0.AddSeconds(60), 11, 100, TimeSpan.Zero).Should().Be(StartupWaitVerdict.Stalled);
    }

    [Fact]
    public void NoActivityForTheStallWindow_IsStalled()
    {
        var tracker = NewTracker(stallSeconds: 60);
        tracker.Observe(T0.AddSeconds(1), 11, 1_000, TimeSpan.FromSeconds(1)).Should().Be(StartupWaitVerdict.Waiting);
        tracker.Observe(T0.AddSeconds(59), 11, 1_000, TimeSpan.FromSeconds(1)).Should().Be(StartupWaitVerdict.Waiting);
        tracker.Observe(T0.AddSeconds(61), 11, 1_000, TimeSpan.FromSeconds(1)).Should().Be(StartupWaitVerdict.Stalled);
    }

    [Fact]
    public void WorkingSetShrinking_IsNotProgress()
    {
        // Only a new high-water mark counts: a process paging out is not making headway.
        var tracker = NewTracker(stallSeconds: 60);
        tracker.Observe(T0.AddSeconds(1), 11, 5_000, TimeSpan.Zero).Should().Be(StartupWaitVerdict.Waiting);
        tracker.Observe(T0.AddSeconds(30), 11, 4_000, TimeSpan.Zero).Should().Be(StartupWaitVerdict.Waiting);
        tracker.Observe(T0.AddSeconds(62), 11, 3_000, TimeSpan.Zero).Should().Be(StartupWaitVerdict.Stalled);
    }

    [Fact]
    public void ContinuousProgress_PastTheAbsoluteCap_IsTimedOut()
    {
        // The cap is the guard against a process that keeps allocating forever without ever
        // answering /health; it is meant to be far above any legitimate start.
        var tracker = NewTracker(stallSeconds: 60, capMinutes: 5);
        for (var s = 1; s < 300; s++)
        {
            tracker.Observe(T0.AddSeconds(s), 11, 1_000L * s, TimeSpan.Zero).Should().Be(StartupWaitVerdict.Waiting);
        }

        tracker.Observe(T0.AddSeconds(300), 11, 1_000L * 300, TimeSpan.Zero).Should().Be(StartupWaitVerdict.TimedOut);
    }

    [Fact]
    public void Describe_NamesWhatWasObserved()
    {
        // The exception message is the only diagnostic a CI log gets; it has to say which limit
        // fired and what the process looked like when it did, not just "failed within N seconds".
        var tracker = NewTracker(stallSeconds: 60);
        tracker.Observe(T0.AddSeconds(10), 11, 4_000_000_000, TimeSpan.FromSeconds(3));
        tracker.Observe(T0.AddSeconds(75), 11, 4_000_000_000, TimeSpan.FromSeconds(3)).Should().Be(StartupWaitVerdict.Stalled);

        var text = tracker.Describe(T0.AddSeconds(75));

        text.Should().Contain("no progress for 65s", "the age of the last observed activity is the actionable number");
        text.Should().Contain("stall limit 60s");
        text.Should().Contain("elapsed 75s");
        text.Should().Contain("working set 3815 MB");
        text.Should().Contain("cpu time 3.0s");
    }

    [Fact]
    public void StallTimeout_MustNotExceedTimeout()
    {
        // A stall window longer than the cap would make the stall limit unreachable — the cap would
        // always fire first and the diagnostic would name the wrong limit.
        var act = () => new StartupProgressTracker(T0, stallTimeout: TimeSpan.FromMinutes(20), timeout: TimeSpan.FromMinutes(10));
        act.Should().Throw<ArgumentOutOfRangeException>();
    }

    [Fact]
    public void NonPositiveLimits_AreRejected()
    {
        var zeroStall = () => new StartupProgressTracker(T0, stallTimeout: TimeSpan.Zero, timeout: TimeSpan.FromMinutes(10));
        var zeroCap = () => new StartupProgressTracker(T0, stallTimeout: TimeSpan.FromSeconds(1), timeout: TimeSpan.Zero);
        zeroStall.Should().Throw<ArgumentOutOfRangeException>();
        zeroCap.Should().Throw<ArgumentOutOfRangeException>();
    }
}
