using System.Diagnostics;
using System.Reflection;
using AwesomeAssertions;
using LMSupply.Exceptions;
using LMSupply.Inference;
using Microsoft.ML.OnnxRuntime;

namespace LMSupply.Core.Tests.Inference;

/// <summary>
/// Entry-condition tests for <see cref="RecoverableOnnxSession"/>'s two recovery paths. The
/// sessions here are null and the model path nonexistent, so every recovery attempt fails at
/// session creation — which is enough to observe whether the path was entered, what it logged, and
/// that a provider is never retried twice. The positive path (a real replacement session) needs a
/// real model and is exercised by the module samples instead.
/// </summary>
public class RecoverableOnnxSessionTests
{
    private sealed class WarningCapture : TraceListener
    {
        public List<string> Warnings { get; } = [];
        public override void Write(string? message) { }
        public override void WriteLine(string? message) { if (message is not null) Warnings.Add(message); }
        public override void TraceEvent(TraceEventCache? e, string source, TraceEventType eventType, int id, string? message, params object?[]? data)
        {
            if (eventType == TraceEventType.Warning && message is not null) Warnings.Add(message);
        }
    }

    private static RecoverableOnnxSession Create(ExecutionProvider requested, params string[] active)
        => new(null!, active, active.Any(p => !p.Contains("CPU")), requested, "/nonexistent/recoverable-test.onnx", logPrefix: "[TestEngine]");

    private static OnnxRuntimeException OnnxCrash(string message)
    {
        var errorCodeType = typeof(OnnxRuntimeException).Assembly.GetType("Microsoft.ML.OnnxRuntime.ErrorCode")!;
        var errorCode = Enum.ToObject(errorCodeType, 6); // RuntimeException
        return (OnnxRuntimeException)Activator.CreateInstance(
            typeof(OnnxRuntimeException), BindingFlags.NonPublic | BindingFlags.Instance, null, [errorCode, message], null)!;
    }

    [Fact]
    public void TryRecoverAfterTimeout_GpuActiveUnderAuto_EntersFallback_UsesCallerPrefix()
    {
        var capture = new WarningCapture();
        Trace.Listeners.Add(capture);
        try
        {
            using var session = Create(ExecutionProvider.Auto, "DmlExecutionProvider", "CPUExecutionProvider");

            session.TryRecoverAfterTimeout(new InferenceTimeoutException(TimeSpan.FromSeconds(60)))
                .Should().BeFalse("replacement creation fails on a nonexistent model");

            capture.Warnings.Should().Contain(w => w.StartsWith("[TestEngine]", StringComparison.Ordinal) && w.Contains("timed out on DirectML"));
        }
        finally { Trace.Listeners.Remove(capture); }
    }

    [Theory]
    [InlineData(ExecutionProvider.Cpu, "CPUExecutionProvider")]
    [InlineData(ExecutionProvider.Auto, "CPUExecutionProvider")]
    public void TryRecoverAfterTimeout_NothingBelowCpu_DoesNothing(ExecutionProvider requested, string active)
    {
        var capture = new WarningCapture();
        Trace.Listeners.Add(capture);
        try
        {
            using var session = Create(requested, active);

            session.TryRecoverAfterTimeout(new InferenceTimeoutException(TimeSpan.FromSeconds(60))).Should().BeFalse();
            capture.Warnings.Should().NotContain(w => w.StartsWith("[TestEngine]", StringComparison.Ordinal) && w.Contains("timed out on"));
        }
        finally { Trace.Listeners.Remove(capture); }
    }

    [Fact]
    public void TryFallback_GpuCrash_EntersFallback_AndBlacklistsProviderForTimeoutToo()
    {
        var capture = new WarningCapture();
        Trace.Listeners.Add(capture);
        try
        {
            using var session = Create(ExecutionProvider.DirectML, "DmlExecutionProvider", "CPUExecutionProvider");

            session.TryFallback(OnnxCrash("simulated DML crash")).Should().BeFalse();
            capture.Warnings.Should().Contain(w => w.StartsWith("[TestEngine]", StringComparison.Ordinal) && w.Contains("Inference failed on DirectML"));

            // The same provider is not retried by the other recovery path either — one blacklist.
            session.TryRecoverAfterTimeout(new InferenceTimeoutException(TimeSpan.FromSeconds(60))).Should().BeFalse();
            capture.Warnings.Count(w => w.StartsWith("[TestEngine]", StringComparison.Ordinal) && w.Contains("Attempting fallback")).Should().Be(1);
        }
        finally { Trace.Listeners.Remove(capture); }
    }

    [Fact]
    public void Run_CancelledToken_ThrowsOperationCanceled_WithoutFallback()
    {
        using var session = Create(ExecutionProvider.Auto, "DmlExecutionProvider", "CPUExecutionProvider");
        using var cts = new CancellationTokenSource();
        cts.Cancel();

        // Gate acquisition observes the token before any work runs.
        var act = () => session.Run<int>((_, _) => throw new InvalidOperationException("must not run"), cts.Token);

        act.Should().Throw<OperationCanceledException>();
    }

    [Fact]
    public void Run_CpuRequested_CrashSurfacesWithoutFallbackAttempt()
    {
        var capture = new WarningCapture();
        Trace.Listeners.Add(capture);
        try
        {
            using var session = Create(ExecutionProvider.Cpu, "CPUExecutionProvider");

            var act = () => session.Run<int>((_, _) => throw OnnxCrash("cpu crash"));

            act.Should().Throw<OnnxRuntimeException>();
            capture.Warnings.Should().NotContain(w => w.StartsWith("[TestEngine]", StringComparison.Ordinal));
        }
        finally { Trace.Listeners.Remove(capture); }
    }

    [Fact]
    public async Task Dispose_RunStillInFlightOnCurrentHandle_LeaksInsteadOfDisposing()
    {
        // docket iyulab/lm-supply#193. A caller-side timeout unblocks the caller but leaves the
        // native call running -- and TryRecoverAfterTimeout only moves that handle aside when a
        // fallback exists. On a CPU-requested session (the reported configuration) it returns
        // false immediately, so the in-flight run stays on the *current* handle. Disposing it
        // there is an access violation that takes the process down, uncatchable, which is exactly
        // the reported symptom. Dispose must poll the gate for the current handle too, not only
        // for abandoned ones.
        var capture = new WarningCapture();
        Trace.Listeners.Add(capture);
        try
        {
            var ct = TestContext.Current.CancellationToken;
            var session = Create(ExecutionProvider.Cpu, "CPUExecutionProvider");
            using var runEntered = new ManualResetEventSlim(false);
            using var releaseRun = new ManualResetEventSlim(false);

            // Stands in for a native Run that has not returned: it holds the handle's gate for as
            // long as the real one would.
            var inFlight = Task.Run(() => session.Run((_, _) =>
            {
                runEntered.Set();
                releaseRun.Wait(TimeSpan.FromSeconds(30), ct);
                return 0;
            }, ct), ct);

            runEntered.Wait(TimeSpan.FromSeconds(10), ct).Should().BeTrue("the run must be in flight before disposing");

            session.Dispose();

            capture.Warnings.Should().Contain(
                w => w.StartsWith("[TestEngine]", StringComparison.Ordinal) && w.Contains("Leaking a session"),
                "a session whose run has not returned must be leaked, never disposed under the call");

            releaseRun.Set();
            (await inFlight).Should().Be(0, "the in-flight run must still complete normally");
        }
        finally { Trace.Listeners.Remove(capture); }
    }

    [Fact]
    public void Dispose_NoRunInFlight_ReclaimsNormally()
    {
        // The other half: with the gate free there is nothing to protect, so Dispose must reclaim
        // rather than leak -- otherwise the fix above would turn every teardown into a leak.
        var capture = new WarningCapture();
        Trace.Listeners.Add(capture);
        try
        {
            var session = Create(ExecutionProvider.Cpu, "CPUExecutionProvider");
            session.Run((_, _) => 1, TestContext.Current.CancellationToken).Should().Be(1);

            session.Dispose();

            capture.Warnings.Should().NotContain(w => w.Contains("Leaking a session"));
        }
        finally { Trace.Listeners.Remove(capture); }
    }

    [Fact]
    public void Run_AfterDispose_Throws()
    {
        var session = Create(ExecutionProvider.Cpu, "CPUExecutionProvider");
        session.Dispose();

        var act = () => session.Run((_, _) => 1);

        act.Should().Throw<ObjectDisposedException>();
    }

    [Fact]
    public async Task Run_GpuCrash_FallbackFails_RethrowsOriginal_AndReleasesGate()
    {
        var capture = new WarningCapture();
        Trace.Listeners.Add(capture);
        try
        {
            using var session = Create(ExecutionProvider.Auto, "DmlExecutionProvider", "CPUExecutionProvider");
            var crash = OnnxCrash("simulated DML crash");
            var ct = TestContext.Current.CancellationToken;

            var act = () => session.Run<int>((_, _) => throw crash, ct);

            // The fallback path is entered (replacement creation fails on the nonexistent model), and
            // the original exception — not a wrapper — reaches the caller.
            act.Should().Throw<OnnxRuntimeException>().Which.Should().BeSameAs(crash);
            capture.Warnings.Should().Contain(w => w.StartsWith("[TestEngine]", StringComparison.Ordinal) && w.Contains("Inference failed on DirectML"));

            // The gate was released on the way out: a second run on the same session must not deadlock.
            var runsAgain = Task.Run(() => session.Run((_, _) => 42, ct), ct);
            var winner = await Task.WhenAny(runsAgain, Task.Delay(TimeSpan.FromSeconds(5), ct));
            winner.Should().BeSameAs(runsAgain, "the crashed run must release the session gate");
            (await runsAgain).Should().Be(42);
        }
        finally { Trace.Listeners.Remove(capture); }
    }

    [Fact]
    public void SharedBlacklist_SiblingFailure_MakesOtherSessionLeaveProviderBeforeItsNextRun()
    {
        var capture = new WarningCapture();
        Trace.Listeners.Add(capture);
        try
        {
            var shared = new ProviderBlacklist();
            using var encoder = new RecoverableOnnxSession(null!, ["DmlExecutionProvider", "CPUExecutionProvider"], true,
                ExecutionProvider.Auto, "/nonexistent/encoder.onnx", logPrefix: "[Encoder]", blacklist: shared);
            using var decoder = new RecoverableOnnxSession(null!, ["DmlExecutionProvider", "CPUExecutionProvider"], true,
                ExecutionProvider.Auto, "/nonexistent/decoder.onnx", logPrefix: "[Decoder]", blacklist: shared);

            // The encoder fails on DirectML. Replacement creation fails (nonexistent model), but the
            // provider is now on the blacklist both sessions consult.
            encoder.TryFallback(OnnxCrash("simulated DML crash")).Should().BeFalse();
            shared.Contains(ExecutionProvider.DirectML).Should().BeTrue();
            decoder.Blacklist.Should().BeSameAs(shared);

            // The decoder has not failed itself, yet its next run first tries to leave DirectML
            // because a sibling blacklisted it — observable as the [Decoder] fallback attempt. With
            // no replacement available it runs where it is rather than failing the caller.
            var ran = false;
            decoder.Run((_, _) => { ran = true; return 0; }, TestContext.Current.CancellationToken);

            ran.Should().BeTrue();
            capture.Warnings.Should().Contain(w => w.StartsWith("[Decoder]", StringComparison.Ordinal) && w.Contains("blacklisted by another session"));
        }
        finally { Trace.Listeners.Remove(capture); }
    }

    [Fact]
    public void FromResult_SeedsBlacklist_WithProvidersTheLoadTimeChainSawFail()
    {
        var shared = new ProviderBlacklist();
        var loadedOnCpuAfterDmlRefusedIt = new SessionCreationResult
        {
            Session = null!,
            RequestedProvider = ExecutionProvider.Auto,
            ActiveProviders = ["CPUExecutionProvider"],
            FailedProviders = [ExecutionProvider.DirectML]
        };

        using var encoder = RecoverableOnnxSession.FromResult(loadedOnCpuAfterDmlRefusedIt, "/nonexistent/encoder.onnx", blacklist: shared);

        // A sibling created afterwards consults the same blacklist, so it will not try DirectML.
        shared.Contains(ExecutionProvider.DirectML).Should().BeTrue();
        encoder.Blacklist.Contains(ExecutionProvider.DirectML).Should().BeTrue();
    }

    [Fact]
    public void PrivateBlacklist_IsPerSession_ByDefault()
    {
        using var a = Create(ExecutionProvider.Auto, "DmlExecutionProvider", "CPUExecutionProvider");
        using var b = Create(ExecutionProvider.Auto, "DmlExecutionProvider", "CPUExecutionProvider");

        a.TryFallback(OnnxCrash("simulated DML crash")).Should().BeFalse();

        a.Blacklist.Contains(ExecutionProvider.DirectML).Should().BeTrue();
        b.Blacklist.Contains(ExecutionProvider.DirectML).Should().BeFalse("sessions created without a shared blacklist do not influence each other");
    }
}
