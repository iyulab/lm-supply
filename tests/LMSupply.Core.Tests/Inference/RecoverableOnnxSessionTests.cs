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
    public void Run_AfterDispose_Throws()
    {
        var session = Create(ExecutionProvider.Cpu, "CPUExecutionProvider");
        session.Dispose();

        var act = () => session.Run((_, _) => 1);

        act.Should().Throw<ObjectDisposedException>();
    }
}
