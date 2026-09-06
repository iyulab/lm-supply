using System.Diagnostics;
using System.Reflection;
using AwesomeAssertions;
using LMSupply.Embedder.Inference;
using LMSupply.Exceptions;
using LMSupply.Inference;

namespace LMSupply.Embedder.Tests;

/// <summary>
/// Tests for <see cref="OnnxInferenceEngine.TryRecoverAfterTimeout"/> — the fallback step taken
/// when a run exceeds <see cref="CancellableInference"/>'s bound while a GPU provider is active.
///
/// Background: the run-time fallback chain only ever triggered on <c>OnnxRuntimeException</c>. The
/// timeout guard introduced for cold-GPU kernel hangs throws <see cref="InferenceTimeoutException"/>
/// instead, which never entered that chain — so a first-run DirectML hang surfaced to the caller
/// as a hard failure with no CPU fallback, exactly on the path a new user hits first. These tests
/// pin the recovery entry conditions; the positive path (session actually swapped and the retry
/// succeeding) requires a real model and is exercised by the live sample instead.
///
/// Like <see cref="OnnxInferenceEngineTests"/>, the engine is built via reflection with a null
/// session and a nonexistent model path: the recovery attempt itself fails at session creation,
/// which is enough to observe whether the fallback was entered and what it logged.
/// </summary>
public class OnnxInferenceEngineTimeoutRecoveryTests
{
    private sealed class WarningCapture : TraceListener
    {
        public List<string> Warnings { get; } = [];

        public override void Write(string? message) { }

        public override void WriteLine(string? message)
        {
            if (message is not null)
                Warnings.Add(message);
        }

        public override void TraceEvent(
            TraceEventCache? e, string source, TraceEventType eventType, int id, string? message, params object?[]? data)
        {
            if (eventType == TraceEventType.Warning && message is not null)
                Warnings.Add(message);
        }
    }

    private static OnnxInferenceEngine CreateEngineViaReflection(
        ExecutionProvider requestedProvider,
        IReadOnlyList<string> activeProviders)
    {
        var ctor = typeof(OnnxInferenceEngine).GetConstructors(
            BindingFlags.NonPublic | BindingFlags.Instance).Single();

        return (OnnxInferenceEngine)ctor.Invoke([
            null!,                      // session — never touched on the failure path
            384,                        // hiddenSize
            false,                      // hasTokenTypeIds
            "last_hidden_state",        // outputName
            requestedProvider != ExecutionProvider.Cpu, // isGpuProvider
            activeProviders.Any(p => !p.Contains("CPU", StringComparison.OrdinalIgnoreCase)), // isGpuActive
            activeProviders,
            requestedProvider,
            "/nonexistent/test_model_timeout_recovery.onnx"
        ]);
    }

    private static InferenceTimeoutException Timeout() => new(TimeSpan.FromSeconds(60));

    [Fact]
    public void TryRecoverAfterTimeout_DirectMLActiveUnderAuto_EntersFallbackAndLogsTimeout()
    {
        var capture = new WarningCapture();
        Trace.Listeners.Add(capture);
        try
        {
            var engine = CreateEngineViaReflection(
                ExecutionProvider.Auto,
                ["DmlExecutionProvider", "CPUExecutionProvider"]);

            var recovered = engine.TryRecoverAfterTimeout(Timeout());

            // Session recreation fails (nonexistent model), so recovery reports false — but the
            // fallback must have been entered, and the log must name the timeout, not a crash.
            recovered.Should().BeFalse("session recreation fails on a nonexistent model path");
            capture.Warnings.Should().Contain(
                w => w.Contains("[OnnxInferenceEngine]") && w.Contains("timed out on DirectML"),
                "a GPU timeout under Auto must enter the provider fallback chain");
        }
        finally
        {
            Trace.Listeners.Remove(capture);
        }
    }

    [Fact]
    public void TryRecoverAfterTimeout_ExplicitCpuRequested_DoesNothing()
    {
        var capture = new WarningCapture();
        Trace.Listeners.Add(capture);
        try
        {
            var engine = CreateEngineViaReflection(ExecutionProvider.Cpu, ["CPUExecutionProvider"]);

            engine.TryRecoverAfterTimeout(Timeout()).Should().BeFalse(
                "an explicit CPU request has no provider below it to fall back to");
            // Trace.Listeners is process-global and other test classes log their own
            // "[OnnxInferenceEngine] Inference failed on ..." lines concurrently, so assert only on
            // the timeout wording this path would emit.
            capture.Warnings.Should().NotContain(w => w.Contains("timed out on"),
                "no timeout fallback attempt must be logged when CPU was requested");
        }
        finally
        {
            Trace.Listeners.Remove(capture);
        }
    }

    [Fact]
    public void TryRecoverAfterTimeout_CpuAlreadyActiveUnderAuto_DoesNothing()
    {
        var capture = new WarningCapture();
        Trace.Listeners.Add(capture);
        try
        {
            // Auto that already landed on CPU (e.g. no GPU on the machine): a timeout here is a
            // genuinely slow inference, not a provider hang — surface it unchanged.
            var engine = CreateEngineViaReflection(ExecutionProvider.Auto, ["CPUExecutionProvider"]);

            engine.TryRecoverAfterTimeout(Timeout()).Should().BeFalse(
                "CPU is the end of the fallback chain");
            capture.Warnings.Should().NotContain(w => w.Contains("timed out on"));
        }
        finally
        {
            Trace.Listeners.Remove(capture);
        }
    }

    [Fact]
    public void TryRecoverAfterTimeout_SameProviderTwice_SecondCallGivesUp()
    {
        var capture = new WarningCapture();
        Trace.Listeners.Add(capture);
        try
        {
            var engine = CreateEngineViaReflection(
                ExecutionProvider.DirectML,
                ["DmlExecutionProvider", "CPUExecutionProvider"]);

            engine.TryRecoverAfterTimeout(Timeout()).Should().BeFalse();
            var attemptsAfterFirst = capture.Warnings.Count(w => w.Contains("timed out on DirectML"));

            engine.TryRecoverAfterTimeout(Timeout()).Should().BeFalse();
            var attemptsAfterSecond = capture.Warnings.Count(w => w.Contains("timed out on DirectML"));

            attemptsAfterFirst.Should().Be(1);
            attemptsAfterSecond.Should().Be(1,
                "a provider that already failed once is blacklisted, so the second timeout must not re-attempt it");
        }
        finally
        {
            Trace.Listeners.Remove(capture);
        }
    }
}
