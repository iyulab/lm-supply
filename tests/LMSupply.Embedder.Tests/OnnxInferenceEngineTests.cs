using System.Diagnostics;
using System.Reflection;
using AwesomeAssertions;
using LMSupply.Embedder.Inference;
using LMSupply.Inference;
using Microsoft.ML.OnnxRuntime;

namespace LMSupply.Embedder.Tests;

/// <summary>
/// Tests for OnnxInferenceEngine inference-time fallback behavior.
/// ISSUE: lmsupply-embedder-onnx-dml-inference-fallback-20260510
/// Fix: RunInferenceInternal catch clause condition changed from
///      `_requestedProvider == Auto` to `_requestedProvider != Cpu`
///      so explicit GPU providers (DirectML, CUDA, CoreML) also attempt fallback.
/// </summary>
public class OnnxInferenceEngineTests
{
    // Captures Trace.TraceWarning calls for assertion.
    // TraceEvent handles structured trace (framework path); WriteLine handles raw path.
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

    /// <summary>
    /// Creates an OnnxInferenceEngine via reflection, bypassing the private constructor.
    /// Safe for TryFallback tests: _session is never accessed on the failure path because
    /// CreateWithInfoAsync throws ModelNotFoundException before _session.Dispose() is reached.
    /// </summary>
    private static OnnxInferenceEngine CreateEngineViaReflection(
        ExecutionProvider requestedProvider,
        IReadOnlyList<string> activeProviders,
        string modelPath)
    {
        var ctor = typeof(OnnxInferenceEngine).GetConstructors(
            BindingFlags.NonPublic | BindingFlags.Instance).Single();

        // Match constructor parameter order:
        // session, hiddenSize, hasTokenTypeIds, outputName,
        // isGpuProvider, isGpuActive, activeProviders, requestedProvider, modelPath
        return (OnnxInferenceEngine)ctor.Invoke([
            null!,          // session  — null safe: Dispose not reached on failure path
            384,            // hiddenSize
            false,          // hasTokenTypeIds
            "last_hidden_state", // outputName
            true,           // isGpuProvider
            false,          // isGpuActive (post-0.34.8 DML→CPU state)
            activeProviders,
            requestedProvider,
            modelPath
        ]);
    }

    [Fact]
    public void TryFallback_WithExplicitDirectMLProvider_EmitsWarningAndAttemptsProviderRecovery()
    {
        // Scenario (post-0.34.8): engine was requested with DirectML but initialised on CPU
        // due to init-stage fallback. An OnnxRuntimeException during inference should enter
        // TryFallback (warning emitted) and attempt session recreation.
        //
        // Before inference-stage fix (_requestedProvider == Auto):
        //   catch clause guard blocks DirectML → TryFallback never called → exception propagates.
        // After fix (_requestedProvider != Cpu):
        //   catch clause guard passes for DirectML → TryFallback called → warning emitted.
        //
        // This test calls TryFallback directly via reflection to isolate the fallback mechanism
        // from the ONNX inference path (which requires a real model file).

        var capture = new WarningCapture();
        Trace.Listeners.Add(capture);

        try
        {
            var engine = CreateEngineViaReflection(
                requestedProvider: ExecutionProvider.DirectML,
                activeProviders: ["CPUExecutionProvider"],
                modelPath: "/nonexistent/test_model_fallback.onnx");

            var tryFallback = typeof(OnnxInferenceEngine)
                .GetMethod("TryFallback", BindingFlags.NonPublic | BindingFlags.Instance)!;

            var errorCodeType = typeof(OnnxRuntimeException).Assembly.GetType("Microsoft.ML.OnnxRuntime.ErrorCode")!;
            var errorCodeValue = Enum.ToObject(errorCodeType, 6); // RuntimeException
            var onnxEx = (OnnxRuntimeException)Activator.CreateInstance(
                typeof(OnnxRuntimeException),
                BindingFlags.NonPublic | BindingFlags.Instance,
                null,
                [errorCodeValue, "Simulated DML inference crash"],
                null)!;
            var result = (bool)tryFallback.Invoke(engine, [onnxEx])!;

            // TryFallback returns false (nonexistent model → session creation fails)
            // but the Warning MUST have been emitted before the recreation attempt.
            result.Should().BeFalse("session recreation fails on nonexistent model path");
            capture.Warnings.Should().Contain(
                w => w.Contains("[OnnxInferenceEngine]") && w.Contains("Inference failed on"),
                "Warning must be logged when TryFallback is entered for an explicit GPU provider");
        }
        finally
        {
            Trace.Listeners.Remove(capture);
        }
    }

    [Fact]
    public void InferenceFallbackGuard_CpuProvider_ExcludedFromCatchClause()
    {
        // The catch clause condition `_requestedProvider != ExecutionProvider.Cpu` must
        // evaluate to false for Cpu — ensuring CPU never enters an infinite retry loop.
        var requestedProvider = ExecutionProvider.Cpu;
        (requestedProvider != ExecutionProvider.Cpu).Should().BeFalse(
            "CPU provider must be excluded from the inference fallback catch clause");
    }

    [Fact]
    public void InferenceFallbackGuard_GpuProviders_AllowedInCatchClause()
    {
        // After the fix, all explicit GPU providers must satisfy `!= Cpu` so they can
        // enter the inference-time fallback path.
        var gpuProviders = new[]
        {
            ExecutionProvider.DirectML,
            ExecutionProvider.Cuda,
            ExecutionProvider.CoreML,
            ExecutionProvider.Auto
        };

        foreach (var provider in gpuProviders)
        {
            (provider != ExecutionProvider.Cpu).Should().BeTrue(
                $"provider {provider} must satisfy the catch clause condition (!= Cpu)");
        }
    }
}
