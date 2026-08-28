using System.Diagnostics;
using AwesomeAssertions;
using LMSupply.Generator.Internal.Llama;
using Xunit;

namespace LMSupply.Generator.Tests.Internal.Llama;

/// <summary>
/// Tests for the severity-aware Trace emission helper that runs at GPU layer decision time.
/// Verifies the regression fix for B-3 (silent CPU-only fallback): full CPU fallback emits
/// TraceWarning, partial offload stays at TraceInformation.
/// </summary>
[Collection("TraceListeners")]
public class LlamaOffloadTraceHelperTests
{
    /// <summary>
    /// Captures only this helper's own emissions, identified by their unique signatures, so a
    /// concurrent test emitting to the process-global <see cref="Trace.Listeners"/> cannot pollute
    /// the assertions. This keeps <c>Information.Should().BeEmpty()</c> meaningful: a stray
    /// "Auto partial offload" wrongly emitted by the 0-layer path is still caught, but unrelated
    /// cross-test noise (ctx-adjust, VramAware, etc.) is not. Regression guard:
    /// <see cref="CapturingTraceListener_IgnoresUnrelatedCrossTestEmissions"/>.
    /// </summary>
    private static bool IsOffloadMessage(string message)
        => message.Contains("CPU-only fallback", StringComparison.Ordinal)
           || message.Contains("Auto partial offload", StringComparison.Ordinal);

    [Fact]
    public void TraceOffloadDecision_ZeroLayers_EmitsWarning()
    {
        var listener = new CapturingTraceListener(IsOffloadMessage);
        Trace.Listeners.Add(listener);
        try
        {
            var estimate = new ResourceEstimate
            {
                EstimatedVramBytes = 0,
                EstimatedRamBytes = 22L * 1024 * 1024 * 1024,
                RecommendedGpuLayers = 0,
                TotalLayers = 32,
                CanFitInVram = false,
                CanFitInRam = true,
            };

            LlamaOffloadTraceHelper.TraceOffloadDecision(
                estimate,
                freeVramBytes: 512L * 1024 * 1024,
                totalVramBytes: 24L * 1024 * 1024 * 1024);

            listener.Warnings.Should().ContainSingle(w =>
                w.Contains("CPU-only fallback") &&
                w.Contains("0/32") &&
                w.Contains("LMSUPPLY_VRAM_BUDGET_MB") &&
                w.Contains("LMSupplyTraceListener"),
                "0 GPU layers must emit TraceWarning with VRAM figures, override hint, and ILogger bridge hint");
            listener.Information.Should().BeEmpty(
                "the 0-layer path must NOT also emit TraceInformation");
        }
        finally
        {
            Trace.Listeners.Remove(listener);
        }
    }

    [Fact]
    public void TraceOffloadDecision_PartialOffload_EmitsInformation()
    {
        var listener = new CapturingTraceListener(IsOffloadMessage);
        Trace.Listeners.Add(listener);
        try
        {
            var estimate = new ResourceEstimate
            {
                EstimatedVramBytes = 12L * 1024 * 1024 * 1024,
                EstimatedRamBytes = 10L * 1024 * 1024 * 1024,
                RecommendedGpuLayers = 18,
                TotalLayers = 32,
                CanFitInVram = false,
                CanFitInRam = true,
            };

            LlamaOffloadTraceHelper.TraceOffloadDecision(
                estimate,
                freeVramBytes: 13L * 1024 * 1024 * 1024,
                totalVramBytes: 24L * 1024 * 1024 * 1024);

            listener.Information.Should().ContainSingle(s =>
                s.Contains("Auto partial offload") && s.Contains("18/32"),
                "partial offload must emit TraceInformation");
            listener.Warnings.Should().BeEmpty(
                "partial offload must NOT trigger Warning severity");
        }
        finally
        {
            Trace.Listeners.Remove(listener);
        }
    }

    [Fact]
    public void TraceOffloadDecision_ZeroLayers_MessageIsAsciiOnly()
    {
        var listener = new CapturingTraceListener(IsOffloadMessage);
        Trace.Listeners.Add(listener);
        try
        {
            var estimate = new ResourceEstimate
            {
                EstimatedVramBytes = 0,
                EstimatedRamBytes = 8L * 1024 * 1024 * 1024,
                RecommendedGpuLayers = 0,
                TotalLayers = 24,
                CanFitInVram = false,
                CanFitInRam = true,
            };

            LlamaOffloadTraceHelper.TraceOffloadDecision(
                estimate, freeVramBytes: 0, totalVramBytes: 8L * 1024 * 1024 * 1024);

            var warning = listener.Warnings.Should().ContainSingle().Subject;
            // Logging convention: operational logs must be ASCII (English) only.
            warning.Should().MatchRegex("^[\\x00-\\x7F]+$",
                "operational log messages must be ASCII per FluxIndex/LMSupply Logging convention");
        }
        finally
        {
            Trace.Listeners.Remove(listener);
        }
    }

    /// <summary>
    /// Regression guard for the cross-test Trace pollution flake: a signature-filtered listener must
    /// ignore unrelated emissions from other tests sharing the process-global Trace.Listeners, so the
    /// helper's own assertions stay deterministic even under parallel execution. This is the surgical
    /// isolation the assembly-level DisableTestParallelization (blunt) backstops — verifying it here
    /// means the helper tests no longer depend on serialization to pass.
    /// </summary>
    [Fact]
    public void CapturingTraceListener_IgnoresUnrelatedCrossTestEmissions()
    {
        var listener = new CapturingTraceListener(IsOffloadMessage);
        Trace.Listeners.Add(listener);
        try
        {
            // Simulate the pollution: other tests emitting to the global listener concurrently.
            Trace.TraceInformation("[LlamaServerGeneratorModel] Auto-cap context length 16384 -> 10469 (ctx-adjust)");
            Trace.TraceInformation("[VramAware] some unrelated info");
            Trace.TraceWarning("[SomethingElse] unrelated warning");

            listener.Information.Should().BeEmpty(
                "signature filter must drop cross-test Information (ctx-adjust, VramAware) that previously polluted BeEmpty");
            listener.Warnings.Should().BeEmpty(
                "signature filter must drop cross-test Warnings unrelated to the offload helper");
        }
        finally
        {
            Trace.Listeners.Remove(listener);
        }
    }

    private sealed class CapturingTraceListener(Predicate<string> accept) : TraceListener
    {
        private readonly Predicate<string> _accept = accept;

        public List<string> Warnings { get; } = [];
        public List<string> Information { get; } = [];

        public override void TraceEvent(TraceEventCache? cache, string source, TraceEventType eventType, int id, string? message)
        {
            // Only capture messages this listener owns; ignore concurrent cross-test emissions to the
            // process-global Trace.Listeners. This isolates the assertions from scheduling races.
            if (message == null || !_accept(message)) return;
            switch (eventType)
            {
                case TraceEventType.Warning:
                    Warnings.Add(message);
                    break;
                case TraceEventType.Information:
                    Information.Add(message);
                    break;
            }
        }

        public override void Write(string? message) { /* unused */ }
        public override void WriteLine(string? message) { /* unused */ }
    }
}
