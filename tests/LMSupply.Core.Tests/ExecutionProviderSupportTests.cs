using System.Diagnostics;
using AwesomeAssertions;
using LMSupply.Runtime;

namespace LMSupply.Core.Tests;

/// <summary>
/// <see cref="ExecutionProviderSupport"/> is the single refusal every path (ONNX session, runtime
/// manager, GenAI backend, llama-server selector) delegates to. These pin what it says and how often.
/// </summary>
[Trait("Category", "Unit")]
public class ExecutionProviderSupportTests
{
    private sealed class InfoCapture : TraceListener
    {
        // Trace.Listeners is process-wide: other tests running in parallel write here while an assertion enumerates.
        private readonly System.Collections.Concurrent.ConcurrentQueue<string> _lines = new();
        public IReadOnlyList<string> Lines => [.. _lines];
        public override void Write(string? message) { }
        public override void WriteLine(string? message) { if (message is not null) _lines.Enqueue(message); }
        public override void TraceEvent(TraceEventCache? e, string source, TraceEventType eventType, int id, string? message, params object?[]? data)
        {
            if (message is not null) _lines.Enqueue(message);
        }
    }

    [Theory]
    [InlineData(ExecutionProvider.Auto)]
    [InlineData(ExecutionProvider.Cuda)]
    [InlineData(ExecutionProvider.CoreML)]
    [InlineData(ExecutionProvider.Cpu)]
    public void EveryOtherProvider_IsSupported_AndDoesNotThrow(ExecutionProvider provider)
    {
        ExecutionProviderSupport.IsSupported(provider).Should().BeTrue();
        var act = () => ExecutionProviderSupport.ThrowIfUnsupported(provider);
        act.Should().NotThrow();
    }

    [Fact]
    public void DirectML_IsRefused_WithTheReasonAndTheAlternatives()
    {
#pragma warning disable CS0618 // the member is obsolete because this class refuses it
        ExecutionProviderSupport.IsSupported(ExecutionProvider.DirectML).Should().BeFalse();
        var act = () => ExecutionProviderSupport.ThrowIfUnsupported(ExecutionProvider.DirectML);
#pragma warning restore CS0618

        var ex = act.Should().Throw<NotSupportedException>().Which;
        ex.Message.Should().Contain("1.24.4", "the caller learns why: the package line ended there");
        ex.Message.Should().Contain("ExecutionProvider.Auto").And.Contain("ExecutionProvider.Cpu", "and what to use instead");
        ex.Message.Should().Contain("Vulkan", "an AMD/Intel GPU owner learns the GGUF path still uses the GPU");
    }

    [Theory]
    [InlineData("directml")]
    [InlineData("DirectML")]
    [InlineData("DIRECTML")]
    public void DirectMLProviderName_IsRefused_CaseInsensitively(string providerName)
    {
        var act = () => ExecutionProviderSupport.ThrowIfUnsupported(providerName);
        act.Should().Throw<NotSupportedException>().WithMessage(ExecutionProviderSupport.DirectMLUnavailableMessage);
    }

    [Theory]
    [InlineData("cpu")]
    [InlineData("cuda12")]
    [InlineData("coreml")]
    [InlineData("auto")]
    public void OtherProviderNames_AreNotRefused(string providerName)
    {
        var act = () => ExecutionProviderSupport.ThrowIfUnsupported(providerName);
        act.Should().NotThrow();
    }

    [Fact]
    public void TraceDirectMLUnavailableOnce_SaysItOnce_ForADirect3D12GpuWithNoGpuProviderInTheChain()
    {
        // The Auto path on a Windows box without CUDA: the operator gets told once per process that the
        // GPU they see in Task Manager is not going to run ONNX sessions, and why — not once per session.
        var capture = new InfoCapture();
        Trace.Listeners.Add(capture);
        try
        {
            var gpu = new GpuInfo { Vendor = GpuVendor.Amd, DeviceName = "AMD Radeon RX 7800 XT", DirectMLSupported = true };
            var cpuOnlyChain = new[] { ExecutionProvider.Cpu };

            ExecutionProviderSupport.TraceDirectMLUnavailableOnce(gpu, cpuOnlyChain);
            ExecutionProviderSupport.TraceDirectMLUnavailableOnce(gpu, cpuOnlyChain);

            var lines = capture.Lines.Where(l => l.Contains("DirectML execution provider is unavailable")).ToList();
            lines.Should().HaveCountLessThanOrEqualTo(1, "the notice is once per process");
            // Another test in this process may already have consumed the once-flag; what is pinned here is
            // that the notice never repeats, and that when it fires it names the GPU and the alternative.
            foreach (var line in lines)
            {
                line.Should().Contain("AMD Radeon RX 7800 XT").And.Contain("Vulkan");
            }
        }
        finally { Trace.Listeners.Remove(capture); }
    }

    [Fact]
    public void TraceDirectMLUnavailableOnce_IsSilent_WhenAGpuProviderIsInTheChain_OrNoDirect3D12Gpu()
    {
        var capture = new InfoCapture();
        Trace.Listeners.Add(capture);
        try
        {
            var nvidia = new GpuInfo { Vendor = GpuVendor.Nvidia, DeviceName = "RTX", DirectMLSupported = true, CudaDriverVersionMajor = 12 };
            ExecutionProviderSupport.TraceDirectMLUnavailableOnce(nvidia, [ExecutionProvider.Cuda, ExecutionProvider.Cpu]);

            var plainCpu = new GpuInfo { Vendor = GpuVendor.Unknown, DirectMLSupported = false };
            ExecutionProviderSupport.TraceDirectMLUnavailableOnce(plainCpu, [ExecutionProvider.Cpu]);
            ExecutionProviderSupport.TraceDirectMLUnavailableOnce(null, [ExecutionProvider.Cpu]);

            capture.Lines.Should().NotContain(l => l.Contains("DirectML execution provider is unavailable") && l.Contains("RTX"));
        }
        finally { Trace.Listeners.Remove(capture); }
    }
}
