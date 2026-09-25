using System.Diagnostics;
using AwesomeAssertions;
using Xunit;

using LMSupply.Generator.Onnx;

namespace LMSupply.Generator.Onnx.Tests;

/// <summary>
/// A generator loaded by preset alias with <see cref="ExecutionProvider.Cpu"/> does not probe the GPU. Observed through
/// the module list, the way a consumer sees it, and through the NVML probe trace (which carries the probing stack).
/// Module state is per process, so each fact is meaningful only in a process that has not probed the GPU yet — run one
/// alone (<c>--filter-method "*CpuProviderDriverLibrariesTests.*"</c>). Needs the model download and an NVIDIA host for
/// the positive control; LocalOnly keeps it out of CI.
/// </summary>
[Trait("Category", "LocalOnly")]
public sealed class CpuProviderDriverLibrariesTests
{
    private static readonly string[] DriverLibraries = ["nvml.dll", "nvcuda.dll", "nvcuda64.dll"];

    private static string[] LoadedDriverLibraries()
        => Process.GetCurrentProcess().Modules.Cast<ProcessModule>()
            .Select(m => m.ModuleName.ToLowerInvariant())
            .Where(DriverLibraries.Contains)
            .Distinct()
            .ToArray();

    private sealed class ProbeStackListener : TraceListener
    {
        public List<string> Stacks { get; } = [];
        public override void Write(string? message) { }
        public override void WriteLine(string? message)
        {
            if (message?.Contains("[GpuDetector] NVML probe", StringComparison.Ordinal) == true)
                lock (Stacks) Stacks.Add(Environment.StackTrace);
        }
    }

    [Fact]
    public async Task CpuProvider_PresetAlias_LoadsNoNvidiaDriverLibrary()
    {
        if (!OperatingSystem.IsWindows() || LoadedDriverLibraries().Length > 0)
            Assert.Skip("Needs a Windows process that has not probed the GPU yet — run this fact alone.");

        OnnxGeneratorBackend.Register();
        var listener = new ProbeStackListener();
        Trace.Listeners.Add(listener);
        try
        {
            await using var model = await LocalGenerator.LoadAsync("phi-4-mini",
                new GeneratorOptions { Provider = ExecutionProvider.Cpu },
                cancellationToken: TestContext.Current.CancellationToken);

            listener.Stacks.Should().BeEmpty("an explicit Cpu load must not probe the GPU; probing stack(s):\n{0}",
                string.Join("\n----\n", listener.Stacks));
            LoadedDriverLibraries().Should().BeEmpty();
        }
        finally
        {
            Trace.Listeners.Remove(listener);
        }
    }

    [Fact]
    public async Task AutoProvider_PresetAlias_OnAnNvidiaHost_ProbesTheDriver()
    {
        // Positive control: without it the CPU fact would pass on a host where nothing ever loads these libraries.
        if (!OperatingSystem.IsWindows() || LoadedDriverLibraries().Length > 0)
            Assert.Skip("Needs a Windows process that has not probed the GPU yet — run this fact alone.");

        OnnxGeneratorBackend.Register();
        await using var model = await LocalGenerator.LoadAsync("phi-4-mini",
            new GeneratorOptions { Provider = ExecutionProvider.Auto },
            cancellationToken: TestContext.Current.CancellationToken);

        if (LoadedDriverLibraries().Length == 0)
            Assert.Skip("No NVIDIA driver on this host.");
        LoadedDriverLibraries().Should().Contain("nvml.dll");
    }
}
