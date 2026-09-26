using System.Diagnostics;
using AwesomeAssertions;
using Xunit;

namespace LMSupply.Captioner.Tests;

/// <summary>
/// A captioner loaded with <see cref="ExecutionProvider.Cpu"/> does not probe the GPU. The download ranking once did (it
/// ranked quantizations by the hardware tier, which reads NVML); LMSupply 0.78.0 fixed it on every ONNX entry point, but
/// only the generator and embedder were measured. Observed through the module list, the way a consumer sees it, and
/// through the NVML probe trace (which carries the probing stack).
/// Module state is per process, so each fact is meaningful only in a process that has not probed the GPU yet: run one
/// alone (<c>--filter-method "*CpuProviderDriverLibrariesTests.CpuProvider*"</c>). Needs the model cached and, for the
/// positive control, an NVIDIA host; LocalOnly keeps it out of CI.
/// </summary>
[Trait("Category", "LocalOnly")]
public sealed class CpuProviderDriverLibrariesTests
{
    private const string Alias = "default";
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
            Assert.Skip("Needs a Windows process that has not probed the GPU yet - run this fact alone.");

        var listener = new ProbeStackListener();
        Trace.Listeners.Add(listener);
        try
        {
            await using var model = await LocalCaptioner.LoadAsync(Alias,
                new CaptionerOptions { Provider = ExecutionProvider.Cpu },
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
            Assert.Skip("Needs a Windows process that has not probed the GPU yet - run this fact alone.");

        await using var model = await LocalCaptioner.LoadAsync(Alias,
            new CaptionerOptions { Provider = ExecutionProvider.Auto },
            cancellationToken: TestContext.Current.CancellationToken);

        if (LoadedDriverLibraries().Length == 0)
            Assert.Skip("No NVIDIA driver on this host.");
        LoadedDriverLibraries().Should().Contain("nvml.dll");
    }
}
