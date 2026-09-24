using System.Diagnostics;
using AwesomeAssertions;
using Xunit;

namespace LMSupply.Embedder.Tests;

/// <summary>
/// An embedder loaded with <see cref="ExecutionProvider.Cpu"/> does not bring the NVIDIA driver libraries into the
/// process. Observed through the module list, the way a consumer sees it. Module state is per process, so each fact is
/// meaningful only in a process that has not probed the GPU yet — run one alone
/// (<c>--filter-method "*CpuProviderDriverLibrariesTests.*"</c>). Needs the model download and an NVIDIA host for the
/// positive control; LocalOnly keeps it out of CI.
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

    [Fact]
    public async Task CpuProvider_LoadsNoNvidiaDriverLibrary()
    {
        if (!OperatingSystem.IsWindows() || LoadedDriverLibraries().Length > 0)
            Assert.Skip("Needs a Windows process that has not probed the GPU yet — run this fact alone.");

        await using var model = await LocalEmbedder.LoadAsync("default", new EmbedderOptions { Provider = ExecutionProvider.Cpu },
            cancellationToken: TestContext.Current.CancellationToken);
        await model.EmbedAsync("hello world", TestContext.Current.CancellationToken);

        LoadedDriverLibraries().Should().BeEmpty();
    }

    [Fact]
    public async Task AutoProvider_OnAnNvidiaHost_ProbesTheDriver()
    {
        // Positive control: without it the CPU fact would pass on a host where nothing ever loads these libraries.
        if (!OperatingSystem.IsWindows() || LoadedDriverLibraries().Length > 0)
            Assert.Skip("Needs a Windows process that has not probed the GPU yet — run this fact alone.");

        await using var model = await LocalEmbedder.LoadAsync("default", new EmbedderOptions { Provider = ExecutionProvider.Auto },
            cancellationToken: TestContext.Current.CancellationToken);

        if (LoadedDriverLibraries().Length == 0)
            Assert.Skip("No NVIDIA driver on this host.");
        LoadedDriverLibraries().Should().Contain("nvml.dll");
    }
}
