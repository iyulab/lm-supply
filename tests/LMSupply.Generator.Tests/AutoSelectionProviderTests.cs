using System.Diagnostics;
using AwesomeAssertions;
using LMSupply.Generator.Internal.Llama;
using LMSupply.Hardware;

namespace LMSupply.Generator.Tests;

/// <summary>
/// <c>"default"</c>, <c>"auto"</c> and <c>"gguf:auto"</c> select with one rule, from the profile of the load's provider
/// (<see cref="GgufModelRegistry.GetAutoSelection(ExecutionProvider)"/>). Before 0.76.0 the first two read the detected
/// GPU even for an explicit <see cref="ExecutionProvider.Cpu"/> and never considered system memory, while
/// <c>"gguf:auto"</c> considered system memory but also read the detected GPU — the same word, two selections, and
/// neither honoured "Cpu".
/// </summary>
[Collection("VramBudget")]
public sealed class AutoSelectionProviderTests : IDisposable
{
    private readonly string? _originalBudgetEnv = Environment.GetEnvironmentVariable(VramBudget.BudgetOverrideEnvVar);

    public void Dispose()
        => Environment.SetEnvironmentVariable(VramBudget.BudgetOverrideEnvVar, _originalBudgetEnv);

    [Fact]
    public void Cpu_SelectsFromSystemMemory_TheVramBudgetDoesNotApply()
    {
        // A VRAM budget large enough for every candidate: an explicit Cpu load must still not "fit VRAM".
        Environment.SetEnvironmentVariable(VramBudget.BudgetOverrideEnvVar, "1000000");

        var selection = GgufModelRegistry.GetAutoSelection(ExecutionProvider.Cpu);

        selection.AvailableVramBytes.Should().Be(0);
        selection.Reason.Should().NotBe(ModelSelectionReason.Fits);
        var ram = HardwareProfile.For(ExecutionProvider.Cpu).SystemMemoryBytes;
        selection.AvailableSystemRamBytes.Should().Be(Math.Max(0, ram - GgufModelRegistry.SystemRamReservedBytes));
    }

    [Fact]
    public void Auto_WithTheSameVramBudget_FitsVram()
    {
        // Positive control for the fact above: the override is what makes every candidate fit when the GPU counts.
        Environment.SetEnvironmentVariable(VramBudget.BudgetOverrideEnvVar, "1000000");

        var selection = GgufModelRegistry.GetAutoSelection(ExecutionProvider.Auto);

        selection.AvailableVramBytes.Should().Be(1_000_000L * 1024 * 1024);
        selection.Reason.Should().Be(ModelSelectionReason.Fits);
    }

    [Theory]
    [InlineData(ExecutionProvider.Cpu)]
    [InlineData(ExecutionProvider.Auto)]
    public void GgufAuto_ResolvesToTheSameModel_AsTheProviderSelection(ExecutionProvider provider)
    {
        var resolved = GgufModelRegistry.Resolve("gguf:auto", provider);

        resolved.Should().NotBeNull();
        resolved!.RepoId.Should().Be(GgufModelRegistry.GetAutoSelection(provider).Selected.RepoId);
    }

    [Fact]
    public void Cpu_PicksTheLargestCandidateThatFitsSystemMemory_ElseTheSmallest()
    {
        var selection = GgufModelRegistry.GetAutoSelection(ExecutionProvider.Cpu);

        // Candidates are ordered largest first.
        var fitting = selection.Candidates.FirstOrDefault(c => c.FitsInSystemRam);
        if (fitting is not null)
        {
            selection.Reason.Should().Be(ModelSelectionReason.FitsInSystemRam);
            selection.Selected.RepoId.Should().Be(fitting.Model.RepoId);
        }
        else
        {
            selection.Reason.Should().Be(ModelSelectionReason.FallbackToSmallest);
            selection.Selected.RepoId.Should().Be(selection.Candidates[^1].Model.RepoId);
        }
    }
}

/// <summary>
/// The CPU selection does not bring the NVIDIA driver libraries into the process. Module state is per process, so each
/// fact is meaningful only in a process that has not probed the GPU yet — run one alone
/// (<c>--filter-method "*AutoSelectionDriverLibrariesTests.*"</c>). LocalOnly keeps it out of CI.
/// </summary>
[Trait("Category", "LocalOnly")]
public sealed class AutoSelectionDriverLibrariesTests
{
    private static readonly string[] DriverLibraries = ["nvml.dll", "nvcuda.dll", "nvcuda64.dll"];

    private static string[] LoadedDriverLibraries()
        => Process.GetCurrentProcess().Modules.Cast<ProcessModule>()
            .Select(m => m.ModuleName.ToLowerInvariant())
            .Where(DriverLibraries.Contains)
            .Distinct()
            .ToArray();

    [Fact]
    public void CpuSelection_LoadsNoNvidiaDriverLibrary()
    {
        if (!OperatingSystem.IsWindows() || LoadedDriverLibraries().Length > 0)
            Assert.Skip("Needs a Windows process that has not probed the GPU yet — run this fact alone.");

        _ = GgufModelRegistry.GetAutoSelection(ExecutionProvider.Cpu);
        _ = GgufModelRegistry.Resolve("gguf:auto", ExecutionProvider.Cpu);

        LoadedDriverLibraries().Should().BeEmpty();
    }

    [Fact]
    public void AutoSelection_OnAnNvidiaHost_ProbesTheDriver()
    {
        // Positive control: without it the CPU fact would pass on a host where nothing ever loads these libraries.
        if (!OperatingSystem.IsWindows() || LoadedDriverLibraries().Length > 0)
            Assert.Skip("Needs a Windows process that has not probed the GPU yet — run this fact alone.");

        _ = GgufModelRegistry.GetAutoSelection(ExecutionProvider.Auto);

        if (LoadedDriverLibraries().Length == 0)
            Assert.Skip("No NVIDIA driver on this host.");
        LoadedDriverLibraries().Should().Contain("nvml.dll");
    }
}
