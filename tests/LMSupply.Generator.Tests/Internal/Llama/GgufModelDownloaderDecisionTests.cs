using AwesomeAssertions;
using LMSupply.Core.Download;
using LMSupply.Generator.Internal.Llama;

namespace LMSupply.Generator.Tests.Internal.Llama;

/// <summary>
/// Tests for GgufModelDownloader.DecideRegistryFile — the pure, HW/network-free decision that picks
/// which GGUF quantization file to download for a registry model given a memory budget.
/// Capable budgets keep the registry default quant; tight budgets downscale to a smaller quant;
/// when nothing fits, the smallest is returned with a fallback signal. HW-free: budget + groups injected.
/// </summary>
public class GgufModelDownloaderDecisionTests
{
    private const long GB = 1024L * 1024 * 1024;

    private static GgufModelInfo Model() => new()
    {
        RepoId = "test/Model-GGUF",
        DisplayName = "Test Model",
        DefaultFile = "Model-Q4_K_M.gguf",
        ChatFormat = "chatml",
        ContextLength = 4096,
        License = LicenseTier.MIT,
        QuantizationType = "Q4_K_M",
    };

    private static GgufFileGroup Grp(string name, double gb) =>
        new() { PrimaryFileName = name, Parts = [name], TotalSizeBytes = (long)(gb * GB) };

    // Q4 5.0GB, Q3 3.5GB, Q2 2.5GB — distinct quant tiers of the same model.
    private static IReadOnlyList<GgufFileGroup> Groups() =>
    [
        Grp("Model-Q4_K_M.gguf", 5.0),
        Grp("Model-Q3_K_M.gguf", 3.5),
        Grp("Model-Q2_K.gguf", 2.5),
    ];

    [Fact]
    public void DefaultFits_AmpleRam_KeepsDefaultQuant()
    {
        // 32GB RAM (usable 28GB) holds Q4 (~7.5GB total incl. KV) → keep the registry default.
        var budget = new AvailableMemory(VramBytes: 0, RamBytes: 32 * GB);
        var d = GgufModelDownloader.DecideRegistryFile(Model(), Groups(), budget, vramOnly: false);

        d.Reason.Should().Be(GgufModelDownloader.RegistryFileReason.DefaultFits);
        d.FileName.Should().Be("Model-Q4_K_M.gguf");
    }

    [Fact]
    public void Downscaled_DefaultTooBig_PicksLargestSmallerQuantThatFits()
    {
        // 10GB RAM (usable 6GB): Q4 (~7.5GB) does NOT fit, Q3 (~5.2GB) fits → downscale to Q3.
        var budget = new AvailableMemory(VramBytes: 0, RamBytes: 10 * GB);
        var d = GgufModelDownloader.DecideRegistryFile(Model(), Groups(), budget, vramOnly: false);

        d.Reason.Should().Be(GgufModelDownloader.RegistryFileReason.Downscaled);
        d.FileName.Should().Be("Model-Q3_K_M.gguf");
    }

    [Fact]
    public void FallbackSmallest_NothingFits_ReturnsSmallestQuant()
    {
        // 7GB RAM (usable 3GB): even Q2 (~4GB total) does not fit → smallest as best-effort.
        var budget = new AvailableMemory(VramBytes: 0, RamBytes: 7 * GB);
        var d = GgufModelDownloader.DecideRegistryFile(Model(), Groups(), budget, vramOnly: false);

        d.Reason.Should().Be(GgufModelDownloader.RegistryFileReason.FallbackSmallest);
        d.FileName.Should().Be("Model-Q2_K.gguf");
    }

    [Fact]
    public void GroupsUnavailable_FallsBackToRegistryDefaultFile()
    {
        // Repo listing failed (empty) → degrade to the registry default file rather than fail.
        var budget = new AvailableMemory(VramBytes: 0, RamBytes: 32 * GB);
        var d = GgufModelDownloader.DecideRegistryFile(Model(), [], budget, vramOnly: false);

        d.Reason.Should().Be(GgufModelDownloader.RegistryFileReason.GroupsUnavailable);
        d.FileName.Should().Be("Model-Q4_K_M.gguf");
    }

    [Fact]
    public void VramOnly_LargeVram_KeepsDefaultViaVramBudget()
    {
        // GPU host: 24GB VRAM (usable 22GB) holds Q4 → keep default via VRAM budget.
        var budget = new AvailableMemory(VramBytes: 24 * GB, RamBytes: 4 * GB);
        var d = GgufModelDownloader.DecideRegistryFile(Model(), Groups(), budget, vramOnly: true);

        d.Reason.Should().Be(GgufModelDownloader.RegistryFileReason.DefaultFits);
        d.FileName.Should().Be("Model-Q4_K_M.gguf");
    }

    // ─── BuildSelectionBudget: backend-aware budget (D2 integrated-GPU consistency) ───

    [Fact]
    public void BuildSelectionBudget_CpuBackend_ZeroesVram_UsesRam()
    {
        // Integrated GPU reports a (useless) 128MB VRAM; on a CPU backend it must be ignored.
        var gpu = new LMSupply.Runtime.GpuInfo
        {
            Vendor = LMSupply.Runtime.GpuVendor.Intel,
            TotalMemoryBytes = 128L * 1024 * 1024,
        };
        var budget = GgufModelDownloader.BuildSelectionBudget(
            cpuBackend: true, gpu, systemRamBytes: 16 * GB, contextLength: 4096, out var vramOnly);

        vramOnly.Should().BeFalse();
        budget.VramBytes.Should().Be(0, "CPU backend must not use the integrated-GPU VRAM reading");
        budget.RamBytes.Should().Be(16 * GB);
    }

    [Fact]
    public void BuildSelectionBudget_GpuBackend_UsesVram()
    {
        var gpu = new LMSupply.Runtime.GpuInfo
        {
            Vendor = LMSupply.Runtime.GpuVendor.Nvidia,
            TotalMemoryBytes = 24 * GB,
            FreeMemoryBytes = 22 * GB,
        };
        var budget = GgufModelDownloader.BuildSelectionBudget(
            cpuBackend: false, gpu, systemRamBytes: 32 * GB, contextLength: 4096, out var vramOnly);

        vramOnly.Should().BeTrue();
        budget.VramBytes.Should().Be(gpu.EffectiveAvailableBytes!.Value);
        budget.RamBytes.Should().Be(32 * GB);
    }
}
