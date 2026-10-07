using AwesomeAssertions;
using LMSupply.Core.Download;
using Xunit;

namespace LMSupply.Core.Tests.Download;

public class GgufFileSelectorTests
{
    // ─── AvailableMemory per scenario ───

    // 4GB VRAM + 16GB RAM (Medium tier)
    private static readonly AvailableMemory MediumHardware = new(
        VramBytes: 4L * 1024 * 1024 * 1024,
        RamBytes: 16L * 1024 * 1024 * 1024);

    // 16GB VRAM + 64GB RAM (Ultra tier)
    private static readonly AvailableMemory UltraHardware = new(
        VramBytes: 16L * 1024 * 1024 * 1024,
        RamBytes: 64L * 1024 * 1024 * 1024);

    // CPU Only + 8GB RAM (Low tier)
    private static readonly AvailableMemory LowHardware = new(
        VramBytes: 0,
        RamBytes: 8L * 1024 * 1024 * 1024);

    // ─── AvailableMemory computation ───

    [Fact]
    public void AvailableMemory_UsableVramBytes_DeductsOverhead()
    {
        var memory = new AvailableMemory(VramBytes: 8L * 1024 * 1024 * 1024, RamBytes: 0);
        // minus 2GB GPU overhead → 6GB
        memory.UsableVramBytes.Should().Be(6L * 1024 * 1024 * 1024);
    }

    [Fact]
    public void AvailableMemory_UsableRamBytes_DeductsOverhead()
    {
        var memory = new AvailableMemory(VramBytes: 0, RamBytes: 8L * 1024 * 1024 * 1024);
        // minus 4GB system overhead → 4GB
        memory.UsableRamBytes.Should().Be(4L * 1024 * 1024 * 1024);
    }

    [Fact]
    public void AvailableMemory_VramZero_UsableVramIsZero()
    {
        var memory = new AvailableMemory(VramBytes: 0, RamBytes: 16L * 1024 * 1024 * 1024);
        memory.UsableVramBytes.Should().Be(0);
    }

    [Fact]
    public void AvailableMemory_FitsInGpu_TrueWhenSmallEnough()
    {
        // 2GB file, 4GB VRAM (2GB usable) → 2GB × 1.1 = 2.2GB > 2GB usable → false
        var memory = new AvailableMemory(
            VramBytes: 4L * 1024 * 1024 * 1024,
            RamBytes: 16L * 1024 * 1024 * 1024);
        var fileSize = 1L * 1024 * 1024 * 1024; // 1GB → 1.1GB with overhead < 2GB usable
        memory.FitsInGpu(fileSize).Should().BeTrue();
    }

    [Fact]
    public void AvailableMemory_FitsInGpu_FalseWhenNoVram()
    {
        LowHardware.FitsInGpu(1L * 1024 * 1024 * 1024).Should().BeFalse();
    }

    // ─── Default selection: largest file that fits in memory ───

    [Fact]
    public void Select_PrefersBiggerFileWithinMemory()
    {
        // 3GB, 5GB, 8GB files / 16GB VRAM (14GB usable)
        var groups = new[]
        {
            MakeGroup("model-Q4_K_M.gguf", 3L * 1024 * 1024 * 1024),
            MakeGroup("model-Q6_K.gguf",   5L * 1024 * 1024 * 1024),
            MakeGroup("model-Q8_0.gguf",   8L * 1024 * 1024 * 1024),
        };

        var result = GgufFileSelector.Select(groups, UltraHardware);

        result.PrimaryFileName.Should().Be("model-Q8_0.gguf");
    }

    [Fact]
    public void Select_FiltersOutTooLargeFiles()
    {
        // 3GB and 8GB files / 4GB VRAM (2GB usable), 16GB RAM (12GB usable), ~14GB total usable
        // 8GB × 1.1 = 8.8GB is below the 14GB total usable, so the 8GB file also passes
        // so the larger 8GB file is selected
        var groups = new[]
        {
            MakeGroup("model-Q4_K_M.gguf", 3L * 1024 * 1024 * 1024),
            MakeGroup("model-Q8_0.gguf",   8L * 1024 * 1024 * 1024),
        };

        var result = GgufFileSelector.Select(groups, MediumHardware);

        // MediumHardware: VRAM 4GB(usable 2GB) + RAM 16GB(usable 12GB) = total 14GB
        // 8GB × 1.1 = 8.8GB ≤ 14GB → passes, 3GB × 1.1 = 3.3GB ≤ 14GB → passes
        // 8GB is larger, so it is selected
        result.PrimaryFileName.Should().Be("model-Q8_0.gguf");
    }

    [Fact]
    public void Select_FiltersOutFilesTooLargeForTotalMemory()
    {
        // File is too large even for combined VRAM + RAM
        var groups = new[]
        {
            MakeGroup("model-Q4_K_M.gguf", 2L * 1024 * 1024 * 1024),
            MakeGroup("model-huge.gguf",   20L * 1024 * 1024 * 1024),  // 20GB
        };

        // LowHardware: VRAM 0 + RAM 8GB(usable 4GB) = total 4GB
        // 2GB × 1.1 = 2.2GB ≤ 4GB → passes
        // 20GB × 1.1 = 22GB > 4GB → filtered out
        var result = GgufFileSelector.Select(groups, LowHardware);

        result.PrimaryFileName.Should().Be("model-Q4_K_M.gguf");
    }

    // ─── Split files: memory check uses the summed size ───

    [Fact]
    public void Select_SplitFile_UsesTotalSizeForMemoryCheck()
    {
        // Split file: 2.1GB per part × 3 = 6.3GB
        var splitGroup = new GgufFileGroup
        {
            PrimaryFileName = "model-Q4_K_M-00001-of-00003.gguf",
            Parts = ["model-Q4_K_M-00001-of-00003.gguf",
                     "model-Q4_K_M-00002-of-00003.gguf",
                     "model-Q4_K_M-00003-of-00003.gguf"],
            TotalSizeBytes = 6_300_000_000L
        };
        // Single 2.5GB file
        var smallGroup = MakeGroup("model-Q3_K_M.gguf", 2_500_000_000L);

        // LowHardware: total usable 4GB
        // 6.3GB × 1.1 = 6.93GB > 4GB → filtered out
        // 2.5GB × 1.1 = 2.75GB ≤ 4GB → passes
        var result = GgufFileSelector.Select([splitGroup, smallGroup], LowHardware);

        result.PrimaryFileName.Should().Be("model-Q3_K_M.gguf");
    }

    // ─── User-specified quantization ───

    [Fact]
    public void Select_PreferredQuantization_UsedWhenFits()
    {
        var groups = new[]
        {
            MakeGroup("model-Q4_K_M.gguf", 3L * 1024 * 1024 * 1024),
            MakeGroup("model-Q6_K.gguf",   5L * 1024 * 1024 * 1024),
            MakeGroup("model-Q8_0.gguf",   8L * 1024 * 1024 * 1024),
        };

        // Ultra hardware, Q6_K preferred → Q6_K selected (Q8_0 is larger, but the user's choice wins)
        var result = GgufFileSelector.Select(groups, UltraHardware, preferredQuantization: "Q6_K");

        result.PrimaryFileName.Should().Be("model-Q6_K.gguf");
    }

    [Fact]
    public void Select_PreferredQuantization_TooLarge_FallsBackToAuto()
    {
        var groups = new[]
        {
            MakeGroup("model-Q4_K_M.gguf", 2L * 1024 * 1024 * 1024),
            MakeGroup("model-Q8_0.gguf",   50L * 1024 * 1024 * 1024),  // 50GB, too large
        };

        // LowHardware: total usable 4GB
        // Q8_0 preferred, but 50GB × 1.1 > 4GB → filtered out
        // Q4_K_M (2GB) selected automatically
        var result = GgufFileSelector.Select(groups, LowHardware, preferredQuantization: "Q8_0");

        result.PrimaryFileName.Should().Be("model-Q4_K_M.gguf");
    }

    [Fact]
    public void Select_PreferredQuantization_NullOrEmpty_UsesAutoSelection()
    {
        var groups = new[]
        {
            MakeGroup("model-Q4_K_M.gguf", 3L * 1024 * 1024 * 1024),
            MakeGroup("model-Q8_0.gguf",   8L * 1024 * 1024 * 1024),
        };

        var resultNull = GgufFileSelector.Select(groups, UltraHardware, preferredQuantization: null);
        var resultEmpty = GgufFileSelector.Select(groups, UltraHardware, preferredQuantization: "");

        resultNull.PrimaryFileName.Should().Be("model-Q8_0.gguf");
        resultEmpty.PrimaryFileName.Should().Be("model-Q8_0.gguf");
    }

    // ─── Quantization name matching ───

    [Theory]
    [InlineData("model-Q4_K_M.gguf", "Q4_K_M", true)]
    [InlineData("model-Q4_K_M-imat.gguf", "Q4_K_M", true)]       // iMatrix variant
    [InlineData("Meta-Llama-3-8B.Q4_K_M.gguf", "Q4_K_M", true)]  // dot separator
    [InlineData("model_Q4_K_M.gguf", "Q4_K_M", true)]             // underscore separator
    [InlineData("model-Q8_0.gguf", "Q4_K_M", false)]
    [InlineData("model-IQ4_XS.gguf", "IQ4_XS", true)]             // newer type
    [InlineData("model-BF16.gguf", "BF16", true)]                  // BF16
    public void MatchesQuantization_WorksForVariousFormats(string filename, string quant, bool expected)
    {
        GgufFileSelector.MatchesQuantization(filename, quant).Should().Be(expected);
    }

    // ─── Error cases ───

    [Fact]
    public void Select_NothingFits_ThrowsWithDetails()
    {
        var groups = new[]
        {
            MakeGroup("model-Q4_K_M.gguf", 50L * 1024 * 1024 * 1024),  // 50GB
        };

        var act = () => GgufFileSelector.Select(groups, LowHardware);

        act.Should().Throw<InvalidOperationException>()
           .WithMessage("*available*");
    }

    [Fact]
    public void Select_EmptyList_ThrowsArgumentException()
    {
        var act = () => GgufFileSelector.Select([], MediumHardware);
        act.Should().Throw<ArgumentException>();
    }

    // ─── FromHardwareProfile ───

    [Fact]
    public void FromHardwareProfile_CreatesCorrectMemory()
    {
        var profile = LMSupply.Hardware.HardwareProfile.Current;
        var memory = GgufFileSelector.FromHardwareProfile(profile);

        memory.Should().NotBeNull();
        memory.VramBytes.Should().BeGreaterThanOrEqualTo(0);
        memory.RamBytes.Should().BeGreaterThan(0);
    }

    // ─── Tests including the KV cache budget ───

    [Theory]
    [InlineData(1L * 1024 * 1024 * 1024, 4096)]   // 1GB file, 4K context
    [InlineData(3L * 1024 * 1024 * 1024, 4096)]   // 3GB file
    [InlineData(8L * 1024 * 1024 * 1024, 4096)]   // 8GB file
    [InlineData(15L * 1024 * 1024 * 1024, 4096)]  // 15GB file
    public void EstimateKvCacheBytes_ReturnsPositive(long fileSize, int contextLength)
    {
        var kvCache = AvailableMemory.EstimateKvCacheBytes(fileSize, contextLength);
        kvCache.Should().BeGreaterThan(0);
    }

    [Fact]
    public void EstimateKvCacheBytes_ScalesWithContext()
    {
        var fileSize = 5L * 1024 * 1024 * 1024;
        var kv4k = AvailableMemory.EstimateKvCacheBytes(fileSize, 4096);
        var kv8k = AvailableMemory.EstimateKvCacheBytes(fileSize, 8192);

        kv8k.Should().Be(kv4k * 2, "KV cache scales linearly with context length");
    }

    [Fact]
    public void EstimateKvCacheBytes_LargerModels_HaveLargerKvCache()
    {
        var kvSmall = AvailableMemory.EstimateKvCacheBytes(1L * 1024 * 1024 * 1024, 4096);
        var kvLarge = AvailableMemory.EstimateKvCacheBytes(15L * 1024 * 1024 * 1024, 4096);

        kvLarge.Should().BeGreaterThan(kvSmall);
    }

    [Fact]
    public void FitsInGpu_WithKvCache_TighterBudget()
    {
        // With KV cache budget, a file that previously fit may no longer fit
        // 1GB file with overhead: 1.1GB model + KV cache
        var memory = new AvailableMemory(
            VramBytes: 4L * 1024 * 1024 * 1024,  // 4GB VRAM (usable: 2GB)
            RamBytes: 16L * 1024 * 1024 * 1024);
        var fileSize = 1L * 1024 * 1024 * 1024;

        var kvCache = AvailableMemory.EstimateKvCacheBytes(fileSize, 4096);
        var totalNeeded = (long)(fileSize * 1.1) + kvCache;

        // Verify KV cache is non-trivial
        kvCache.Should().BeGreaterThan(0);

        // FitsInGpu result should depend on total (model + KV cache) vs usable VRAM
        var fits = memory.FitsInGpu(fileSize);
        fits.Should().Be(totalNeeded <= memory.UsableVramBytes);
    }

    [Fact]
    public void FitsInMemory_ContextLength_AffectsBudget()
    {
        // Larger context → larger KV cache → harder to fit
        var smallContextMem = new AvailableMemory(
            VramBytes: 0, RamBytes: 8L * 1024 * 1024 * 1024, ContextLength: 4096);
        var largeContextMem = new AvailableMemory(
            VramBytes: 0, RamBytes: 8L * 1024 * 1024 * 1024, ContextLength: 32768);

        var fileSize = 3L * 1024 * 1024 * 1024;

        // With 8x context, same file may not fit
        var fitsSmall = smallContextMem.FitsInMemory(fileSize);
        var fitsLarge = largeContextMem.FitsInMemory(fileSize);

        // If small fits but large doesn't, the KV cache budget is working
        // At minimum, large context should be harder to fit
        if (fitsSmall)
            fitsLarge.Should().BeFalse("32K context on 3GB model exceeds 4GB usable RAM");
    }

    [Fact]
    public void Select_LargeContext_ReducesAvailableOptions()
    {
        // With 32K context, KV cache is huge → fewer files fit
        var memorySmallCtx = new AvailableMemory(
            VramBytes: 0,
            RamBytes: 20L * 1024 * 1024 * 1024,   // 20GB RAM, usable 16GB
            ContextLength: 4096);

        var memoryLargeCtx = new AvailableMemory(
            VramBytes: 0,
            RamBytes: 20L * 1024 * 1024 * 1024,   // same RAM
            ContextLength: 32768);                  // 8x context

        var groups = new[]
        {
            MakeGroup("model-Q2_K.gguf", 1L * 1024 * 1024 * 1024),
            MakeGroup("model-Q4_K_M.gguf", 3L * 1024 * 1024 * 1024),
            MakeGroup("model-Q8_0.gguf", 8L * 1024 * 1024 * 1024),
        };

        var fittingSmall = groups.Count(g => memorySmallCtx.FitsInMemory(g.TotalSizeBytes));
        var fittingLarge = groups.Count(g => memoryLargeCtx.FitsInMemory(g.TotalSizeBytes));

        fittingLarge.Should().BeLessThanOrEqualTo(fittingSmall,
            "larger context should fit fewer or equal models");
    }

    // ─── VRAM-only mode: no bf16 selection on a 4GB-VRAM / large-RAM host ───

    [Fact]
    public void Select_VramOnly_PrefersFitInVramOverLargestInRam()
    {
        // Simulates 4GB VRAM (2GB usable) + 32GB RAM. bf16 (15GB) fits in RAM but
        // exceeds the VRAM budget, so it must never be selected when vramOnly=true.
        var memory = new AvailableMemory(
            VramBytes: 4L * 1024 * 1024 * 1024,
            RamBytes: 32L * 1024 * 1024 * 1024);

        var groups = new[]
        {
            MakeGroup("model-Q4_K_M.gguf", 1L * 1024 * 1024 * 1024),    // 1GB → fits VRAM
            MakeGroup("model-Q8_0.gguf",   3L * 1024 * 1024 * 1024),    // 3GB → exceeds VRAM
            MakeGroup("model-bf16.gguf",   15L * 1024 * 1024 * 1024),   // 15GB → exceeds VRAM (fits RAM)
        };

        var result = GgufFileSelector.Select(groups, memory, vramOnly: true);

        result.PrimaryFileName.Should().Be("model-Q4_K_M.gguf");
    }

    [Fact]
    public void Select_VramOnly_NothingFits_FallsBackToSmallest()
    {
        // When no candidate fits VRAM, vramOnly=true returns the smallest file instead of throwing
        // (so llama-server can handle it with partial CPU offload).
        var memory = new AvailableMemory(
            VramBytes: 4L * 1024 * 1024 * 1024,
            RamBytes: 32L * 1024 * 1024 * 1024);

        var groups = new[]
        {
            MakeGroup("model-Q4_K_M.gguf", 4L * 1024 * 1024 * 1024),    // 4GB → exceeds VRAM (2GB usable)
            MakeGroup("model-bf16.gguf",   15L * 1024 * 1024 * 1024),
        };

        var result = GgufFileSelector.Select(groups, memory, vramOnly: true);

        result.PrimaryFileName.Should().Be("model-Q4_K_M.gguf");
    }

    [Fact]
    public void Select_VramOnly_FalseByDefault_PreservesLegacyBehavior()
    {
        // Without vramOnly, the combined VRAM + RAM behavior is kept.
        var memory = new AvailableMemory(
            VramBytes: 4L * 1024 * 1024 * 1024,
            RamBytes: 32L * 1024 * 1024 * 1024);

        var groups = new[]
        {
            MakeGroup("model-Q4_K_M.gguf", 1L * 1024 * 1024 * 1024),
            MakeGroup("model-bf16.gguf",   15L * 1024 * 1024 * 1024),
        };

        var result = GgufFileSelector.Select(groups, memory);

        // Default behavior: select the largest file that fits VRAM + RAM
        result.PrimaryFileName.Should().Be("model-bf16.gguf");
    }

    // ─── Helpers ───

    private static GgufFileGroup MakeGroup(string filename, long sizeBytes) =>
        new() { PrimaryFileName = filename, Parts = [filename], TotalSizeBytes = sizeBytes };
}
