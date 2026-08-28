using AwesomeAssertions;
using LMSupply.Generator.Internal.Llama;
using LMSupply.Runtime;
using Xunit;

namespace LMSupply.Generator.Tests;

public class GgufModelRegistryTests
{
    [Theory]
    [InlineData("gguf:gemma4-default")]
    [InlineData("gguf:gemma4-fast")]
    [InlineData("gguf:gemma4-quality")]
    [InlineData("gguf:gemma4-balanced")]
    [InlineData("gguf:gemma4-large")]
    [InlineData("gguf:xlarge")]
    [InlineData("gguf:qwen3-fast")]
    [InlineData("gguf:qwen3-default")]
    [InlineData("gguf:qwen3-balanced")]
    public void Resolve_WithPrefixedAlias_ReturnsModelInfo(string alias)
    {
        var result = GgufModelRegistry.Resolve(alias);

        result.Should().NotBeNull();
        result!.RepoId.Should().NotBeNullOrWhiteSpace();
        result.DefaultFile.Should().EndWith(".gguf");
        result.ChatFormat.Should().NotBeNullOrWhiteSpace();
    }

    [Theory]
    [InlineData("gemma4-default")]
    [InlineData("gemma4-fast")]
    [InlineData("gemma4-quality")]
    [InlineData("qwen3-fast")]
    [InlineData("qwen3-balanced")]
    public void Resolve_WithoutPrefix_ReturnsModelInfo(string alias)
    {
        var result = GgufModelRegistry.Resolve(alias);

        result.Should().NotBeNull();
        result!.RepoId.Should().Contain("/");
    }

    [Theory]
    [InlineData("unknown-model")]
    [InlineData("nonexistent")]
    [InlineData("")]
    [InlineData(null)]
    public void Resolve_WithInvalidAlias_ReturnsNull(string? alias)
    {
        var result = GgufModelRegistry.Resolve(alias!);

        result.Should().BeNull();
    }

    [Fact]
    public void GetAllModels_ReturnsNonEmptyList()
    {
        var models = GgufModelRegistry.GetAllModels();

        models.Should().NotBeEmpty();
        models.Should().AllSatisfy(m =>
        {
            m.RepoId.Should().NotBeNullOrWhiteSpace();
            m.DefaultFile.Should().EndWith(".gguf");
            m.ContextLength.Should().BeGreaterThan(0);
        });
    }

    [Fact]
    public void DefaultModel_HasValidConfiguration()
    {
        var model = GgufModelRegistry.Resolve("gguf:gemma4-default");

        model.Should().NotBeNull();
        model!.RepoId.Should().Contain("gemma-4");
        model.ChatFormat.Should().Be("gemma4");
        model.DefaultFile.Should().Contain("Q4_0"); // no K-quant is published for this repo
        model.ContextLength.Should().BeGreaterThanOrEqualTo(4096);
    }

    [Fact]
    public void AllModels_HaveValidChatFormats()
    {
        var validFormats = new[] { "chatml", "gemma", "gemma4", "phi3" };
        var models = GgufModelRegistry.GetAllModels();

        models.Should().AllSatisfy(m =>
        {
            validFormats.Should().Contain(m.ChatFormat,
                $"Model {m.DisplayName} has unexpected chat format: {m.ChatFormat}");
        });
    }

    [Fact]
    public void GetAliases_ReturnsExpectedAliases()
    {
        var aliases = GgufModelRegistry.GetAliases();

        aliases.Should().Contain("gguf:gemma4-default");
        aliases.Should().Contain("gguf:gemma4-fast");
        aliases.Should().Contain("gguf:gemma4-quality");
        aliases.Should().Contain("gguf:gemma4-balanced");
        aliases.Should().Contain("gguf:gemma4-large");
        aliases.Should().Contain("gguf:xlarge");
        aliases.Should().Contain("gguf:qwen3-fast");
        aliases.Should().Contain("gguf:qwen3-default");
        aliases.Should().Contain("gguf:qwen3-balanced");
        aliases.Should().Contain("gguf:qwen3-quality");
        aliases.Should().Contain("gguf:qwen3-large");
    }

    [Theory]
    [InlineData("gguf:gemma4-fast")]
    [InlineData("gguf:gemma4-default")]
    [InlineData("gguf:gemma4-balanced")]
    [InlineData("gguf:gemma4-quality")]
    [InlineData("gguf:gemma4-large")]
    public void RegisteredGemmaModels_HaveArchitectureFields(string alias)
    {
        var model = GgufModelRegistry.Resolve(alias);

        model.Should().NotBeNull();
        model!.NumLayers.Should().BeGreaterThan(0,
            because: $"{alias} must declare NumLayers for KV cache budgeting");
        model.HiddenSize.Should().BeGreaterThan(0,
            because: $"{alias} must declare HiddenSize for KV cache budgeting");
    }

    [Fact]
    public void GetAutoSelection_ReturnsResultWithCandidatesAndReason()
    {
        var gpu = new GpuInfo
        {
            Vendor = GpuVendor.Nvidia,
            DeviceName = "Test GPU",
            TotalMemoryBytes = 12L * 1024 * 1024 * 1024,
            FreeMemoryBytes = 11L * 1024 * 1024 * 1024,
        };

        var result = GgufModelRegistry.GetAutoSelection(gpu);

        result.Should().NotBeNull();
        result.Selected.Should().NotBeNull();
        result.AvailableVramBytes.Should().BeGreaterThan(0);
        result.BudgetContextLength.Should().BeGreaterThan(0);
        result.Reason.Should().BeOneOf(
            ModelSelectionReason.Fits,
            ModelSelectionReason.FallbackToSmallest);
        result.Candidates.Should().NotBeEmpty();
        result.Candidates.Should().AllSatisfy(c =>
        {
            c.Model.Should().NotBeNull();
            c.WeightsBytes.Should().BeGreaterThan(0);
            c.KvCacheBytes.Should().BeGreaterThan(0);
            c.TotalBytes.Should().Be(c.WeightsBytes + c.KvCacheBytes);
        });
    }

    [Fact]
    public void GetAutoSelection_4GBVram_SelectsFast()
    {
        // 4GB card, 3GB free.
        // Windows (≤4GB, low-VRAM): budget = min(4×0.75, 3×0.95) = min(3.0, 2.85) = 2.85GB.
        // Linux: budget = min(4×0.85, 3×0.95) = min(3.4, 2.85) = 2.85GB.
        // qwen3-fast (1.5GB + ~0.75GB KV = 2.25GB) fits on both platforms.
        var gpu = new GpuInfo
        {
            Vendor = GpuVendor.Nvidia,
            DeviceName = "NVIDIA RTX 4060 Laptop GPU",
            TotalMemoryBytes = 4L * 1024 * 1024 * 1024,
            FreeMemoryBytes = 3L * 1024 * 1024 * 1024,
        };

        var result = GgufModelRegistry.GetAutoSelection(gpu);

        result.Selected.AliasName.Should().Be("gguf:qwen3-fast");
        result.Reason.Should().Be(ModelSelectionReason.Fits);
    }

    [Fact]
    public void GetAutoSelection_KvCacheCountedInBudget()
    {
        // 6.5GB total, 5.5GB free (non-low-VRAM card, margin 15%).
        // totalCap = 6.5 × 0.85 = 5.525GB, freeCap = 5.5 × 0.95 = 5.225GB → budget = 5.225GB.
        // qwen3-balanced (5.0GB + ~2.25GB KV ≈ 7.25GB) does NOT fit.
        // qwen3-default (3.0GB + ~1.25GB KV ≈ 4.25GB) fits.
        var gpu = new GpuInfo
        {
            Vendor = GpuVendor.Nvidia,
            DeviceName = "Test 6.5GB",
            TotalMemoryBytes = (long)(6.5 * 1024L * 1024 * 1024),
            FreeMemoryBytes = (long)(5.5 * 1024L * 1024 * 1024),
        };

        var result = GgufModelRegistry.GetAutoSelection(gpu);

        result.Selected.AliasName.Should().Be("gguf:qwen3-default",
            because: "with KV cache @ 4096 included, qwen3-balanced (7.25GB) exceeds the 5.225GB budget");
        result.Reason.Should().Be(ModelSelectionReason.Fits);
    }

    [Fact]
    public void Resolve_PopulatesAliasName()
    {
        var model = GgufModelRegistry.Resolve("gguf:gemma4-default");
        model.Should().NotBeNull();
        model!.AliasName.Should().Be("gguf:gemma4-default");
    }

    [Fact]
    public void GetAutoModel_LargeVram_SelectsLargestFitting()
    {
        var gpu = new GpuInfo
        {
            Vendor = GpuVendor.Nvidia,
            TotalMemoryBytes = 24L * 1024 * 1024 * 1024,
            FreeMemoryBytes = 22L * 1024 * 1024 * 1024
        };

        var model = GgufModelRegistry.GetAutoModel(gpu);
        model.Should().NotBeNull();
    }

    [Fact]
    public void GetAutoModel_LargeVram_SelectsQuality()
    {
        var gpu = new GpuInfo
        {
            Vendor = GpuVendor.Nvidia,
            TotalMemoryBytes = 24L * 1024 * 1024 * 1024,
            FreeMemoryBytes = 24L * 1024 * 1024 * 1024
        };
        // 24GB × 0.85 = 20.4GB budget. Auto pool (qwen3-*):
        // - qwen3-large excluded from pool; not considered
        // - qwen3-quality (17.7GB + ~1.25GB KV = 18.95GB) → fits
        var model = GgufModelRegistry.GetAutoModel(gpu);
        model.ParameterCount.Should().Be(35_000_000_000);
    }

    [Fact]
    public void GetAutoModel_12GBVram_SelectsBalanced()
    {
        var gpu = new GpuInfo
        {
            Vendor = GpuVendor.Nvidia,
            TotalMemoryBytes = 12L * 1024 * 1024 * 1024,
            FreeMemoryBytes = 11L * 1024 * 1024 * 1024
        };
        // budget = min(12 × 0.85, 11 × 0.95) = min(10.2GB, 10.45GB) = 10.2GB (totalCap binding).
        // qwen3-balanced (5.0GB Q4_K_M + ~2.25GB KV ≈ 7.25GB) fits. qwen3-quality (~19GB) doesn't fit.
        var model = GgufModelRegistry.GetAutoModel(gpu);
        model.QuantizationType.Should().Be("Q4_K_M");
        model.ParameterCount.Should().Be(8_000_000_000);
    }

    [Fact]
    public void GetAutoModel_6GBVram_SelectsDefault()
    {
        var gpu = new GpuInfo
        {
            Vendor = GpuVendor.Nvidia,
            TotalMemoryBytes = 6L * 1024 * 1024 * 1024,
            FreeMemoryBytes = 5L * 1024 * 1024 * 1024
        };
        // Windows (≤6GB, low-VRAM margin 25%): budget = min(6×0.75, 5×0.95) = min(4.5, 4.75) = 4.5GB.
        // Linux (margin 15%): budget = min(6×0.85, 5×0.95) = min(5.1, 4.75) = 4.75GB.
        // qwen3-default (3.0GB + ~1.25GB KV ≈ 4.25GB) fits on both platforms.
        var model = GgufModelRegistry.GetAutoModel(gpu);
        model.QuantizationType.Should().Be("Q4_K_M");
        model.ParameterCount.Should().Be(4_000_000_000);
    }

    [Fact]
    public void GetAutoModel_8GBVram_SelectsDefault()
    {
        // budget = min(8 × 0.85, 7.5 × 0.95) = min(6.8, 7.125) = 6.8GB (totalCap is binding here).
        // - qwen3-default (3.0GB + ~1.25GB KV ≈ 4.25GB) fits
        // - qwen3-balanced (5.0GB + ~2.25GB KV ≈ 7.25GB) does NOT fit
        var gpu = new GpuInfo
        {
            Vendor = GpuVendor.Nvidia,
            TotalMemoryBytes = 8L * 1024 * 1024 * 1024,
            FreeMemoryBytes = (long)(7.5 * 1024 * 1024 * 1024)
        };
        var model = GgufModelRegistry.GetAutoModel(gpu);
        model.QuantizationType.Should().Be("Q4_K_M");
        model.ParameterCount.Should().Be(4_000_000_000);
    }

    [Fact]
    public void GetAutoModel_8GBVram_LowFree_SelectsDefault()
    {
        // 8GB total but only 6GB free (external process consuming VRAM).
        // totalCap = 8 × 0.85 = 6.8GB, freeCap = 6 × 0.95 = 5.7GB → budget = 5.7GB.
        // qwen3-default (3.0GB + ~1.25GB KV ≈ 4.25GB) fits within 5.7GB budget.
        var gpu = new GpuInfo
        {
            Vendor = GpuVendor.Nvidia,
            TotalMemoryBytes = 8L * 1024 * 1024 * 1024,
            FreeMemoryBytes = 6L * 1024 * 1024 * 1024
        };
        var model = GgufModelRegistry.GetAutoModel(gpu);
        model.ParameterCount.Should().Be(4_000_000_000,
            because: "qwen3-default (4.25GB total) fits within the 5.7GB budget");
    }

    [Fact]
    public void GetAutoModel_TinyVram_SelectsSmallest()
    {
        var gpu = new GpuInfo
        {
            Vendor = GpuVendor.Intel,
            TotalMemoryBytes = 2L * 1024 * 1024 * 1024
        };
        // 2GB × 0.85 = 1.7GB → nothing in auto pool fits → fallback to smallest (qwen3-fast: ~2.25GB total)
        var model = GgufModelRegistry.GetAutoModel(gpu);
        model.AliasName.Should().Be("gguf:qwen3-fast");
    }

    [Fact]
    public void GetAutoModel_CpuOnly_SelectsSmallest()
    {
        var gpu = new GpuInfo
        {
            Vendor = GpuVendor.Unknown
        };
        // CPU-only → 0 VRAM → nothing fits → fallback to smallest in auto pool (qwen3-fast)
        var model = GgufModelRegistry.GetAutoModel(gpu);
        model.AliasName.Should().Be("gguf:qwen3-fast");
    }

    // ─── D3: RAM-aware selection (CPU / low-VRAM, high-RAM machines) ───

    [Fact]
    public void GetAutoSelection_NoVramButAmpleRam_SelectsLargestRamFittingModel()
    {
        // Discovery case: integrated GPU (~0 usable VRAM) but 32GB system RAM.
        // Should NOT fall back to smallest — RAM holds a much larger model on CPU.
        var gpu = new GpuInfo { Vendor = GpuVendor.Intel, TotalMemoryBytes = 128L * 1024 * 1024 };
        var result = GgufModelRegistry.GetAutoSelection(
            gpu, systemRamBytes: 32L * 1024 * 1024 * 1024,
            GgufModelRegistry.DefaultBudgetContextLength, excludeKnownIssues: null);

        result.Reason.Should().Be(ModelSelectionReason.FitsInSystemRam);
        // 32GB - 4GB reserved = 28GB RAM budget; the largest auto-pool model that fits wins.
        result.Selected.AliasName.Should().Be("gguf:qwen3-quality");
        result.AvailableSystemRamBytes.Should().Be(28L * 1024 * 1024 * 1024);
    }

    [Fact]
    public void GetAutoSelection_VramFits_PrefersVramOverRam()
    {
        // A GPU with enough VRAM keeps the VRAM (full-GPU) path even when RAM is also ample.
        var gpu = new GpuInfo
        {
            Vendor = GpuVendor.Nvidia,
            TotalMemoryBytes = 8L * 1024 * 1024 * 1024,
            FreeMemoryBytes = 7L * 1024 * 1024 * 1024
        };
        var result = GgufModelRegistry.GetAutoSelection(
            gpu, systemRamBytes: 64L * 1024 * 1024 * 1024,
            GgufModelRegistry.DefaultBudgetContextLength, excludeKnownIssues: null);

        result.Reason.Should().Be(ModelSelectionReason.Fits);
    }

    [Fact]
    public void GetAutoSelection_LowVramAndLowRam_FallsBackToSmallest()
    {
        // Neither VRAM nor RAM can hold the smallest model → genuine fallback.
        var gpu = new GpuInfo { Vendor = GpuVendor.Unknown };
        var result = GgufModelRegistry.GetAutoSelection(
            gpu, systemRamBytes: 4L * 1024 * 1024 * 1024, // 4GB - 4GB reserved = 0 RAM budget
            GgufModelRegistry.DefaultBudgetContextLength, excludeKnownIssues: null);

        result.Reason.Should().Be(ModelSelectionReason.FallbackToSmallest);
        result.Selected.AliasName.Should().Be("gguf:qwen3-fast");
    }

    [Fact]
    public void GetAutoSelection_GpuOnlyOverload_IgnoresRam_StaysSmallest()
    {
        // The GPU-only overload has no RAM info → RAM fallback disabled → smallest (unchanged behavior).
        var gpu = new GpuInfo { Vendor = GpuVendor.Unknown };
        var result = GgufModelRegistry.GetAutoSelection(gpu);

        result.Reason.Should().Be(ModelSelectionReason.FallbackToSmallest);
        result.AvailableSystemRamBytes.Should().Be(0);
    }

    [Fact]
    public void XLargeModel_HasSplitShardConfiguration()
    {
        var model = GgufModelRegistry.Resolve("gguf:xlarge");
        model.Should().NotBeNull();
        model!.ShardCount.Should().Be(3);
        model.DefaultFile.Should().Contain("-00001-of-00003");
        model.DefaultFile.Should().StartWith("Q4_K_M/");
    }

    [Fact]
    public void GetAutoSelection_AliasName_IsResolvable()
    {
        // Regression for v0.29.0 → v0.30.0 fix: LoadAutoAsync passes
        // selection.Selected.AliasName (not RepoId) downstream so the loader can
        // re-resolve it to the registry entry's DefaultFile. If AliasName is
        // empty or not in the registry, the loader falls through to
        // GgufFileSelector which can pick bf16 on small-VRAM hosts.
        var lowVram = new GpuInfo
        {
            Vendor = GpuVendor.Nvidia,
            TotalMemoryBytes = 4L * 1024 * 1024 * 1024,
            FreeMemoryBytes = 3L * 1024 * 1024 * 1024,
        };
        var highVram = new GpuInfo
        {
            Vendor = GpuVendor.Nvidia,
            TotalMemoryBytes = 24L * 1024 * 1024 * 1024,
            FreeMemoryBytes = 22L * 1024 * 1024 * 1024,
        };

        foreach (var gpu in new[] { lowVram, highVram })
        {
            var selection = GgufModelRegistry.GetAutoSelection(gpu);
            selection.Selected.AliasName.Should().NotBeNullOrEmpty(
                because: "LoadAutoAsync needs an alias to round-trip through Resolve");

            var roundTripped = GgufModelRegistry.Resolve(selection.Selected.AliasName);
            roundTripped.Should().NotBeNull(
                because: "the alias from GetAutoSelection must be resolvable via Resolve");
            roundTripped!.DefaultFile.Should().Be(selection.Selected.DefaultFile,
                because: "round-trip must preserve the exact DefaultFile to avoid bf16 fallback");
        }
    }

    [Theory]
    [InlineData("gguf:gemma4-default", true)]
    [InlineData("gguf:gemma4-fast", true)]
    [InlineData("gguf:gemma4-quality", true)]
    [InlineData("gguf:xlarge", true)]
    [InlineData("default", false)] // Plain aliases are reserved for ONNX
    [InlineData("fast", false)]    // Plain aliases are reserved for ONNX
    [InlineData("unknown", false)]
    public void IsAlias_ReturnsCorrectResult(string value, bool expected)
    {
        var result = GgufModelRegistry.IsAlias(value);

        result.Should().Be(expected);
    }

    [Theory]
    [InlineData("gguf:gemma4-fast")]
    [InlineData("gguf:gemma4-default")]
    [InlineData("gguf:gemma4-balanced")]
    [InlineData("gguf:gemma4-quality")]
    [InlineData("gguf:gemma4-large")]
    public void Gemma4Models_HaveToolUseKnownIssue(string alias)
    {
        var model = GgufModelRegistry.Resolve(alias);

        model.Should().NotBeNull();
        model!.KnownIssues.Should().Contain(GgufModelKnownIssues.ToolUseUnreliableQ4,
            because: $"{alias} is a Gemma 4 Q4_K_M model affected by llama.cpp #21375/#21882");
        model.KnownIssues.Should().Contain(GgufModelKnownIssues.InstructionFollowingUnreliableQ4);
    }

    [Fact]
    public void NonGemma4Models_HaveNoKnownIssues()
    {
        var model = GgufModelRegistry.Resolve("gguf:xlarge");

        model.Should().NotBeNull();
        model!.KnownIssues.Should().BeEmpty(
            because: "gguf:xlarge (Qwen 3.5) has no known llama.cpp compatibility issues");
    }

    [Fact]
    public void GetAutoSelection_ExcludeToolUseUnreliable_FiltersOutAllGemma4()
    {
        var gpu = new GpuInfo
        {
            Vendor = GpuVendor.Nvidia,
            TotalMemoryBytes = 24L * 1024 * 1024 * 1024,
            FreeMemoryBytes = 24L * 1024 * 1024 * 1024,
        };

        var result = GgufModelRegistry.GetAutoSelection(
            gpu,
            GgufModelRegistry.DefaultBudgetContextLength,
            excludeKnownIssues: [GgufModelKnownIssues.ToolUseUnreliableQ4]);

        result.Selected.KnownIssues.Should().NotContain(GgufModelKnownIssues.ToolUseUnreliableQ4,
            because: "excluded models should not be selected");
        result.Selected.ChatFormat.Should().NotBe("gemma4",
            because: "all Gemma 4 entries are excluded");
    }

    [Fact]
    public void GetAutoSelection_NullExcludeList_BehavesLikeOriginalOverload()
    {
        var gpu = new GpuInfo
        {
            Vendor = GpuVendor.Nvidia,
            TotalMemoryBytes = 24L * 1024 * 1024 * 1024,
            FreeMemoryBytes = 24L * 1024 * 1024 * 1024,
        };

        var withNull = GgufModelRegistry.GetAutoSelection(gpu, GgufModelRegistry.DefaultBudgetContextLength, null);
        var withoutParam = GgufModelRegistry.GetAutoSelection(gpu, GgufModelRegistry.DefaultBudgetContextLength);

        withNull.Selected.AliasName.Should().Be(withoutParam.Selected.AliasName);
        withNull.Reason.Should().Be(withoutParam.Reason);
    }

    [Theory]
    [InlineData("gguf:qwen3-fast")]
    [InlineData("gguf:qwen3-default")]
    [InlineData("gguf:qwen3-balanced")]
    [InlineData("gguf:qwen3-quality")]
    [InlineData("gguf:qwen3-large")]
    public void Qwen3Models_HaveValidChatFormat(string alias)
    {
        var model = GgufModelRegistry.Resolve(alias);

        model.Should().NotBeNull();
        model!.ChatFormat.Should().Be("chatml",
            because: $"{alias} uses ChatML format");
    }

    [Theory]
    [InlineData("gguf:qwen3-default")]
    [InlineData("gguf:qwen3-quality")]
    [InlineData("gguf:qwen3-large")]
    public void Qwen3ThinkingModels_HaveThinkingEnabledByDefaultTag(string alias)
    {
        var model = GgufModelRegistry.Resolve(alias);

        model.Should().NotBeNull();
        model!.KnownIssues.Should().Contain(GgufModelKnownIssues.ThinkingEnabledByDefault,
            because: $"{alias} generates <think> blocks by default");
    }

    [Theory]
    [InlineData("gguf:qwen3-fast")]
    [InlineData("gguf:qwen3-balanced")]
    public void Qwen3NonThinkingModels_HaveNoThinkingTag(string alias)
    {
        var model = GgufModelRegistry.Resolve(alias);

        model.Should().NotBeNull();
        model!.KnownIssues.Should().NotContain(GgufModelKnownIssues.ThinkingEnabledByDefault,
            because: $"{alias} does not generate think blocks by default");
    }

    [Fact]
    public void Qwen3AutoPool_ContainsExactlyFourAliases()
    {
        var pool = GgufModelRegistry.GetAutoSelectionAliases();

        pool.Should().HaveCount(4);
        pool.Should().Contain("gguf:qwen3-fast");
        pool.Should().Contain("gguf:qwen3-default");
        pool.Should().Contain("gguf:qwen3-balanced");
        pool.Should().Contain("gguf:qwen3-quality");
        pool.Should().NotContain("gguf:qwen3-large",
            because: "qwen3-large (23.35GB) exceeds the 24GB × 85% = 20.4GB budget threshold");
    }

    [Fact]
    public void GetAutoSelection_ExcludeThinking_SelectsNonThinkingModel()
    {
        // 24GB card: budget = 20.4GB. Pool after excluding ThinkingEnabledByDefault:
        // qwen3-fast (~2.25GB) and qwen3-balanced (~7.25GB) remain.
        // qwen3-balanced is larger and fits → selected.
        var gpu = new GpuInfo
        {
            Vendor = GpuVendor.Nvidia,
            TotalMemoryBytes = 24L * 1024 * 1024 * 1024,
            FreeMemoryBytes = 24L * 1024 * 1024 * 1024,
        };

        var result = GgufModelRegistry.GetAutoSelection(
            gpu,
            GgufModelRegistry.DefaultBudgetContextLength,
            excludeKnownIssues: [GgufModelKnownIssues.ThinkingEnabledByDefault]);

        result.Selected.AliasName.Should().Be("gguf:qwen3-balanced",
            because: "qwen3-balanced is the largest non-thinking model that fits in 20.4GB");
        result.Selected.KnownIssues.Should().NotContain(GgufModelKnownIssues.ThinkingEnabledByDefault);
    }

    [Theory]
    [InlineData("gguf:qwen3-fast")]
    [InlineData("gguf:qwen3-balanced")]
    public void Qwen3FastAndBalanced_HaveNoKnownIssues(string alias)
    {
        var model = GgufModelRegistry.Resolve(alias);

        model.Should().NotBeNull();
        model!.KnownIssues.Should().BeEmpty(
            because: $"{alias} has no known compatibility issues");
    }

    [Theory]
    [InlineData("gguf:qwen3-fast")]
    [InlineData("gguf:qwen3-default")]
    [InlineData("gguf:qwen3-balanced")]
    [InlineData("gguf:qwen3-quality")]
    [InlineData("gguf:qwen3-large")]
    public void Qwen3Models_HaveArchitectureFields(string alias)
    {
        var model = GgufModelRegistry.Resolve(alias);

        model.Should().NotBeNull();
        model!.NumLayers.Should().BeGreaterThan(0,
            because: $"{alias} must declare NumLayers for KV cache budgeting");
        model.HiddenSize.Should().BeGreaterThan(0,
            because: $"{alias} must declare HiddenSize for KV cache budgeting");
    }

    [Theory]
    [InlineData("gguf:fast")]
    [InlineData("gguf:default")]
    [InlineData("gguf:balanced")]
    [InlineData("gguf:quality")]
    [InlineData("gguf:large")]
    public void Resolve_OldUnprefixedTierAliases_ReturnsNull(string alias)
    {
        var result = GgufModelRegistry.Resolve(alias);

        result.Should().BeNull(
            because: $"old unprefixed alias '{alias}' was removed in favour of gguf:gemma4-* and gguf:qwen3-*");
    }
}
