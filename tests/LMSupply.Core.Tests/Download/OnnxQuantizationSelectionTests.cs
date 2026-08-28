using AwesomeAssertions;
using LMSupply.Core.Download;
using LMSupply.Hardware;

namespace LMSupply.Core.Tests.Download;

public class OnnxQuantizationSelectionTests
{
    [Fact]
    public void SelectBestQuantizationVariants_Quant4Priority_SelectsInt4File()
    {
        var candidates = new List<RepoFile>
        {
            new() { Path = "onnx/model.onnx", Type = "file", Size = 4_000_000_000L },
            new() { Path = "onnx/model_int8.onnx", Type = "file", Size = 1_000_000_000L },
            new() { Path = "onnx/model_int4.onnx", Type = "file", Size = 500_000_000L },
        };

        var prefs = ModelPreferences.ForTier(PerformanceTier.Low);  // Quant4 우선
        var selected = ModelDiscoveryService.SelectBestVariantsForTest(candidates, prefs);

        selected.Should().Contain("onnx/model_int4.onnx");
        selected.Should().NotContain("onnx/model.onnx");
    }

    [Fact]
    public void SelectBestQuantizationVariants_DefaultPriority_SelectsFp32()
    {
        var candidates = new List<RepoFile>
        {
            new() { Path = "onnx/model.onnx", Type = "file", Size = 4_000_000_000L },
            new() { Path = "onnx/model_int4.onnx", Type = "file", Size = 500_000_000L },
        };

        var prefs = ModelPreferences.ForTier(PerformanceTier.Ultra);  // Default 우선
        var selected = ModelDiscoveryService.SelectBestVariantsForTest(candidates, prefs);

        selected.Should().Contain("onnx/model.onnx");
    }

    [Fact]
    public void SelectBestQuantizationVariants_ExtendedSuffixes_HighTierSelectsQuant8()
    {
        // _uint8 → Quant8, _q4 → Quant4 are now recognized suffixes.
        // High tier priority: [Fp16, Default, Quant8, Quant4]
        // → Fp16: none → Default: none (both have suffix) → Quant8: _uint8 matches!
        var candidates = new List<RepoFile>
        {
            new() { Path = "onnx/model_uint8.onnx", Type = "file", Size = 800_000_000L },
            new() { Path = "onnx/model_q4.onnx",   Type = "file", Size = 2_000_000_000L },
        };

        var prefs = ModelPreferences.ForTier(PerformanceTier.High);
        var selected = ModelDiscoveryService.SelectBestVariantsForTest(candidates, prefs);

        selected.Should().HaveCount(1);
        selected.Should().Contain("onnx/model_uint8.onnx");
    }

    [Fact]
    public void SelectBestQuantizationVariants_ExtendedSuffixes_LowTierPrefersQuant4()
    {
        // Low tier priority: [Quant4, Quant8, Fp16, Default]
        // → Quant4: _q4 matches!
        var candidates = new List<RepoFile>
        {
            new() { Path = "onnx/model_uint8.onnx", Type = "file", Size = 800_000_000L },
            new() { Path = "onnx/model_q4.onnx",   Type = "file", Size = 2_000_000_000L },
        };

        var prefs = ModelPreferences.ForTier(PerformanceTier.Low);
        var selected = ModelDiscoveryService.SelectBestVariantsForTest(candidates, prefs);

        selected.Should().HaveCount(1);
        selected.Should().Contain("onnx/model_q4.onnx");
    }

    [Fact]
    public void SelectBestQuantizationVariants_ExtendedSuffixes_MatchEvenWithZeroSize()
    {
        // Suffix matching takes precedence over size-based fallback,
        // so Size=0 files are selected if they match the quantization priority.
        var candidates = new List<RepoFile>
        {
            new() { Path = "onnx/model_uint8.onnx", Type = "file", Size = 0L },
            new() { Path = "onnx/model_q4.onnx",   Type = "file", Size = 2_000_000_000L },
        };

        var prefs = ModelPreferences.ForTier(PerformanceTier.High); // Quant8 before Quant4
        var selected = ModelDiscoveryService.SelectBestVariantsForTest(candidates, prefs);

        // _uint8 matches Quant8 by suffix even with Size=0
        selected.Should().HaveCount(1);
        selected.Should().Contain("onnx/model_uint8.onnx");
    }

    [Fact]
    public void SelectBestQuantizationVariants_UnrecognizedSuffix_TreatedAsDefaultIfNoKnownSuffix()
    {
        // Files with unrecognized suffixes (_abc, _xyz) are not stripped by GetBaseModelName,
        // so they get distinct base names and each forms its own group.
        // Within each single-file group, the Default check succeeds (no known quant suffix).
        var candidates = new List<RepoFile>
        {
            new() { Path = "onnx/model_abc.onnx", Type = "file", Size = 800_000_000L },
            new() { Path = "onnx/model_xyz.onnx", Type = "file", Size = 2_000_000_000L },
        };

        var prefs = ModelPreferences.ForTier(PerformanceTier.High);
        var selected = ModelDiscoveryService.SelectBestVariantsForTest(candidates, prefs);

        // Each file is its own group (unrecognized suffixes aren't stripped)
        // Default matches both → both selected
        selected.Should().HaveCount(2);
        selected.Should().Contain("onnx/model_abc.onnx");
        selected.Should().Contain("onnx/model_xyz.onnx");
    }
}
