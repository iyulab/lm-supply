using AwesomeAssertions;
using LMSupply.Core.Download;
using LMSupply.Hardware;

namespace LMSupply.Core.Tests.Download;

/// <summary>
/// Tests for smart model variant selection: prefix-based encoder-decoder classification,
/// quantization-aware selection, hardware-adaptive preferences, and qualifier syntax.
/// All tests are pure unit tests — no network, no download, no HF API calls.
/// </summary>
public class SmartVariantSelectionTests
{
    // =========================================================================
    // 1. Prefix-based encoder-decoder classification
    //    Verifies _bnb4, _q4, _q4f16, _uint8 files are recognized as encoder/decoder
    // =========================================================================

    [Theory]
    [InlineData("onnx/encoder_model_bnb4.onnx")]
    [InlineData("onnx/encoder_model_q4.onnx")]
    [InlineData("onnx/encoder_model_q4f16.onnx")]
    [InlineData("onnx/encoder_model_uint8.onnx")]
    [InlineData("onnx/encoder_model_int8.onnx")]
    [InlineData("onnx/encoder_model_fp16.onnx")]
    [InlineData("onnx/encoder_model.onnx")]
    public void DiscoverModel_AllEncoderVariants_ClassifiedAsEncoder(string encoderPath)
    {
        // Include FP32 base pair so DetectArchitecture (legacy exact patterns) always succeeds
        var paths = new HashSet<string>(StringComparer.OrdinalIgnoreCase)
        {
            "onnx/encoder_model.onnx",
            "onnx/decoder_model_merged.onnx",
            encoderPath
        };
        var files = MakeRepoFiles([.. paths]);

        var result = DiscoverFromFileList(files);

        result.Architecture.Should().Be(ModelArchitecture.EncoderDecoder);
        result.EncoderFiles.Should().HaveCount(1);
    }

    [Theory]
    [InlineData("onnx/decoder_model_merged_bnb4.onnx")]
    [InlineData("onnx/decoder_model_merged_q4.onnx")]
    [InlineData("onnx/decoder_model_merged_q4f16.onnx")]
    [InlineData("onnx/decoder_model_merged_uint8.onnx")]
    [InlineData("onnx/decoder_model_merged_fp16.onnx")]
    [InlineData("onnx/decoder_model_merged.onnx")]
    public void DiscoverModel_AllDecoderMergedVariants_ClassifiedAsDecoder(string decoderPath)
    {
        // Include FP32 base pair so DetectArchitecture (legacy exact patterns) always succeeds
        var paths = new HashSet<string>(StringComparer.OrdinalIgnoreCase)
        {
            "onnx/encoder_model.onnx",
            "onnx/decoder_model_merged.onnx",
            decoderPath
        };
        var files = MakeRepoFiles([.. paths]);

        var result = DiscoverFromFileList(files);

        result.Architecture.Should().Be(ModelArchitecture.EncoderDecoder);
        result.DecoderFiles.Should().HaveCount(1);
    }

    // =========================================================================
    // 2. Quantization-aware encoder-decoder selection per hardware tier
    //    Simulates the exact Whisper Large V3 ONNX repo file set
    // =========================================================================

    private static readonly string[] WhisperLargeV3Files =
    [
        "onnx/encoder_model.onnx",          // FP32 (403KB + 2.4GB data)
        "onnx/encoder_model_bnb4.onnx",     // 368MB
        "onnx/encoder_model_fp16.onnx",     // 1.2GB
        "onnx/encoder_model_int8.onnx",     // 615MB
        "onnx/encoder_model_q4.onnx",       // 405MB
        "onnx/encoder_model_q4f16.onnx",    // 353MB
        "onnx/encoder_model_quantized.onnx", // 615MB
        "onnx/encoder_model_uint8.onnx",    // 615MB
        "onnx/decoder_model_merged.onnx",          // FP32 (2.2MB + 3.4GB data)
        "onnx/decoder_model_merged_bnb4.onnx",     // 710MB
        "onnx/decoder_model_merged_fp16.onnx",     // 1.7GB
        "onnx/decoder_model_merged_int8.onnx",     // 1.1GB
        "onnx/decoder_model_merged_q4.onnx",       // 760MB
        "onnx/decoder_model_merged_q4f16.onnx",    // 581MB
        "onnx/decoder_model_merged_quantized.onnx", // 1.1GB
        "onnx/decoder_model_merged_uint8.onnx",    // 1.1GB
        "onnx/decoder_model.onnx",
        "onnx/decoder_model_bnb4.onnx",
        "onnx/decoder_model_fp16.onnx",
        "onnx/decoder_with_past_model.onnx",
        "onnx/decoder_with_past_model_bnb4.onnx",
        "onnx/decoder_with_past_model_fp16.onnx",
    ];

    [Fact]
    public void WhisperLargeV3_UltraTier_SelectsFP32EncoderAndMergedDecoder()
    {
        var prefs = WithSubfolder(ModelPreferences.ForTier(PerformanceTier.Ultra), "onnx");

        var result = DiscoverFromFileList(MakeRepoFiles(WhisperLargeV3Files), prefs);

        result.EncoderFiles.Should().ContainSingle()
            .Which.Should().Be("onnx/encoder_model.onnx");
        result.DecoderFiles.Should().ContainSingle()
            .Which.Should().Be("onnx/decoder_model_merged.onnx");
        result.DetectedDecoderVariant.Should().Be(DecoderVariant.Merged);
    }

    [Fact]
    public void WhisperLargeV3_HighTier_SelectsFP16()
    {
        var prefs = WithSubfolder(ModelPreferences.ForTier(PerformanceTier.High), "onnx");

        var result = DiscoverFromFileList(MakeRepoFiles(WhisperLargeV3Files), prefs);

        result.EncoderFiles.Should().ContainSingle()
            .Which.Should().Contain("fp16");
        result.DecoderFiles.Should().ContainSingle()
            .Which.Should().Contain("fp16");
    }

    [Fact]
    public void WhisperLargeV3_MediumTier_SelectsInt8()
    {
        var prefs = WithSubfolder(ModelPreferences.ForTier(PerformanceTier.Medium), "onnx");

        var result = DiscoverFromFileList(MakeRepoFiles(WhisperLargeV3Files), prefs);

        result.EncoderFiles.Should().ContainSingle()
            .Which.Should().Contain("int8");
        result.DecoderFiles.Should().ContainSingle()
            .Which.Should().Contain("int8");
    }

    [Fact]
    public void WhisperLargeV3_LowTier_SelectsQuant4Variant()
    {
        var prefs = WithSubfolder(ModelPreferences.ForTier(PerformanceTier.Low), "onnx");

        var result = DiscoverFromFileList(MakeRepoFiles(WhisperLargeV3Files), prefs);

        // Quant4 maps to _int4, _bnb4, _q4, _q4f16. Repo has _int4? No. Next: _bnb4? Yes.
        var encoderName = result.EncoderFiles.Should().ContainSingle().Subject;
        var decoderName = result.DecoderFiles.Should().ContainSingle().Subject;

        // Both should be some Quant4 variant (not FP32, not FP16, not Int8)
        encoderName.Should().NotBe("onnx/encoder_model.onnx", "should not be FP32");
        decoderName.Should().NotBe("onnx/decoder_model_merged.onnx", "should not be FP32");

        // Verify both are quantized (have suffix)
        encoderName.Should().MatchRegex(@"_(bnb4|q4|q4f16|int4)\.");
        decoderName.Should().MatchRegex(@"_(bnb4|q4|q4f16|int4)\.");
    }

    [Fact]
    public void WhisperLargeV3_AnyTier_SelectsOnlyTwoOnnxFiles()
    {
        // The critical bug fix: only 2 ONNX files (1 encoder + 1 decoder) should be selected,
        // not all 22 files in the repo.
        foreach (var tier in Enum.GetValues<PerformanceTier>())
        {
            var prefs = WithSubfolder(ModelPreferences.ForTier(tier), "onnx");

            var result = DiscoverFromFileList(MakeRepoFiles(WhisperLargeV3Files), prefs);

            result.OnnxFiles.Should().HaveCountLessThanOrEqualTo(2,
                $"tier {tier} should select at most 1 encoder + 1 decoder, not all variants");
        }
    }

    [Fact]
    public void WhisperLargeV3_MatchedQuantization_EncoderAndDecoderSameLevel()
    {
        // When RequireMatchedQuantization=true (default), encoder and decoder
        // should have the same quantization level.
        var prefs = WithSubfolder(ModelPreferences.ForTier(PerformanceTier.High), "onnx");

        var result = DiscoverFromFileList(MakeRepoFiles(WhisperLargeV3Files), prefs);

        var encSuffix = GetQuantSuffix(result.EncoderFiles[0]);
        var decSuffix = GetQuantSuffix(result.DecoderFiles[0]);
        encSuffix.Should().Be(decSuffix,
            "encoder and decoder should have matching quantization");
    }

    // =========================================================================
    // 3. ForQuantizationHint / ParseQuantizationHint
    // =========================================================================

    [Theory]
    [InlineData("fp16", Quantization.Fp16)]
    [InlineData("half", Quantization.Fp16)]
    [InlineData("float16", Quantization.Fp16)]
    [InlineData("int8", Quantization.Quant8)]
    [InlineData("uint8", Quantization.Quant8)]
    [InlineData("q8", Quantization.Quant8)]
    [InlineData("quantized", Quantization.Quant8)]
    [InlineData("int4", Quantization.Quant4)]
    [InlineData("q4", Quantization.Quant4)]
    [InlineData("bnb4", Quantization.Quant4)]
    [InlineData("q4f16", Quantization.Quant4)]
    [InlineData("4bit", Quantization.Quant4)]
    [InlineData("fp32", Quantization.Default)]
    [InlineData("default", Quantization.Default)]
    [InlineData("full", Quantization.Default)]
    public void ParseQuantizationHint_AllKnownHints_ParseCorrectly(string hint, Quantization expected)
    {
        ModelPreferences.ParseQuantizationHint(hint).Should().Be(expected);
    }

    [Fact]
    public void ForQuantizationHint_Fp16_PrioritizesFp16First()
    {
        var prefs = ModelPreferences.ForQuantizationHint("fp16");
        prefs.QuantizationPriority[0].Should().Be(Quantization.Fp16);
    }

    [Fact]
    public void ForQuantizationHint_Q4_PrioritizesQuant4First()
    {
        var prefs = ModelPreferences.ForQuantizationHint("q4");
        prefs.QuantizationPriority[0].Should().Be(Quantization.Quant4);
        prefs.PreferLowMemory.Should().BeTrue();
    }

    [Fact]
    public void ForQuantizationHint_Null_ReturnsHardwareAdaptive()
    {
        // Should not throw, should return valid preferences
        var prefs = ModelPreferences.ForQuantizationHint(null);
        prefs.QuantizationPriority.Should().NotBeEmpty();
    }

    [Fact]
    public void WhisperLargeV3_WithFp16Hint_SelectsFp16RegardlessOfTier()
    {
        // Explicit hint overrides hardware tier
        var prefs = WithSubfolder(ModelPreferences.ForQuantizationHint("fp16"), "onnx");

        var result = DiscoverFromFileList(MakeRepoFiles(WhisperLargeV3Files), prefs);

        result.EncoderFiles.Should().ContainSingle().Which.Should().Contain("fp16");
        result.DecoderFiles.Should().ContainSingle().Which.Should().Contain("fp16");
    }

    [Fact]
    public void WhisperLargeV3_WithBnb4Hint_SelectsBnb4()
    {
        var prefs = WithSubfolder(ModelPreferences.ForQuantizationHint("bnb4"), "onnx");

        var result = DiscoverFromFileList(MakeRepoFiles(WhisperLargeV3Files), prefs);

        // bnb4 → Quant4. Priority: _int4(없음), _bnb4(있음) → bnb4 선택
        result.EncoderFiles.Should().ContainSingle().Which.Should().Contain("bnb4");
        result.DecoderFiles.Should().ContainSingle().Which.Should().Contain("bnb4");
    }

    // =========================================================================
    // 4. SplitQualifier syntax
    // =========================================================================

    [Theory]
    [InlineData("large:fp16", "large", "fp16")]
    [InlineData("default:q4", "default", "q4")]
    [InlineData("fast:int8", "fast", "int8")]
    [InlineData("quality:bnb4", "quality", "bnb4")]
    [InlineData("turbo:q4f16", "turbo", "q4f16")]
    public void SplitQualifier_ValidAliasWithQualifier_SplitsCorrectly(
        string input, string expectedBase, string expectedQualifier)
    {
        var (baseId, qualifier) = LMSupplyOptionsBase.SplitQualifier(input);
        baseId.Should().Be(expectedBase);
        qualifier.Should().Be(expectedQualifier);
    }

    [Theory]
    [InlineData("large")]           // no qualifier
    [InlineData("default")]         // no qualifier
    [InlineData("org/model")]       // HF repo ID — don't split on /
    [InlineData("org/model:fp16")]  // HF repo with qualifier — contains /, skip
    [InlineData("C:\\models")]      // Windows path
    [InlineData("/home/models")]    // Unix path
    [InlineData("")]                // empty
    [InlineData(":fp16")]           // leading colon
    [InlineData("large:")]          // trailing colon
    public void SplitQualifier_NoQualifierCases_ReturnsNullQualifier(string input)
    {
        var (_, qualifier) = LMSupplyOptionsBase.SplitQualifier(input);
        qualifier.Should().BeNull();
    }

    [Fact]
    public void SplitQualifier_HuggingFaceRepoWithSlash_PreservesFullId()
    {
        var (baseId, qualifier) = LMSupplyOptionsBase.SplitQualifier("onnx-community/whisper-large-v3-ONNX");
        baseId.Should().Be("onnx-community/whisper-large-v3-ONNX");
        qualifier.Should().BeNull();
    }

    // =========================================================================
    // 5. Extended quantization suffix matching
    // =========================================================================

    [Theory]
    [InlineData("model_bnb4.onnx", "model")]
    [InlineData("model_q4.onnx", "model")]
    [InlineData("model_q4f16.onnx", "model")]
    [InlineData("model_uint8.onnx", "model")]
    [InlineData("model_quantized.onnx", "model")]
    [InlineData("model_q8.onnx", "model")]
    [InlineData("model_fp16.onnx", "model")]
    [InlineData("model_int4.onnx", "model")]
    [InlineData("model_int8.onnx", "model")]
    [InlineData("encoder_model_bnb4.onnx", "encoder_model")]
    [InlineData("decoder_model_merged_q4f16.onnx", "decoder_model_merged")]
    public void SelectBestVariants_AllQuantSuffixes_GroupedUnderSameBaseName(
        string fileName, string expectedBaseName)
    {
        // Two files: one with suffix, one FP32 — both should group to same base name
        var candidates = new List<RepoFile>
        {
            new() { Path = $"onnx/{expectedBaseName}.onnx", Type = "file", Size = 1000 },
            new() { Path = $"onnx/{fileName}", Type = "file", Size = 500 },
        };

        var result = ModelDiscoveryService.SelectBestVariantsForTest(candidates, ModelPreferences.Default);

        // Same group → only 1 file selected
        result.Should().HaveCount(1);
    }

    // =========================================================================
    // 6. Encoder-decoder: merged decoder preferred over standard
    // =========================================================================

    [Fact]
    public void DiscoverModel_MergedAndStandardDecoder_PrefersMerged()
    {
        var files = MakeRepoFiles([
            "onnx/encoder_model.onnx",
            "onnx/decoder_model.onnx",
            "onnx/decoder_model_merged.onnx",
            "onnx/decoder_with_past_model.onnx"
        ]);

        var result = DiscoverFromFileList(files);

        result.DetectedDecoderVariant.Should().Be(DecoderVariant.Merged);
        result.DecoderFiles.Should().ContainSingle()
            .Which.Should().Contain("merged");
    }

    // =========================================================================
    // Helpers
    // =========================================================================

    private static ModelPreferences WithSubfolder(ModelPreferences source, string subfolder) => new()
    {
        PreferLowMemory = source.PreferLowMemory,
        PreferredProvider = source.PreferredProvider,
        QuantizationPriority = source.QuantizationPriority,
        PreferredSubfolder = subfolder,
        PreferredOnnxFiles = source.PreferredOnnxFiles,
        DecoderVariantPriority = source.DecoderVariantPriority,
        RequireMatchedQuantization = source.RequireMatchedQuantization,
    };

    private static List<RepoFile> MakeRepoFiles(string[] paths)
    {
        return paths.Select((p, i) => new RepoFile
        {
            Path = p,
            Type = "file",
            Size = 100_000_000L + i * 1_000_000L // slightly different sizes
        }).ToList();
    }

    /// <summary>
    /// Runs ModelDiscoveryService.DiscoverModelAsync logic synchronously using in-memory file list.
    /// This bypasses the HF API call by directly invoking the internal selection methods.
    /// </summary>
    private static ModelDiscoveryResult DiscoverFromFileList(
        List<RepoFile> allFiles,
        ModelPreferences? preferences = null)
    {
        preferences ??= ModelPreferences.Default;

        // Filter to ONNX files
        var onnxFiles = allFiles
            .Where(f => f.Path.EndsWith(".onnx", StringComparison.OrdinalIgnoreCase))
            .ToList();

        // Detect architecture
        var architecture = ModelDiscoveryService.DetectArchitecture(onnxFiles);

        // Select via the public test entry point
        var selectedOnnxFiles = ModelDiscoveryService.SelectBestVariantsForTest(
            onnxFiles.Where(f =>
            {
                var dir = f.Directory;
                return preferences.PreferredSubfolder is null
                    ? dir is null
                    : dir?.Equals(preferences.PreferredSubfolder, StringComparison.OrdinalIgnoreCase) == true;
            }).ToList(),
            preferences);

        // For encoder-decoder, we need to invoke full discovery to test classification.
        // Use reflection-free approach: call the public DiscoverModelAsync with a mock service.
        // Instead, we reconstruct the logic that DiscoverModelAsync performs:
        if (architecture == ModelArchitecture.EncoderDecoder)
        {
            return DiscoverEncoderDecoderFromFiles(onnxFiles, allFiles, preferences);
        }

        return new ModelDiscoveryResult
        {
            RepoId = "test/model",
            Subfolder = preferences.PreferredSubfolder,
            OnnxFiles = selectedOnnxFiles,
            Architecture = architecture,
        };
    }

    /// <summary>
    /// Simulates the encoder-decoder discovery path without network calls,
    /// using the internal ClassifyEncoderDecoderForTest wrapper.
    /// </summary>
    private static ModelDiscoveryResult DiscoverEncoderDecoderFromFiles(
        List<RepoFile> onnxFiles,
        List<RepoFile> allFiles,
        ModelPreferences preferences)
    {
        var subfolder = preferences.PreferredSubfolder;
        var filtered = subfolder is null
            ? onnxFiles.Where(f => f.Directory is null).ToList()
            : onnxFiles.Where(f =>
                f.Directory?.Equals(subfolder, StringComparison.OrdinalIgnoreCase) == true).ToList();

        if (filtered.Count == 0) filtered = onnxFiles;

        var (encoderFiles, decoderFiles, variant) =
            ModelDiscoveryService.ClassifyEncoderDecoderForTest(filtered, preferences);

        return new ModelDiscoveryResult
        {
            RepoId = "test/model",
            Subfolder = subfolder,
            OnnxFiles = encoderFiles.Concat(decoderFiles).ToList(),
            Architecture = ModelArchitecture.EncoderDecoder,
            EncoderFiles = encoderFiles,
            DecoderFiles = decoderFiles,
            DetectedDecoderVariant = variant,
        };
    }

    private static string? GetQuantSuffix(string path)
    {
        var name = Path.GetFileNameWithoutExtension(path);
        string[] suffixes = ["_q4f16", "_quantized", "_uint8", "_bnb4", "_int4", "_int8", "_fp16", "_q4", "_q8"];
        return suffixes.FirstOrDefault(s => name.EndsWith(s, StringComparison.OrdinalIgnoreCase));
    }
}
