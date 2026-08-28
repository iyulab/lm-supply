using AwesomeAssertions;
using LMSupply.Core.Download;

namespace LMSupply.Core.Tests.Download;

/// <summary>
/// Tests for Whisper ONNX quantization variant selection.
/// Regression tests for the bug where all quantization variants (_bnb4, _q4f16)
/// were downloaded instead of only the optimal one.
/// See: ISSUE-lm-supply-20260305-model-all-quantizations-downloaded
/// </summary>
public class WhisperQuantizationSelectionTests
{
    // --- Grouping: _bnb4 must be treated as same base model as full precision ---

    [Fact]
    public void SelectBestVariants_EncoderWithBnb4Variant_SelectsOneFile()
    {
        // whisper-large-v3-ONNX has: encoder_model.onnx, encoder_model_bnb4.onnx, encoder_model_q4f16.onnx
        // All three are variants of the same "encoder_model" base → should select exactly 1
        var candidates = MakeFiles([
            "onnx/encoder_model.onnx",
            "onnx/encoder_model_bnb4.onnx",
            "onnx/encoder_model_q4f16.onnx"
        ]);

        var result = ModelDiscoveryService.SelectBestVariantsForTest(candidates, ModelPreferences.Default);

        result.Should().HaveCount(1, "all three are quantization variants of the same encoder model");
    }

    [Fact]
    public void SelectBestVariants_DecoderMergedWithBnb4AndQ4f16_SelectsOneFile()
    {
        // decoder_model_merged.onnx / decoder_model_merged_bnb4.onnx / decoder_model_merged_q4f16.onnx
        var candidates = MakeFiles([
            "onnx/decoder_model_merged.onnx",
            "onnx/decoder_model_merged_bnb4.onnx",
            "onnx/decoder_model_merged_q4f16.onnx"
        ]);

        var result = ModelDiscoveryService.SelectBestVariantsForTest(candidates, ModelPreferences.Default);

        result.Should().HaveCount(1, "all three are quantization variants of the same merged decoder");
    }

    [Fact]
    public void SelectBestVariants_DecoderWithPastVariants_SelectsOneFile()
    {
        var candidates = MakeFiles([
            "onnx/decoder_with_past_model.onnx",
            "onnx/decoder_with_past_model_bnb4.onnx",
            "onnx/decoder_with_past_model_q4f16.onnx"
        ]);

        var result = ModelDiscoveryService.SelectBestVariantsForTest(candidates, ModelPreferences.Default);

        result.Should().HaveCount(1);
    }

    // --- Quantization priority: Default preferences → full precision ---

    [Fact]
    public void SelectBestVariants_DefaultPreferences_SelectsFullPrecisionEncoder()
    {
        var candidates = MakeFiles([
            "onnx/encoder_model.onnx",
            "onnx/encoder_model_bnb4.onnx",
            "onnx/encoder_model_q4f16.onnx"
        ]);

        var result = ModelDiscoveryService.SelectBestVariantsForTest(candidates, ModelPreferences.Default);

        result.Should().ContainSingle().Which.Should().EndWith("encoder_model.onnx",
            "Default preferences prioritize full precision (no quantization suffix)");
    }

    [Fact]
    public void SelectBestVariants_DefaultPreferences_SelectsFullPrecisionDecoderMerged()
    {
        var candidates = MakeFiles([
            "onnx/decoder_model_merged.onnx",
            "onnx/decoder_model_merged_bnb4.onnx",
            "onnx/decoder_model_merged_q4f16.onnx"
        ]);

        var result = ModelDiscoveryService.SelectBestVariantsForTest(candidates, ModelPreferences.Default);

        result.Should().ContainSingle().Which.Should().EndWith("decoder_model_merged.onnx");
    }

    // --- Full Whisper Large V3 file set simulation ---

    [Fact]
    public void SelectBestVariants_AllWhisperLargeV3OnnxFiles_SelectsFourFiles()
    {
        // Simulates the exact onnx/ subfolder contents of onnx-community/whisper-large-v3-ONNX
        // 4 base models × 3 variants = 12 files; should select 1 per group = 4 files
        var candidates = MakeFiles([
            "onnx/encoder_model.onnx",
            "onnx/encoder_model_bnb4.onnx",
            "onnx/encoder_model_q4f16.onnx",
            "onnx/decoder_model.onnx",
            "onnx/decoder_model_bnb4.onnx",
            "onnx/decoder_model_q4f16.onnx",
            "onnx/decoder_model_merged.onnx",
            "onnx/decoder_model_merged_bnb4.onnx",
            "onnx/decoder_model_merged_q4f16.onnx",
            "onnx/decoder_with_past_model.onnx",
            "onnx/decoder_with_past_model_bnb4.onnx",
            "onnx/decoder_with_past_model_q4f16.onnx"
        ]);

        var result = ModelDiscoveryService.SelectBestVariantsForTest(candidates, ModelPreferences.Default);

        result.Should().HaveCount(4,
            "encoder_model, decoder_model, decoder_model_merged, decoder_with_past_model = 4 groups");
    }

    [Fact]
    public void SelectBestVariants_AllWhisperLargeV3OnnxFiles_DefaultPreferences_SelectsFullPrecisionOnly()
    {
        var candidates = MakeFiles([
            "onnx/encoder_model.onnx",
            "onnx/encoder_model_bnb4.onnx",
            "onnx/encoder_model_q4f16.onnx",
            "onnx/decoder_model.onnx",
            "onnx/decoder_model_bnb4.onnx",
            "onnx/decoder_model_q4f16.onnx",
            "onnx/decoder_model_merged.onnx",
            "onnx/decoder_model_merged_bnb4.onnx",
            "onnx/decoder_model_merged_q4f16.onnx",
            "onnx/decoder_with_past_model.onnx",
            "onnx/decoder_with_past_model_bnb4.onnx",
            "onnx/decoder_with_past_model_q4f16.onnx"
        ]);

        var result = ModelDiscoveryService.SelectBestVariantsForTest(candidates, ModelPreferences.Default);

        result.Should().NotContain(f => f.Contains("_bnb4") || f.Contains("_q4f16"),
            "Default preferences should select full-precision files, not quantized variants");
    }

    // --- Integration: DiscoverModelAsync for EncoderDecoder must use encoder+decoder files only ---

    [Fact]
    [Trait("Category", "Integration")]
    public async Task DiscoverModel_WhisperLargeV3_OnnxFilesContainsOnlyEncoderAndDecoder()
    {
        using var service = new ModelDiscoveryService();

        var result = await service.DiscoverModelAsync(
            "onnx-community/whisper-large-v3-ONNX",
            new ModelPreferences { PreferredSubfolder = "onnx" });

        result.Architecture.Should().Be(ModelArchitecture.EncoderDecoder);

        // OnnxFiles (used for download) must only contain the selected encoder + decoder pair
        // NOT all quantization variants
        result.OnnxFiles.Count.Should().BeLessThanOrEqualTo(2,
            "only one encoder file and one decoder file should be in OnnxFiles for download");

        result.OnnxFiles.Should().Contain(f => f.Contains("encoder"),
            "OnnxFiles must contain the encoder");
        result.OnnxFiles.Should().Contain(f => f.Contains("decoder"),
            "OnnxFiles must contain the decoder");
    }

    // --- Helper ---

    private static List<RepoFile> MakeFiles(string[] paths, long sizeBase = 100_000_000)
    {
        return paths.Select((p, i) => new RepoFile
        {
            Path = p,
            Type = "file",
            Size = sizeBase + i * 1000 // slightly different sizes
        }).ToList();
    }
}
