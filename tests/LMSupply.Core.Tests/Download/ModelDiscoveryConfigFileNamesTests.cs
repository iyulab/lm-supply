using LMSupply.Core.Download;
using Xunit;
using AwesomeAssertions;

namespace LMSupply.Core.Tests.Download;

/// <summary>
/// Tests for ModelDiscoveryService ConfigFileNames - verifies fix for GitHub Issue #7.
/// Ensures SentencePiece tokenizer files (source.spm, target.spm) are included in discovery.
/// </summary>
public class ModelDiscoveryConfigFileNamesTests
{
    [Fact]
    [Trait("Category", "Integration")]
    public async Task DiscoverModel_OpusMtKoEn_ShouldIncludeSentencePieceFiles()
    {
        // Arrange - This tests the fix for GitHub Issue #7
        using var service = new ModelDiscoveryService();

        // Act - Discover onnx-community repo
        var result = await service.DiscoverModelAsync("onnx-community/opus-mt-ko-en", cancellationToken: TestContext.Current.CancellationToken);

        // Assert - ConfigFiles should include SentencePiece tokenizer files
        result.ConfigFiles.Should().Contain(
            f => f.EndsWith("source.spm", StringComparison.OrdinalIgnoreCase) ||
                 f.EndsWith("sentencepiece.bpe.model", StringComparison.OrdinalIgnoreCase),
            "Should discover SentencePiece tokenizer files for MarianMT models");
    }

    [Fact]
    [Trait("Category", "Integration")]
    public async Task DiscoverModel_OpusMtKoEn_ShouldPreserveSubfolderPath()
    {
        // Arrange - This tests path preservation for GitHub Issue #7
        using var service = new ModelDiscoveryService();

        // Act
        var result = await service.DiscoverModelAsync("onnx-community/opus-mt-ko-en", cancellationToken: TestContext.Current.CancellationToken);

        // Assert - Primary files should have subfolder path preserved
        result.PrimaryEncoderFile.Should().Contain("onnx/",
            "Encoder file path should preserve 'onnx/' subfolder");

        result.PrimaryDecoderFile.Should().Contain("onnx/",
            "Decoder file path should preserve 'onnx/' subfolder");
    }

    [Fact]
    [Trait("Category", "Integration")]
    public async Task DiscoverModel_OpusMtKoEn_ShouldDetectConfigFilesFromRoot()
    {
        // Arrange
        using var service = new ModelDiscoveryService();

        // Act
        var result = await service.DiscoverModelAsync("onnx-community/opus-mt-ko-en", cancellationToken: TestContext.Current.CancellationToken);

        // Assert - Config files from root should be included
        result.ConfigFiles.Should().Contain(
            f => f.Equals("config.json", StringComparison.OrdinalIgnoreCase) ||
                 f.Equals("tokenizer_config.json", StringComparison.OrdinalIgnoreCase),
            "Should discover config files from repository root");
    }
}
