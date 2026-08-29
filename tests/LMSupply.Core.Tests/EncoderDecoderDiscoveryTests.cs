using LMSupply.Core.Download;
using Xunit;
using AwesomeAssertions;

namespace LMSupply.Core.Tests;

/// <summary>
/// Tests for encoder-decoder model auto-discovery (GitHub Issue #6 fix verification).
/// </summary>
public class EncoderDecoderDiscoveryTests
{
    [Fact]
    [Trait("Category", "Integration")]
    public async Task DiscoverModel_OpusMtKoEn_ShouldDetectEncoderDecoderArchitecture()
    {
        // Arrange - This tests the fix for GitHub Issue #6
        using var service = new ModelDiscoveryService();
        
        // Act - Discover onnx-community repo (the correct repo with ONNX files)
        var result = await service.DiscoverModelAsync("onnx-community/opus-mt-ko-en", cancellationToken: TestContext.Current.CancellationToken);
        
        // Assert
        result.Architecture.Should().Be(ModelArchitecture.EncoderDecoder,
            "MarianMT models have encoder-decoder architecture");
        
        result.EncoderFiles.Should().NotBeEmpty(
            "Should discover encoder_model.onnx or variant");
        
        result.DecoderFiles.Should().NotBeEmpty(
            "Should discover decoder_model.onnx or variant");
        
        result.PrimaryEncoderFile.Should().Contain("encoder",
            "Primary encoder file should contain 'encoder' in name");
        
        result.PrimaryDecoderFile.Should().Contain("decoder",
            "Primary decoder file should contain 'decoder' in name");
        
        result.Subfolder.Should().Be("onnx",
            "ONNX files should be in 'onnx' subfolder");
    }
    
    [Fact]
    public void IsEncoderDecoderModel_WithEncoderAndDecoder_ShouldReturnTrue()
    {
        // Arrange
        var files = new[]
        {
            "onnx/encoder_model.onnx",
            "onnx/decoder_model.onnx"
        };
        
        // Act
        var result = ModelDiscoveryService.IsEncoderDecoderModel(files);
        
        // Assert
        result.Should().BeTrue();
    }
    
    [Fact]
    public void IsEncoderDecoderModel_WithMergedDecoder_ShouldReturnTrue()
    {
        // Arrange
        var files = new[]
        {
            "encoder_model.onnx",
            "decoder_model_merged.onnx"
        };
        
        // Act
        var result = ModelDiscoveryService.IsEncoderDecoderModel(files);
        
        // Assert
        result.Should().BeTrue();
    }
    
    [Fact]
    public void IsEncoderDecoderModel_WithOnlyEncoder_ShouldReturnFalse()
    {
        // Arrange
        var files = new[] { "encoder_model.onnx" };
        
        // Act
        var result = ModelDiscoveryService.IsEncoderDecoderModel(files);
        
        // Assert
        result.Should().BeFalse();
    }
    
    [Fact]
    public void IsEncoderDecoderModel_WithSingleModel_ShouldReturnFalse()
    {
        // Arrange - Typical embedder model
        var files = new[] { "model.onnx" };
        
        // Act
        var result = ModelDiscoveryService.IsEncoderDecoderModel(files);
        
        // Assert
        result.Should().BeFalse();
    }
}
