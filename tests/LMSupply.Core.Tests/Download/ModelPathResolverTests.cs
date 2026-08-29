using AwesomeAssertions;
using LMSupply.Core.Download;
using LMSupply.Download;

namespace LMSupply.Core.Tests.Download;

public class ModelPathResolverTests : IDisposable
{
    private readonly string _tempDir;

    public ModelPathResolverTests()
    {
        _tempDir = Path.Combine(Path.GetTempPath(), $"lmsupply-test-{Guid.NewGuid()}");
        Directory.CreateDirectory(_tempDir);
    }

    public void Dispose()
    {
        if (Directory.Exists(_tempDir))
        {
            Directory.Delete(_tempDir, recursive: true);
        }
        GC.SuppressFinalize(this);
    }

    [Fact]
    public async Task ResolveModelAsync_WithLocalFilePath_ShouldReturnFilePath()
    {
        // Arrange
        var modelFile = Path.Combine(_tempDir, "model.onnx");
        await File.WriteAllTextAsync(modelFile, "dummy onnx content", TestContext.Current.CancellationToken);

        using var resolver = new ModelPathResolver(_tempDir);

        // Act
        var result = await resolver.ResolveModelAsync(modelFile, cancellationToken: TestContext.Current.CancellationToken);

        // Assert
        result.ModelPath.Should().Be(modelFile);
        result.BaseDirectory.Should().Be(_tempDir);
        result.OnnxDirectory.Should().Be(_tempDir);
        result.Discovery.Should().BeNull();
    }

    [Fact]
    public async Task ResolveModelAsync_WithLocalDirectory_ShouldFindOnnxFile()
    {
        // Arrange
        var modelDir = Path.Combine(_tempDir, "my-model");
        Directory.CreateDirectory(modelDir);
        var modelFile = Path.Combine(modelDir, "model.onnx");
        await File.WriteAllTextAsync(modelFile, "dummy onnx content", TestContext.Current.CancellationToken);

        using var resolver = new ModelPathResolver(_tempDir);

        // Act
        var result = await resolver.ResolveModelAsync(modelDir, "model.onnx", cancellationToken: TestContext.Current.CancellationToken);

        // Assert
        result.ModelPath.Should().Be(modelFile);
        result.BaseDirectory.Should().Be(modelDir);
        result.OnnxDirectory.Should().Be(modelDir);
        result.Discovery.Should().BeNull();
    }

    [Fact]
    public async Task ResolveModelAsync_WithLocalDirectoryAndSubfolder_ShouldFindOnnxInSubfolder()
    {
        // Arrange - simulate onnx subfolder structure like HuggingFace repos
        var modelDir = Path.Combine(_tempDir, "hf-style-model");
        var onnxSubfolder = Path.Combine(modelDir, "onnx");
        Directory.CreateDirectory(onnxSubfolder);

        var modelFile = Path.Combine(onnxSubfolder, "model.onnx");
        await File.WriteAllTextAsync(modelFile, "dummy onnx content", TestContext.Current.CancellationToken);

        // Also create config in root
        await File.WriteAllTextAsync(Path.Combine(modelDir, "config.json"), "{}", TestContext.Current.CancellationToken);

        // Write a manifest so directory validation passes (subfolder ONNX files are not visible at root)
        var manifest = new DownloadManifest
        {
            Files =
            [
                new ManifestFileEntry { Path = "onnx/model.onnx", Size = (new FileInfo(modelFile)).Length },
                new ManifestFileEntry { Path = "config.json", Size = 2 }
            ]
        };
        await DownloadManifest.WriteAsync(modelDir, manifest);

        using var resolver = new ModelPathResolver(_tempDir);

        // Act - when specified file not in root, should search subdirectories
        var result = await resolver.ResolveModelAsync(modelDir, "model.onnx", cancellationToken: TestContext.Current.CancellationToken);

        // Assert
        result.ModelPath.Should().Be(modelFile);
        result.BaseDirectory.Should().Be(modelDir);
        result.OnnxDirectory.Should().Be(onnxSubfolder);
    }

    [Fact]
    public async Task ResolveModelAsync_WithNonExistentDirectory_ShouldThrow()
    {
        // Arrange
        var nonExistent = Path.Combine(_tempDir, "does-not-exist");
        using var resolver = new ModelPathResolver(_tempDir);

        // Act & Assert - non-existent local path will try HuggingFace download which should fail
        // For truly local paths, this simulates the behavior
        var act = async () => await resolver.ResolveModelAsync(nonExistent);

        // This will try to download from HuggingFace and fail (which is expected)
        await act.Should().ThrowAsync<Exception>();
    }

    [Fact]
    public async Task ResolveModelAsync_WithEmptyDirectory_ShouldThrowFileNotFound()
    {
        // Arrange
        var emptyDir = Path.Combine(_tempDir, "empty-model");
        Directory.CreateDirectory(emptyDir);

        using var resolver = new ModelPathResolver(_tempDir);

        // Act & Assert - no ONNX files found in empty directory
        var act = async () => await resolver.ResolveModelAsync(emptyDir, "model.onnx");
        await act.Should().ThrowAsync<FileNotFoundException>();
    }

    [Fact]
    public async Task ResolveEncoderDecoderAsync_WithLocalDirectory_ShouldFindBothFiles()
    {
        // Arrange
        var modelDir = Path.Combine(_tempDir, "encoder-decoder-model");
        Directory.CreateDirectory(modelDir);

        var encoderFile = Path.Combine(modelDir, "encoder_model.onnx");
        var decoderFile = Path.Combine(modelDir, "decoder_model_merged.onnx");
        await File.WriteAllTextAsync(encoderFile, "encoder content", TestContext.Current.CancellationToken);
        await File.WriteAllTextAsync(decoderFile, "decoder content", TestContext.Current.CancellationToken);

        using var resolver = new ModelPathResolver(_tempDir);

        // Act
        var result = await resolver.ResolveEncoderDecoderAsync(modelDir, cancellationToken: TestContext.Current.CancellationToken);

        // Assert
        result.EncoderPath.Should().Be(encoderFile);
        result.DecoderPath.Should().Be(decoderFile);
        result.BaseDirectory.Should().Be(modelDir);
        result.OnnxDirectory.Should().Be(modelDir);
        result.Discovery.Should().BeNull();
    }

    [Fact]
    public async Task ResolveEncoderDecoderAsync_WithMissingEncoder_ShouldThrow()
    {
        // Arrange - only decoder exists
        var modelDir = Path.Combine(_tempDir, "partial-model");
        Directory.CreateDirectory(modelDir);

        var decoderFile = Path.Combine(modelDir, "decoder_model_merged.onnx");
        await File.WriteAllTextAsync(decoderFile, "decoder content", TestContext.Current.CancellationToken);

        using var resolver = new ModelPathResolver(_tempDir);

        // Act & Assert - will try HuggingFace download which fails
        var act = async () => await resolver.ResolveEncoderDecoderAsync(modelDir);
        await act.Should().ThrowAsync<Exception>();
    }

    [Fact]
    public void Constructor_WithNullCacheDirectory_ShouldUseDefault()
    {
        // Act
        using var resolver = new ModelPathResolver(null);

        // Assert - should not throw
        resolver.Should().NotBeNull();
    }

    [Fact]
    public void Dispose_ShouldNotThrow()
    {
        // Arrange
        var resolver = new ModelPathResolver(_tempDir);

        // Act & Assert
        var act = () => resolver.Dispose();
        act.Should().NotThrow();
    }

    [Fact]
    public void Dispose_CalledMultipleTimes_ShouldNotThrow()
    {
        // Arrange
        var resolver = new ModelPathResolver(_tempDir);

        // Act & Assert
        resolver.Dispose();
        var act = () => resolver.Dispose();
        act.Should().NotThrow();
    }
}
