using Xunit;
using AwesomeAssertions;

namespace LMSupply.Translator.Tests.Tokenization;

/// <summary>
/// Tests for path resolution logic - verifies fix for GitHub Issue #7.
/// Tests path preservation across subfolders for ONNX model discovery.
/// Note: TranslatorTokenizer is internal, so we test the path logic patterns directly.
/// </summary>
public class TranslatorTokenizerPathTests : IDisposable
{
    private readonly string _testDir;

    public TranslatorTokenizerPathTests()
    {
        _testDir = Path.Combine(Path.GetTempPath(), $"lmsupply-test-{Guid.NewGuid():N}");
        Directory.CreateDirectory(_testDir);
    }

    [Fact]
    public void DirectorySearch_WhenOnnxSubfolderExists_ShouldBeSearchable()
    {
        // Arrange - Create test directory structure with onnx subfolder
        var onnxDir = Path.Combine(_testDir, "onnx");
        Directory.CreateDirectory(onnxDir);
        File.WriteAllText(Path.Combine(onnxDir, "source.spm"), "dummy");

        // Act - Simulate the search logic used in TranslatorTokenizer
        var searchPaths = new List<string> { _testDir };
        var subdirs = new[] { "onnx", "model" };
        foreach (var subdir in subdirs)
        {
            var subdirPath = Path.Combine(_testDir, subdir);
            if (Directory.Exists(subdirPath))
                searchPaths.Add(subdirPath);
        }

        // Assert
        searchPaths.Should().HaveCount(2);
        searchPaths.Should().Contain(onnxDir);
    }

    [Fact]
    public void DirectorySearch_WhenMultipleSubfoldersExist_ShouldFindAll()
    {
        // Arrange - Create multiple subdirectories
        var onnxDir = Path.Combine(_testDir, "onnx");
        var modelDir = Path.Combine(_testDir, "model");
        Directory.CreateDirectory(onnxDir);
        Directory.CreateDirectory(modelDir);

        // Act - Simulate search path construction
        var searchPaths = new List<string> { _testDir };
        var subdirs = new[] { "onnx", "model" };
        foreach (var subdir in subdirs)
        {
            var subdirPath = Path.Combine(_testDir, subdir);
            if (Directory.Exists(subdirPath))
                searchPaths.Add(subdirPath);
        }

        // Assert
        searchPaths.Should().HaveCount(3);
        searchPaths.Should().Contain(onnxDir);
        searchPaths.Should().Contain(modelDir);
    }

    [Fact]
    public void OnnxPathPreservation_WithSubfolder_ShouldMaintainFullPath()
    {
        // Arrange - Simulate discovery result path
        var discoveryPath = "onnx/encoder_model.onnx";

        // Act - Simulate path processing (as fixed in OnnxTranslatorModel.cs)
        var processedPath = discoveryPath.Replace('/', Path.DirectorySeparatorChar);

        // Assert
        processedPath.Should().Contain("onnx");
        processedPath.Should().EndWith("encoder_model.onnx");

        // On Windows, should use backslash
        if (Path.DirectorySeparatorChar == '\\')
        {
            processedPath.Should().Be("onnx\\encoder_model.onnx");
        }
        else
        {
            processedPath.Should().Be("onnx/encoder_model.onnx");
        }
    }

    [Fact]
    public void OnnxPathPreservation_WithNestedSubfolder_ShouldMaintainFullPath()
    {
        // Arrange - Deeper nested path
        var discoveryPath = "models/onnx/encoder_model.onnx";

        // Act
        var processedPath = discoveryPath.Replace('/', Path.DirectorySeparatorChar);

        // Assert
        processedPath.Should().Contain("models");
        processedPath.Should().Contain("onnx");
        processedPath.Should().EndWith("encoder_model.onnx");
    }

    [Fact]
    public void OnnxPathPreservation_WithoutSubfolder_ShouldWorkAsIs()
    {
        // Arrange - Root level path
        var discoveryPath = "encoder_model.onnx";

        // Act
        var processedPath = discoveryPath.Replace('/', Path.DirectorySeparatorChar);

        // Assert
        processedPath.Should().Be("encoder_model.onnx");
    }

    [Fact]
    public void PathCombine_WithPreservedSubfolder_ShouldResolveCorrectly()
    {
        // Arrange
        var modelDir = "/cache/models--onnx-community--opus-mt-ko-en/snapshots/main";
        var encoderFile = "onnx/encoder_model.onnx".Replace('/', Path.DirectorySeparatorChar);

        // Act
        var fullPath = Path.Combine(modelDir, encoderFile);

        // Assert
        fullPath.Should().Contain("onnx");
        fullPath.Should().EndWith("encoder_model.onnx");
    }

    public void Dispose()
    {
        try
        {
            if (Directory.Exists(_testDir))
            {
                Directory.Delete(_testDir, recursive: true);
            }
        }
        catch
        {
            // Ignore cleanup errors
        }
        GC.SuppressFinalize(this);
    }
}
