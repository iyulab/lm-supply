using AwesomeAssertions;
using LMSupply.Download;

namespace LMSupply.Core.Tests.Download;

public class ModelDirectoryValidatorTests : IDisposable
{
    private readonly string _testDir;

    public ModelDirectoryValidatorTests()
    {
        _testDir = Path.Combine(Path.GetTempPath(), $"validator-test-{Guid.NewGuid():N}");
        Directory.CreateDirectory(_testDir);
    }

    public void Dispose()
    {
        if (Directory.Exists(_testDir))
            Directory.Delete(_testDir, true);
        GC.SuppressFinalize(this);
    }

    [Fact]
    public void Validate_NonexistentDirectory_ReturnsInvalid()
    {
        var result = ModelDirectoryValidator.Validate("/nonexistent/path");
        result.IsValid.Should().BeFalse();
        result.Reason.Should().Contain("does not exist");
    }

    [Fact]
    public void Validate_PartFilesPresent_ReturnsInvalid()
    {
        File.WriteAllText(Path.Combine(_testDir, "model.onnx"), "data");
        File.WriteAllText(Path.Combine(_testDir, "model.onnx.data.part"), "partial");

        var result = ModelDirectoryValidator.Validate(_testDir);
        result.IsValid.Should().BeFalse();
        result.Reason.Should().Contain("Incomplete download");
    }

    [Fact]
    public void Validate_LfsPointerFile_ReturnsInvalid()
    {
        File.WriteAllText(Path.Combine(_testDir, "model.onnx"),
            "version https://git-lfs.github.com/spec/v1\noid sha256:abc123\nsize 12345\n");

        var result = ModelDirectoryValidator.Validate(_testDir);
        result.IsValid.Should().BeFalse();
        result.Reason.Should().Contain("LFS pointer");
    }

    [Fact]
    public async Task Validate_WithManifest_AllFilesPresent_ReturnsValid()
    {
        File.WriteAllText(Path.Combine(_testDir, "model.onnx"), new string('x', 100));
        File.WriteAllText(Path.Combine(_testDir, "config.json"), "{}");

        var manifest = new DownloadManifest
        {
            Files =
            [
                new ManifestFileEntry { Path = "model.onnx", Size = 100 },
                new ManifestFileEntry { Path = "config.json", Size = 2 }
            ]
        };
        await DownloadManifest.WriteAsync(_testDir, manifest);

        var result = ModelDirectoryValidator.Validate(_testDir);
        result.IsValid.Should().BeTrue();
    }

    [Fact]
    public async Task Validate_WithManifest_MissingFile_ReturnsInvalid()
    {
        File.WriteAllText(Path.Combine(_testDir, "config.json"), "{}");

        var manifest = new DownloadManifest
        {
            Files =
            [
                new ManifestFileEntry { Path = "model.onnx", Size = 100 },
                new ManifestFileEntry { Path = "config.json", Size = 2 }
            ]
        };
        await DownloadManifest.WriteAsync(_testDir, manifest);

        var result = ModelDirectoryValidator.Validate(_testDir);
        result.IsValid.Should().BeFalse();
        result.MissingFiles.Should().Contain("model.onnx");
    }

    [Fact]
    public async Task Validate_WithManifest_SizeMismatch_ReturnsInvalid()
    {
        File.WriteAllText(Path.Combine(_testDir, "model.onnx"), "short");
        File.WriteAllText(Path.Combine(_testDir, "config.json"), "{}");

        var manifest = new DownloadManifest
        {
            Files =
            [
                new ManifestFileEntry { Path = "model.onnx", Size = 999999 },
                new ManifestFileEntry { Path = "config.json", Size = 2 }
            ]
        };
        await DownloadManifest.WriteAsync(_testDir, manifest);

        var result = ModelDirectoryValidator.Validate(_testDir);
        result.IsValid.Should().BeFalse();
        result.Reason.Should().Contain("size mismatch");
    }

    [Fact]
    public void FallbackValidate_OnnxGenAI_Valid()
    {
        File.WriteAllText(Path.Combine(_testDir, "genai_config.json"), "{}");
        File.WriteAllText(Path.Combine(_testDir, "model.onnx"), "data");

        var result = ModelDirectoryValidator.Validate(_testDir);
        result.IsValid.Should().BeTrue();
    }

    [Fact]
    public void FallbackValidate_GeneralOnnx_WithoutGenaiConfig_ReturnsValid()
    {
        File.WriteAllText(Path.Combine(_testDir, "model.onnx"), "data");

        var result = ModelDirectoryValidator.Validate(_testDir);
        result.IsValid.Should().BeTrue();
    }

    [Fact]
    public void FallbackValidate_EmptyDirectory_ReturnsInvalid()
    {
        var result = ModelDirectoryValidator.Validate(_testDir);
        result.IsValid.Should().BeFalse();
        result.Reason.Should().Contain("No model files");
    }

    [Fact]
    public void FallbackValidate_GgufFile_ValidMagic()
    {
        var ggufData = new byte[8];
        BitConverter.GetBytes(0x46554747u).CopyTo(ggufData, 0);
        BitConverter.GetBytes(3u).CopyTo(ggufData, 4);
        File.WriteAllBytes(Path.Combine(_testDir, "model.gguf"), ggufData);

        var result = ModelDirectoryValidator.Validate(_testDir);
        result.IsValid.Should().BeTrue();
    }

    [Fact]
    public void FallbackValidate_GgufFile_InvalidMagic()
    {
        File.WriteAllBytes(Path.Combine(_testDir, "model.gguf"), new byte[] { 0, 0, 0, 0, 0, 0, 0, 0 });

        var result = ModelDirectoryValidator.Validate(_testDir);
        result.IsValid.Should().BeFalse();
        result.Reason.Should().Contain("GGUF");
    }
}
