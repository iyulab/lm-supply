using FluentAssertions;
using LMSupply.Runtime;

namespace LMSupply.Core.Tests.Runtime;

public class RuntimeManifestTests
{
    private const string ValidManifestJson = """
        {
            "version": "1.0.0",
            "updated": "2024-12-13T00:00:00Z",
            "packages": {
                "onnxruntime": {
                    "description": "ONNX Runtime",
                    "homepage": "https://onnxruntime.ai",
                    "versions": {
                        "1.20.1": {
                            "released": "2024-11-15T00:00:00Z",
                            "binaries": [
                                {
                                    "rid": "win-x64",
                                    "provider": "cpu",
                                    "url": "https://example.com/onnxruntime.zip",
                                    "fileName": "onnxruntime.dll",
                                    "size": 16777216,
                                    "sha256": "abc123"
                                },
                                {
                                    "rid": "win-x64",
                                    "provider": "cuda12",
                                    "url": "https://example.com/onnxruntime-cuda.zip",
                                    "fileName": "onnxruntime.dll",
                                    "size": 220000000,
                                    "sha256": "def456",
                                    "dependencies": ["onnxruntime_providers_cuda.dll"]
                                },
                                {
                                    "rid": "linux-x64",
                                    "provider": "cpu",
                                    "url": "https://example.com/onnxruntime-linux.tar.gz",
                                    "fileName": "libonnxruntime.so",
                                    "size": 15000000,
                                    "sha256": "ghi789"
                                }
                            ]
                        }
                    }
                }
            }
        }
        """;

    [Fact]
    public void Parse_WithValidJson_ShouldReturnManifest()
    {
        // Act
        var manifest = RuntimeManifest.Parse(ValidManifestJson);

        // Assert
        manifest.Should().NotBeNull();
        manifest.Version.Should().Be("1.0.0");
    }

    [Fact]
    public void GetPackage_WithValidPackage_ShouldReturnPackageInfo()
    {
        // Arrange
        var manifest = RuntimeManifest.Parse(ValidManifestJson);

        // Act
        var package = manifest.GetPackage("onnxruntime");

        // Assert
        package.Should().NotBeNull();
        package!.Description.Should().Be("ONNX Runtime");
        package.Homepage.Should().Be("https://onnxruntime.ai");
    }

    [Fact]
    public void GetPackage_WithInvalidPackage_ShouldReturnNull()
    {
        // Arrange
        var manifest = RuntimeManifest.Parse(ValidManifestJson);

        // Act
        var package = manifest.GetPackage("nonexistent");

        // Assert
        package.Should().BeNull();
    }

    [Fact]
    public void GetBinaries_WithValidRid_ShouldReturnBinaries()
    {
        // Arrange
        var manifest = RuntimeManifest.Parse(ValidManifestJson);

        // Act
        var binaries = manifest.GetBinaries("onnxruntime", "win-x64").ToList();

        // Assert
        binaries.Should().HaveCount(2);
        binaries.Should().Contain(b => b.Provider == "cpu");
        binaries.Should().Contain(b => b.Provider == "cuda12");
    }

    [Fact]
    public void GetBinaries_WithInvalidRid_ShouldReturnEmpty()
    {
        // Arrange
        var manifest = RuntimeManifest.Parse(ValidManifestJson);

        // Act
        var binaries = manifest.GetBinaries("onnxruntime", "osx-arm64").ToList();

        // Assert
        binaries.Should().BeEmpty();
    }

    [Fact]
    public void GetLatestBinary_WithValidRid_ShouldReturnBinary()
    {
        // Arrange
        var manifest = RuntimeManifest.Parse(ValidManifestJson);

        // Act
        var binary = manifest.GetLatestBinary("onnxruntime", "win-x64");

        // Assert
        binary.Should().NotBeNull();
        binary!.RuntimeIdentifier.Should().Be("win-x64");
        binary.FileName.Should().Be("onnxruntime.dll");
    }

    [Fact]
    public void GetLatestBinary_WithInvalidRid_ShouldReturnNull()
    {
        // Arrange
        var manifest = RuntimeManifest.Parse(ValidManifestJson);

        // Act
        var binary = manifest.GetLatestBinary("onnxruntime", "linux-arm64");

        // Assert
        binary.Should().BeNull();
    }

    [Fact]
    public void RuntimeBinaryEntry_ShouldHaveDependencies()
    {
        // Arrange
        var manifest = RuntimeManifest.Parse(ValidManifestJson);

        // Act
        var binaries = manifest.GetBinaries("onnxruntime", "win-x64").ToList();
        var cudaBinary = binaries.FirstOrDefault(b => b.Provider == "cuda12");

        // Assert
        cudaBinary.Should().NotBeNull();
        cudaBinary!.Dependencies.Should().Contain("onnxruntime_providers_cuda.dll");
    }

    [Fact]
    public void Parse_WithInvalidJson_ShouldThrowException()
    {
        // Arrange
        var invalidJson = "{ invalid json }";

        // Act
        var action = () => RuntimeManifest.Parse(invalidJson);

        // Assert
        action.Should().Throw<Exception>();
    }

    [Fact]
    public void RuntimeBinaryEntry_SizeMB_ShouldCalculateCorrectly()
    {
        // Arrange
        var manifest = RuntimeManifest.Parse(ValidManifestJson);

        // Act
        var binary = manifest.GetLatestBinary("onnxruntime", "win-x64");

        // Assert
        binary.Should().NotBeNull();
        binary!.SizeMB.Should().BeApproximately(16.0, 0.1); // 16777216 bytes ≈ 16 MB
    }
}
