using AwesomeAssertions;
using LMSupply.Download;

namespace LMSupply.Core.Tests.Download;

public class DownloadManifestTests
{
    [Fact]
    public async Task WriteAndRead_RoundTrips()
    {
        var dir = Path.Combine(Path.GetTempPath(), $"manifest-test-{Guid.NewGuid():N}");
        Directory.CreateDirectory(dir);

        try
        {
            await File.WriteAllTextAsync(Path.Combine(dir, "model.onnx"), new string('x', 1000));
            await File.WriteAllTextAsync(Path.Combine(dir, "config.json"), "{}");

            var manifest = new DownloadManifest
            {
                RepoId = "org/model",
                Revision = "main",
                Files =
                [
                    new ManifestFileEntry { Path = "model.onnx", Size = 1000 },
                    new ManifestFileEntry { Path = "config.json", Size = 2 }
                ]
            };

            await DownloadManifest.WriteAsync(dir, manifest);

            var loaded = await DownloadManifest.ReadAsync(dir);
            loaded.Should().NotBeNull();
            loaded!.RepoId.Should().Be("org/model");
            loaded.Files.Should().HaveCount(2);
            loaded.CompletedAt.Should().NotBe(default);
        }
        finally
        {
            Directory.Delete(dir, true);
        }
    }

    [Fact]
    public async Task ReadAsync_MissingFile_ReturnsNull()
    {
        var dir = Path.Combine(Path.GetTempPath(), $"manifest-test-{Guid.NewGuid():N}");
        Directory.CreateDirectory(dir);

        try
        {
            var result = await DownloadManifest.ReadAsync(dir);
            result.Should().BeNull();
        }
        finally
        {
            Directory.Delete(dir, true);
        }
    }

    [Fact]
    public async Task ReadAsync_CorruptJson_ReturnsNull()
    {
        var dir = Path.Combine(Path.GetTempPath(), $"manifest-test-{Guid.NewGuid():N}");
        Directory.CreateDirectory(dir);

        try
        {
            await File.WriteAllTextAsync(
                Path.Combine(dir, ".lmsupply-manifest.json"), "not json{{{");

            var result = await DownloadManifest.ReadAsync(dir);
            result.Should().BeNull();
        }
        finally
        {
            Directory.Delete(dir, true);
        }
    }

    [Fact]
    public void CreateFromDirectory_CollectsAllFiles()
    {
        var dir = Path.Combine(Path.GetTempPath(), $"manifest-test-{Guid.NewGuid():N}");
        Directory.CreateDirectory(dir);

        try
        {
            File.WriteAllText(Path.Combine(dir, "model.onnx"), new string('a', 500));
            File.WriteAllText(Path.Combine(dir, "vocab.txt"), "hello");
            File.WriteAllText(Path.Combine(dir, "model.onnx.data.part"), "partial");

            var manifest = DownloadManifest.CreateFromDirectory(dir, "org/model", "main");
            manifest.Files.Should().HaveCount(2);
            manifest.Files.Should().NotContain(f => f.Path.Contains(".part"));
            manifest.Files.Should().NotContain(f => f.Path.Contains(".lmsupply-manifest"));
        }
        finally
        {
            Directory.Delete(dir, true);
        }
    }
}
