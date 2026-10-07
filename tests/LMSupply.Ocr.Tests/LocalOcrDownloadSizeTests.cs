using AwesomeAssertions;
using LMSupply.Download;
using LMSupply.Exceptions;
using LMSupply.Ocr.Models;

namespace LMSupply.Ocr.Tests;

/// <summary>
/// <see cref="LocalOcr.GetDownloadSizeBytesAsync"/> is the default detection model plus the language's recognizer and
/// dictionary, at the lengths the repository lists — the files <see cref="LocalOcr.LoadForLanguageAsync"/> fetches. The
/// listing is seeded into the cache (fresh, so no request is made).
/// </summary>
public sealed class LocalOcrDownloadSizeTests : IDisposable
{
    private readonly string _cacheDir = Path.Combine(Path.GetTempPath(), "lmsupply-osize-" + Guid.NewGuid().ToString("N"));

    private static CancellationToken Ct => TestContext.Current.CancellationToken;

    public void Dispose()
    {
        if (Directory.Exists(_cacheDir))
            Directory.Delete(_cacheDir, recursive: true);
    }

    private static string InFolder(string? subfolder, string file) => string.IsNullOrEmpty(subfolder) ? file : $"{subfolder}/{file}";

    private void SeedListing(string repoId, IEnumerable<(string Path, long Size)> files)
    {
        var dir = Path.Combine(_cacheDir, ".discovery-cache");
        Directory.CreateDirectory(dir);
        File.WriteAllText(
            Path.Combine(dir, repoId.Replace('/', '_') + "_main.json"),
            "[" + string.Join(",", files.Select(f => $$"""{"path":"{{f.Path}}","type":"file","size":{{f.Size}}}""")) + "]");
    }

    [Theory]
    [InlineData("en")]
    [InlineData("ko")]
    public async Task Language_IsDetectionPlusThatLanguagesRecognizer(string language)
    {
        var detection = OcrDetectionModelRegistry.Default.Resolve("default");
        var recognition = OcrRecognitionModelRegistry.Default.ResolveForLanguage(language);
        var other = OcrRecognitionModelRegistry.Default.ResolveForLanguage(language == "en" ? "ko" : "en");
        detection.RepoId.Should().Be(recognition.RepoId, "the fixture seeds one listing for the shared repository");

        SeedListing(detection.RepoId,
        [
            (InFolder(detection.Subfolder, detection.ModelFile), 2_400_000),
            (InFolder(recognition.Subfolder, recognition.ModelFile), 7_000_000),
            (InFolder(recognition.Subfolder, recognition.DictFile), 30_000),
            (InFolder(other.Subfolder, other.ModelFile), 9_999_999),
            (InFolder(other.Subfolder, other.DictFile), 99_999),
        ]);

        (await LocalOcr.GetDownloadSizeBytesAsync(language, new OcrOptions { CacheDirectory = _cacheDir }, Ct))
            .Should().Be(9_430_000, "another language's recognizer is not fetched");
    }

    [Fact]
    public async Task RemainingBytes_CountEachModelTheCacheLacks()
    {
        var detection = OcrDetectionModelRegistry.Default.Resolve("default");
        var recognition = OcrRecognitionModelRegistry.Default.ResolveForLanguage("en");
        var detectionFile = InFolder(detection.Subfolder, detection.ModelFile);
        SeedListing(detection.RepoId,
        [
            (detectionFile, 2_400),
            (InFolder(recognition.Subfolder, recognition.ModelFile), 7_000),
            (InFolder(recognition.Subfolder, recognition.DictFile), 30),
        ]);
        var options = new OcrOptions { CacheDirectory = _cacheDir };
        (await LocalOcr.GetRemainingDownloadBytesAsync("en", options, Ct)).Should().Be(9_430);

        // The detector is cached; the recognizer is not
        var path = Path.Combine(CacheManager.GetModelDirectory(_cacheDir, detection.RepoId), detectionFile.Replace('/', Path.DirectorySeparatorChar));
        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        await File.WriteAllBytesAsync(path, new byte[2_400], Ct);

        (await LocalOcr.GetRemainingDownloadBytesAsync("en", options, Ct)).Should().Be(7_030);
        (await LocalOcr.GetDownloadSizeBytesAsync("en", options, Ct)).Should().Be(9_430);
    }

    [Fact]
    public async Task Offline_NeverListed_Throws()
    {
        var act = () => LocalOcr.GetDownloadSizeBytesAsync("en", new OcrOptions { CacheDirectory = _cacheDir, DisableAutoDownload = true }, Ct);

        await act.Should().ThrowAsync<ModelNotFoundException>();
    }
}
