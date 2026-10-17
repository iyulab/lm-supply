using System.Text.Json;
using AwesomeAssertions;
using LMSupply.Download;
using LMSupply.Ocr.Models;

namespace LMSupply.Ocr.Tests;

/// <summary>
/// <see cref="OcrArtifacts"/>, <see cref="LocalOcr.IsLanguageDownloadedAsync"/>, <see cref="LocalOcr.DownloadForLanguageAsync"/> and
/// <see cref="LocalOcr.DeleteLanguageModel"/> over a seeded cache: the repository listing with small sizes and the files a language
/// load opens written at those sizes. Nothing reaches the network. Detection and every recognizer live in one repository, so a
/// language delete must remove that language's recognizer only.
/// </summary>
public sealed class OcrArtifactsTests : IDisposable
{
    private const long FileSize = 32;
    private readonly string _cache = Path.Combine(Path.GetTempPath(), "lmsupply-ocr-artifacts-" + Guid.NewGuid().ToString("N"));

    private static CancellationToken Ct => TestContext.Current.CancellationToken;

    public void Dispose()
    {
        if (Directory.Exists(_cache))
            Directory.Delete(_cache, recursive: true);
    }

    private OcrOptions Options(bool downloadsDisabled = false) => new()
    {
        CacheDirectory = _cache,
        Provider = ExecutionProvider.Cpu,
        DisableAutoDownload = downloadsDisabled,
    };

    private static string InFolder(string? subfolder, string file)
        => string.IsNullOrEmpty(subfolder) ? file : $"{subfolder.TrimEnd('/')}/{file}";

    // The files a load for these languages opens: the shared detection model, then each language's recognizer and dictionary.
    private static string[] FilesFor(params string[] languages)
    {
        var detection = LocalOcr.DetectionRegistry.Resolve("default");
        var files = new List<string> { InFolder(detection.Subfolder, detection.ModelFile) };
        foreach (var language in languages)
        {
            var recognition = OcrRecognitionModelRegistry.Default.ResolveForLanguage(language);
            files.Add(InFolder(recognition.Subfolder, recognition.ModelFile));
            files.Add(InFolder(recognition.Subfolder, recognition.DictFile));
        }
        return files.Distinct().ToArray();
    }

    private void SeedCached(params string[] languages)
    {
        var repoId = LocalOcr.DetectionRegistry.Resolve("default").RepoId;
        var files = FilesFor(languages);
        var listing = Path.Combine(_cache, "models--" + repoId.Replace("/", "--"), ".lmsupply", "listings", "main.json");
        Directory.CreateDirectory(Path.GetDirectoryName(listing)!);
        File.WriteAllText(listing, JsonSerializer.Serialize(files.Select(f => new { path = f, type = "file", size = FileSize })));

        var snapshot = CacheManager.GetModelDirectory(_cache, repoId);
        foreach (var file in files)
        {
            var path = Path.Combine(snapshot, file.Replace('/', Path.DirectorySeparatorChar));
            Directory.CreateDirectory(Path.GetDirectoryName(path)!);
            File.WriteAllBytes(path, new byte[FileSize]);
        }
    }

    [Fact]
    public async Task ACachedLanguage_IsAvailableOffline_AndAnotherScriptIsNot()
    {
        var artifacts = new OcrArtifacts(Options());
        artifacts.Kind.Should().Be("ocr");
        (await artifacts.IsAvailableOfflineAsync("en", Ct)).Should().BeFalse("nothing is cached");

        SeedCached("en");

        (await artifacts.IsAvailableOfflineAsync("en", Ct)).Should().BeTrue();
        (await artifacts.GetRemainingDownloadBytesAsync("en", Ct)).Should().Be(0);
        (await artifacts.IsAvailableOfflineAsync("ko", Ct)).Should().BeFalse("Korean reads with another recognizer, not cached");
    }

    [Fact]
    public async Task Download_WithEverythingCached_KeepsItAvailableOffline()
    {
        SeedCached("en");
        // Downloads disabled: the download must be satisfied from the cache, so this proves nothing is fetched.
        var artifacts = new OcrArtifacts(Options(downloadsDisabled: true));

        await artifacts.DownloadAsync("en", cancellationToken: Ct);

        (await artifacts.IsAvailableOfflineAsync("en", Ct)).Should().BeTrue();
    }

    [Fact]
    public async Task DeletingALanguage_RemovesItsRecognizer_AndKeepsTheSharedDetectionModel()
    {
        SeedCached("en", "ko");
        var artifacts = new OcrArtifacts(Options());

        artifacts.Delete("en").Should().BeTrue();

        (await artifacts.IsAvailableOfflineAsync("en", Ct)).Should().BeFalse("its recognizer is gone");
        (await artifacts.IsAvailableOfflineAsync("ko", Ct)).Should().BeTrue("the detection model and the Korean recognizer stay");
        artifacts.Delete("en").Should().BeFalse("nothing of the English recognizer is left");
    }
}
