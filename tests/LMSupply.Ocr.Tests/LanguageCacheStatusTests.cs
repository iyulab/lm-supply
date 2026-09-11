using AwesomeAssertions;
using LMSupply.Download;
using LMSupply.Exceptions;
using LMSupply.Ocr.Models;

namespace LMSupply.Ocr.Tests;

/// <summary>
/// <see cref="LocalOcr.GetCacheStatusForLanguage"/> answers "would loading this language download anything?"
/// before the load. Since each recognizer sits in its own repository subfolder, a cache that holds one
/// language says nothing about another — the repository as a whole can look complete while the next
/// language's first load still downloads.
/// </summary>
public sealed class LanguageCacheStatusTests : IDisposable
{
    private readonly string _cacheDir = Path.Combine(Path.GetTempPath(), "lmsupply-ocr-cache-" + Guid.NewGuid().ToString("N"));

    private static CancellationToken Ct => TestContext.Current.CancellationToken;

    public void Dispose()
    {
        if (Directory.Exists(_cacheDir))
            Directory.Delete(_cacheDir, recursive: true);
    }

    [Fact]
    public void AnEmptyCache_ListsTheDetectionModel_TheRecognizer_AndItsDictionary()
    {
        var detection = OcrDetectionModelRegistry.Default.Resolve("default");
        var korean = OcrRecognitionModelRegistry.Default.ResolveForLanguage("ko");

        var status = LocalOcr.GetCacheStatusForLanguage("ko", _cacheDir);

        status.IsCached.Should().BeFalse();
        status.RecognitionModel.Should().Be("crnn-korean-v3");
        status.Files.Should().Equal(
            new OcrModelFile(detection.RepoId, "detection/v3", detection.ModelFile, false),
            new OcrModelFile(korean.RepoId, "languages/korean", korean.ModelFile, false),
            new OcrModelFile(korean.RepoId, "languages/korean", korean.DictFile, false));
        status.Missing.Should().HaveCount(3);
    }

    // What happened to a consumer that took "the repository is complete" as "OCR is installed": with only
    // English downloaded, the first Korean load downloaded inside whatever call came first.
    [Fact]
    public async Task EnglishCached_IsCachedForEnglish_ButNotForKorean()
    {
        await SeedAsync(OcrDetectionModelRegistry.Default.Resolve("default"));
        await SeedAsync(OcrRecognitionModelRegistry.Default.ResolveForLanguage("en"));

        LocalOcr.GetCacheStatusForLanguage("en", _cacheDir).IsCached.Should().BeTrue();

        var korean = LocalOcr.GetCacheStatusForLanguage("ko", _cacheDir);
        korean.IsCached.Should().BeFalse();
        korean.Missing.Select(f => f.Subfolder).Should().AllBe("languages/korean");
        korean.Missing.Should().HaveCount(2);
    }

    // Languages of one script share a recognizer: Japanese is read by the Chinese model.
    [Fact]
    public async Task ALanguageSharingACachedRecognizer_IsCached()
    {
        await SeedAsync(OcrDetectionModelRegistry.Default.Resolve("default"));
        await SeedAsync(OcrRecognitionModelRegistry.Default.ResolveForLanguage("zh"));

        LocalOcr.GetCacheStatusForLanguage("ja", _cacheDir).IsCached.Should().BeTrue();
    }

    [Fact]
    public async Task DisableAutoDownload_WithTheLanguageNotCached_FailsTheLoadInsteadOfDownloading()
    {
        var options = new OcrOptions { CacheDirectory = _cacheDir, DisableAutoDownload = true };

        var load = () => LocalOcr.LoadForLanguageAsync("ko", options, cancellationToken: Ct);

        await load.Should().ThrowAsync<ModelNotFoundException>().WithMessage("*downloads are disabled*");
        Directory.GetFiles(_cacheDir, "*.onnx", SearchOption.AllDirectories).Should().BeEmpty();
    }

    private Task SeedAsync(DetectionModelInfo model) => SeedAsync(model.RepoId, model.Subfolder, model.ModelFile);

    private async Task SeedAsync(RecognitionModelInfo model)
    {
        await SeedAsync(model.RepoId, model.Subfolder, model.ModelFile);
        await SeedAsync(model.RepoId, model.Subfolder, model.DictFile);
    }

    // The downloader's layout: <snapshot>/<subfolder>/<file>.
    private async Task SeedAsync(string repoId, string? subfolder, string file)
    {
        var dir = Path.Combine(CacheManager.GetModelDirectory(_cacheDir, repoId), (subfolder ?? "").Replace('/', Path.DirectorySeparatorChar));
        Directory.CreateDirectory(dir);
        await File.WriteAllTextAsync(Path.Combine(dir, file), "content", Ct);
    }
}
