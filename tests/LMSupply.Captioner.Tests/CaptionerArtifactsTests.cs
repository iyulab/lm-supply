using LMSupply.Lifecycle;
using LMSupply.Testing;
using System.Text.Json;
using AwesomeAssertions;
using LMSupply.Download;

namespace LMSupply.Captioner.Tests;

/// <summary>
/// <see cref="CaptionerArtifacts"/>, <see cref="LocalCaptioner.IsModelDownloadedAsync"/>, <see cref="LocalCaptioner.DownloadModelAsync"/>
/// and <see cref="LocalCaptioner.DeleteModel"/> over a seeded cache: the repository listing with small sizes, and the files the
/// load picks written at those sizes. Nothing reaches the network.
/// </summary>
public sealed class CaptionerArtifactsTests : IDisposable
{
    private const long FileSize = 64;
    private readonly string _cache = Path.Combine(Path.GetTempPath(), "lmsupply-caption-artifacts-" + Guid.NewGuid().ToString("N"));

    private static CancellationToken Ct => TestContext.Current.CancellationToken;

    public void Dispose()
    {
        try { Directory.Delete(_cache + "-runtime", recursive: true); } catch { /* best-effort temp cleanup */ }
        if (Directory.Exists(_cache))
            Directory.Delete(_cache, recursive: true);
    }

    private CaptionerOptions Options(bool downloadsDisabled = false) => new()
    {
        CacheDirectory = _cache,
        Provider = ExecutionProvider.Cpu,
        DisableAutoDownload = downloadsDisabled,
    };

    // The files a load of "fast" (ViT-GPT2) picks with these options, under the model's subfolder.
    private string[] FastModelFiles()
    {
        var model = LocalCaptioner.Registry.Resolve("fast");
        var suffix = LocalCaptioner.SelectVariantSuffix(model, Options());
        return LocalCaptioner.GetRequiredFiles(model, suffix).Select(f => $"{model.Subfolder}/{f}").ToArray();
    }

    private void SeedCachedFastModel(bool skipOneFile = false)
    {
        var model = LocalCaptioner.Registry.Resolve("fast");
        var files = FastModelFiles();
        var listing = Path.Combine(_cache, "models--" + model.RepoId.Replace("/", "--"), ".lmsupply", "listings", "main.json");
        Directory.CreateDirectory(Path.GetDirectoryName(listing)!);
        File.WriteAllText(listing, JsonSerializer.Serialize(files.Select(f => new { path = f, type = "file", size = FileSize })));

        var snapshot = CacheManager.GetModelDirectory(_cache, model.RepoId);
        foreach (var file in skipOneFile ? files.Skip(1) : files)
        {
            var path = Path.Combine(snapshot, file.Replace('/', Path.DirectorySeparatorChar));
            Directory.CreateDirectory(Path.GetDirectoryName(path)!);
            File.WriteAllBytes(path, new byte[FileSize]);
        }
    }

    [Fact]
    public async Task ACachedModel_IsAvailableOffline_AndAnEmptyCacheIsNot()
    {
        using var runtime = BundledRuntime.Use(_cache + "-runtime");
        var artifacts = new CaptionerArtifacts(Options());
        artifacts.Kind.Should().Be("captioner");
        (await artifacts.IsAvailableOfflineAsync("fast", Ct)).Should().BeFalse("nothing is cached");
        Directory.Exists(_cache).Should().BeFalse("the check reads the cache only and writes nothing");

        SeedCachedFastModel();

        (await artifacts.IsAvailableOfflineAsync("fast", Ct)).Should().BeTrue();
        (await artifacts.GetRemainingDownloadAsync("fast", Ct)).ModelBytes.Should().Be(0);
    }

    [Fact]
    public async Task ACachedModel_WhoseRuntimeIsNotOnTheMachine_IsNotAvailableOffline()
    {
        SeedCachedFastModel();
        var artifacts = new CaptionerArtifacts(Options());

        using (BundledRuntime.Use(_cache + "-runtime", present: false))
        {
            (await artifacts.IsAvailableOfflineAsync("fast", Ct)).Should().BeFalse("a load would still need the ONNX Runtime");
            (await artifacts.GetRemainingDownloadAsync("fast", Ct)).Should().Be(new RemainingDownload(0, null),
                "an application runtime directory that lacks it is not something a load downloads");
        }

        using (BundledRuntime.Use(_cache + "-runtime"))
        {
            (await artifacts.IsAvailableOfflineAsync("fast", Ct)).Should().BeTrue();
            (await artifacts.GetRemainingDownloadAsync("fast", Ct)).TotalBytes.Should().Be(0);
        }
    }

    [Fact]
    public async Task AMissingFile_IsNotAvailableOffline()
    {
        using var runtime = BundledRuntime.Use(_cache + "-runtime");
        SeedCachedFastModel(skipOneFile: true);

        (await LocalCaptioner.IsModelDownloadedAsync("fast", Options(), Ct)).Should().BeFalse("one file the load opens is missing");
    }

    [Fact]
    public async Task Download_WithEverythingCached_KeepsItAvailableOffline()
    {
        using var runtime = BundledRuntime.Use(_cache + "-runtime");
        SeedCachedFastModel();
        // Downloads disabled: the download must be satisfied from the cache, so this proves nothing is fetched.
        var artifacts = new CaptionerArtifacts(Options(downloadsDisabled: true));

        await artifacts.DownloadAsync("fast", cancellationToken: Ct);

        (await artifacts.IsAvailableOfflineAsync("fast", Ct)).Should().BeTrue();
    }

    [Fact]
    public async Task Delete_RemovesThisLibrarysFiles_SoTheModelIsNoLongerAvailableOffline()
    {
        using var runtime = BundledRuntime.Use(_cache + "-runtime");
        SeedCachedFastModel();
        var artifacts = new CaptionerArtifacts(Options());

        artifacts.Delete("fast").Should().BeTrue();

        (await artifacts.IsAvailableOfflineAsync("fast", Ct)).Should().BeFalse();
        artifacts.Delete("fast").Should().BeFalse("nothing is left to delete");
    }

    [Fact]
    public void Delete_OfALocalDirectory_KeepsIt()
    {
        var dir = Path.Combine(_cache, "local-captioner");
        Directory.CreateDirectory(dir);
        File.WriteAllBytes(Path.Combine(dir, "encoder_model.onnx"), [1]);

        LocalCaptioner.DeleteModel(dir, Options()).Should().BeFalse();

        File.Exists(Path.Combine(dir, "encoder_model.onnx")).Should().BeTrue();
    }
}
