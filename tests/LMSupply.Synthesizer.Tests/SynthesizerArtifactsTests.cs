using AwesomeAssertions;
using LMSupply.Lifecycle;
using LMSupply.Testing;

namespace LMSupply.Synthesizer.Tests;

/// <summary>
/// <see cref="SynthesizerArtifacts"/> and the lifecycle of <see cref="LocalSynthesizer"/>: the size, availability, download
/// and delete of a voice's model and config — in a repository that holds every voice of the catalog, each in a folder of its
/// own — and of the ONNX Runtime it runs on. Network-free: a seeded listing and snapshot in a private cache, and an
/// application runtime directory.
/// </summary>
public sealed class SynthesizerArtifactsTests : IDisposable
{
    private const string Repo = "rhasspy/piper-voices";
    private const string Amy = "en/en_US/amy/medium";        // "quality"
    private const string Semaine = "en/en_GB/semaine/medium"; // "british"
    private readonly string _cache = Path.Combine(Path.GetTempPath(), "lmsupply-synthesizer-artifacts-" + Guid.NewGuid().ToString("N"));

    private static CancellationToken Ct => TestContext.Current.CancellationToken;

    public void Dispose()
    {
        try { Directory.Delete(_cache + "-runtime", recursive: true); } catch { /* best-effort temp cleanup */ }
        try { Directory.Delete(_cache, recursive: true); } catch { /* best-effort temp cleanup */ }
    }

    private SynthesizerOptions Options(bool offline = false) =>
        new() { CacheDirectory = _cache, Provider = ExecutionProvider.Cpu, DisableAutoDownload = offline };

    private string RepoDir => Path.Combine(_cache, "models--" + Repo.Replace("/", "--"));

    private static readonly (string Path, long Size)[] Listing =
    [
        ($"{Amy}/en_US-amy-medium.onnx", 6_000),
        ($"{Amy}/en_US-amy-medium.onnx.json", 50),
        ($"{Semaine}/en_GB-semaine-medium.onnx", 7_000),
        ($"{Semaine}/en_GB-semaine-medium.onnx.json", 60),
    ];

    private void SeedListing()
    {
        var path = Path.Combine(RepoDir, ".lmsupply", "listings", "main.json");
        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        File.WriteAllText(path, "[" + string.Join(",", Listing.Select(f => $$"""{"path":"{{f.Path}}","type":"file","size":{{f.Size}}}""")) + "]");
    }

    private void SeedVoice(string voice, long? modelLength = null)
    {
        foreach (var (file, size) in Listing.Where(f => f.Path.StartsWith(voice + "/", StringComparison.Ordinal)))
        {
            var path = Path.Combine(RepoDir, "snapshots", "main", file);
            Directory.CreateDirectory(Path.GetDirectoryName(path)!);
            File.WriteAllBytes(path, new byte[file.EndsWith(".onnx", StringComparison.Ordinal) ? modelLength ?? size : size]);
        }
    }

    [Fact]
    public async Task The_size_is_the_voices_model_and_config_and_what_is_left_of_them()
    {
        SeedListing();

        (await LocalSynthesizer.GetDownloadSizeBytesAsync("quality", Options(), Ct)).Should().Be(6_050, "the other voice is not this one's");
        (await LocalSynthesizer.GetRemainingDownloadBytesAsync("quality", Options(), Ct)).Should().Be(6_050);
        (await LocalSynthesizer.IsModelDownloadedAsync("quality", Options(), Ct)).Should().BeFalse();

        SeedVoice(Amy, modelLength: 100);
        (await LocalSynthesizer.GetRemainingDownloadBytesAsync("quality", Options(), Ct)).Should().Be(6_000, "a partial file counts in full");

        SeedVoice(Amy);
        (await LocalSynthesizer.GetRemainingDownloadBytesAsync("quality", Options(), Ct)).Should().Be(0);
        (await LocalSynthesizer.IsModelDownloadedAsync("quality", Options(), Ct)).Should().BeTrue();
        (await LocalSynthesizer.IsModelDownloadedAsync("british", Options(), Ct)).Should().BeFalse();
    }

    [Fact]
    public async Task A_cached_voice_is_available_offline_only_with_its_runtime()
    {
        SeedListing();
        SeedVoice(Amy);
        var artifacts = new SynthesizerArtifacts(Options());
        artifacts.Kind.Should().Be("synthesizer");

        using (BundledRuntime.Use(_cache + "-runtime", present: false))
        {
            (await artifacts.IsAvailableOfflineAsync("quality", Ct)).Should().BeFalse();
            (await artifacts.GetRemainingDownloadAsync("quality", Ct)).Should().Be(new RemainingDownload(0, null));
        }

        using (BundledRuntime.Use(_cache + "-runtime"))
        {
            (await artifacts.IsAvailableOfflineAsync("quality", Ct)).Should().BeTrue();
            (await artifacts.GetRemainingDownloadAsync("quality", Ct)).TotalBytes.Should().Be(0);

            // Downloads disabled: the download must be satisfied from the cache, so this proves nothing is fetched
            await new SynthesizerArtifacts(Options(offline: true)).DownloadAsync("quality", cancellationToken: Ct);
        }
    }

    [Fact]
    public async Task A_voice_never_listed_into_this_cache_is_not_downloaded()
    {
        (await LocalSynthesizer.IsModelDownloadedAsync("quality", Options(), Ct)).Should().BeFalse();
        Directory.Exists(_cache).Should().BeFalse("the check reads the cache only and writes nothing");
    }

    [Fact]
    public async Task Deleting_one_voice_keeps_the_others_of_the_repository()
    {
        SeedListing();
        SeedVoice(Amy);
        SeedVoice(Semaine);

        new SynthesizerArtifacts(Options()).Delete("quality").Should().BeTrue();

        (await LocalSynthesizer.IsModelDownloadedAsync("quality", Options(), Ct)).Should().BeFalse();
        (await LocalSynthesizer.IsModelDownloadedAsync("british", Options(), Ct)).Should().BeTrue("another voice of the same repository stays");
        LocalSynthesizer.DeleteModel("quality", Options()).Should().BeFalse("nothing of that voice is left");
    }

    [Fact]
    public void A_voice_on_local_disk_is_not_deleted()
    {
        var local = Path.Combine(_cache + "-runtime", "my-voice");
        Directory.CreateDirectory(local);
        LocalSynthesizer.DeleteModel(local, Options()).Should().BeFalse();
        Directory.Exists(local).Should().BeTrue();
    }
}
