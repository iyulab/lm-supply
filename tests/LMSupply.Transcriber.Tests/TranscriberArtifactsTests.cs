using System.Text.Json;
using AwesomeAssertions;
using LMSupply.Download;
using LMSupply.Transcriber.Models;

namespace LMSupply.Transcriber.Tests;

/// <summary>
/// <see cref="TranscriberArtifacts"/>, <see cref="LocalTranscriber.DownloadModelAsync"/> and <see cref="LocalTranscriber.DeleteModel"/>
/// over a seeded cache (the repository listing with small sizes, the files at those sizes) — nothing reaches the network.
/// </summary>
public sealed class TranscriberArtifactsTests : IDisposable
{
    private readonly string _cache = Path.Combine(Path.GetTempPath(), "lmsupply-stt-artifacts-" + Guid.NewGuid().ToString("N"));

    private static CancellationToken Ct => TestContext.Current.CancellationToken;

    public void Dispose()
    {
        if (Directory.Exists(_cache))
            Directory.Delete(_cache, recursive: true);
    }

    private static readonly (string Path, long Size)[] Listing =
    [
        ("onnx/encoder_model.onnx", 8_300),
        ("onnx/encoder_model_int8.onnx", 2_300),
        ("onnx/decoder_model_merged.onnx", 20_800),
        ("onnx/decoder_model_merged_int8.onnx", 5_200),
        ("config.json", 200),
        ("generation_config.json", 100),
        ("tokenizer.json", 2_000),
        ("preprocessor_config.json", 30),
    ];

    private string SnapshotDir => CacheManager.GetModelDirectory(_cache, TranscriberModels.WhisperBase.Id);

    private void SeedCachedModel()
    {
        var listing = Path.Combine(
            _cache, "models--" + TranscriberModels.WhisperBase.Id.Replace("/", "--"), ".lmsupply", "listings", "main.json");
        Directory.CreateDirectory(Path.GetDirectoryName(listing)!);
        File.WriteAllText(listing, JsonSerializer.Serialize(Listing.Select(f => new { path = f.Path, type = "file", size = f.Size })));
        foreach (var (repoPath, size) in Listing)
        {
            var path = Path.Combine(SnapshotDir, repoPath.Replace('/', Path.DirectorySeparatorChar));
            Directory.CreateDirectory(Path.GetDirectoryName(path)!);
            File.WriteAllBytes(path, new byte[size]);
        }
    }

    private TranscriberOptions Options(bool downloadsDisabled = false, bool preloadDiarization = false) => new()
    {
        CacheDirectory = _cache,
        QuantizationHint = "int8",
        DisableAutoDownload = downloadsDisabled,
        PreloadDiarization = preloadDiarization,
    };

    [Fact]
    public async Task ACachedModel_IsAvailableOffline_AndAnEmptyCacheIsNot()
    {
        var artifacts = new TranscriberArtifacts(Options());
        artifacts.Kind.Should().Be("transcriber");
        (await artifacts.IsAvailableOfflineAsync("default", Ct)).Should().BeFalse();

        SeedCachedModel();

        (await artifacts.IsAvailableOfflineAsync("default", Ct)).Should().BeTrue();
        (await artifacts.GetRemainingDownloadAsync("default", Ct)).ModelBytes.Should().Be(0);
    }

    [Fact]
    public async Task WithDiarizationPreloaded_TheTranscriberKindStillCoversTheModelOnly()
    {
        SeedCachedModel();
        var transcriber = new TranscriberArtifacts(Options(preloadDiarization: true));
        var speakers = new DiarizationArtifacts(Options(preloadDiarization: true));

        (await transcriber.IsAvailableOfflineAsync("default", Ct))
            .Should().BeTrue("the speaker pair is its own kind — transcription is available without it");
        (await transcriber.GetRemainingDownloadAsync("default", Ct)).ModelBytes.Should().Be(0);
        (await speakers.IsAvailableOfflineAsync(DiarizationArtifacts.DefaultModelId, Ct)).Should().BeFalse();
    }

    private void SeedCachedDiarizationPair()
    {
        foreach (var (repo, file) in new[]
                 {
                     (LMSupply.Transcriber.Diarization.SpeakerDiarizer.SegmentationRepo, LMSupply.Transcriber.Diarization.SpeakerDiarizer.SegmentationFile),
                     (LMSupply.Transcriber.Diarization.SpeakerDiarizer.EmbeddingRepo, LMSupply.Transcriber.Diarization.SpeakerDiarizer.EmbeddingFile),
                 })
        {
            var dir = CacheManager.GetModelDirectory(_cache, repo);
            Directory.CreateDirectory(dir);
            File.WriteAllBytes(Path.Combine(dir, file), [1, 2, 3]);
        }
    }

    [Fact]
    public async Task TheDiarizationKind_ReportsThePairApartFromAnyTranscriber()
    {
        var speakers = new DiarizationArtifacts(Options());
        speakers.Kind.Should().Be("diarization");

        (await speakers.IsAvailableOfflineAsync(DiarizationArtifacts.DefaultModelId, Ct)).Should().BeFalse();
        (await speakers.GetRemainingDownloadAsync(DiarizationArtifacts.DefaultModelId, Ct)).ModelBytes
            .Should().Be(LocalTranscriber.DiarizationDownloadSizeBytes);

        SeedCachedDiarizationPair();

        (await speakers.IsAvailableOfflineAsync(DiarizationArtifacts.DefaultModelId, Ct)).Should().BeTrue();
        (await speakers.GetRemainingDownloadAsync(DiarizationArtifacts.DefaultModelId, Ct)).ModelBytes.Should().Be(0);
        (await new TranscriberArtifacts(Options()).IsAvailableOfflineAsync("default", Ct))
            .Should().BeFalse("a cached pair does not make a transcriber model available");
    }

    [Fact]
    public async Task TheDiarizationKind_WithDownloadsDisabled_DownloadsNothing()
    {
        var speakers = new DiarizationArtifacts(Options(downloadsDisabled: true));

        await FluentActions.Awaiting(() => speakers.DownloadAsync(DiarizationArtifacts.DefaultModelId, cancellationToken: Ct))
            .Should().ThrowAsync<LMSupply.Exceptions.ModelNotFoundException>("nothing is cached and downloads are disabled");

        SeedCachedDiarizationPair();
        await speakers.DownloadAsync(DiarizationArtifacts.DefaultModelId, cancellationToken: Ct);
        (await speakers.IsAvailableOfflineAsync(DiarizationArtifacts.DefaultModelId, Ct)).Should().BeTrue();
    }

    [Fact]
    public async Task TheDiarizationKind_DeletesThePairOnly()
    {
        SeedCachedModel();
        SeedCachedDiarizationPair();
        var speakers = new DiarizationArtifacts(Options());

        speakers.Delete(DiarizationArtifacts.DefaultModelId).Should().BeTrue();

        (await speakers.IsAvailableOfflineAsync(DiarizationArtifacts.DefaultModelId, Ct)).Should().BeFalse();
        (await new TranscriberArtifacts(Options()).IsAvailableOfflineAsync("default", Ct)).Should().BeTrue("the transcriber model is kept");
        speakers.Delete(DiarizationArtifacts.DefaultModelId).Should().BeFalse("nothing is left to delete");
    }

    [Fact]
    public async Task TheDiarizationKind_RefusesAnotherModelId()
    {
        var speakers = new DiarizationArtifacts(Options());

        await FluentActions.Awaiting(() => speakers.IsAvailableOfflineAsync("whisper-base", Ct)).Should().ThrowAsync<ArgumentException>();
        FluentActions.Invoking(() => speakers.Delete("whisper-base")).Should().Throw<ArgumentException>();
    }

    [Fact]
    public async Task Download_WithEverythingCached_KeepsItAvailableOffline()
    {
        SeedCachedModel();
        // Downloads disabled: the download must be satisfied from the cache, so this proves nothing is fetched.
        var artifacts = new TranscriberArtifacts(Options(downloadsDisabled: true));

        await artifacts.DownloadAsync("default", cancellationToken: Ct);

        (await artifacts.IsAvailableOfflineAsync("default", Ct)).Should().BeTrue();
    }

    [Fact]
    public async Task Delete_RemovesThisLibrarysFiles_SoTheModelIsNoLongerAvailableOffline()
    {
        SeedCachedModel();
        var artifacts = new TranscriberArtifacts(Options());

        artifacts.Delete("default").Should().BeTrue();

        (await artifacts.IsAvailableOfflineAsync("default", Ct)).Should().BeFalse();
        artifacts.Delete("default").Should().BeFalse("nothing is left to delete");
    }

    [Fact]
    public void Delete_OfALocalDirectory_KeepsIt()
    {
        var dir = Path.Combine(_cache, "local-whisper");
        Directory.CreateDirectory(dir);
        File.WriteAllBytes(Path.Combine(dir, "encoder_model.onnx"), [1]);

        LocalTranscriber.DeleteModel(dir, Options()).Should().BeFalse();

        File.Exists(Path.Combine(dir, "encoder_model.onnx")).Should().BeTrue();
    }
}
