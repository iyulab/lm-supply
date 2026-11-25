using System.Net;
using AwesomeAssertions;
using LMSupply.Download;
using LMSupply.Exceptions;

namespace LMSupply.Embedder.Tests;

/// <summary>
/// <see cref="LocalEmbedder.DownloadModelAsync"/> is how a host fills a cache ahead of an offline load (a container
/// build, a consent step), so it has to fetch what that load opens. These facts run on an isolated cache against a
/// stand-in hub: a fact on the shared cache passes as soon as anything else has loaded the model once.
/// </summary>
public sealed class PrefetchMatchesLoadTests : IDisposable
{
    private const string Repo = "acme/uncatalogued-embedder";

    private readonly string _cacheDir = Path.Combine(Path.GetTempPath(), "lmsupply-prefetch-" + Guid.NewGuid().ToString("N"));

    private static CancellationToken Ct => TestContext.Current.CancellationToken;

    // A sentence-transformers repository: the graph under onnx/, everything that describes it at the root.
    private static readonly Dictionary<string, int> Files = new(StringComparer.Ordinal)
    {
        ["onnx/model.onnx"] = 4000,
        ["config.json"] = 30,
        ["tokenizer.json"] = 90,
        ["tokenizer_config.json"] = 40,
        ["special_tokens_map.json"] = 12,
        ["vocab.txt"] = 55,
        ["modules.json"] = 21,
        ["sentence_bert_config.json"] = 17,
        ["config_sentence_transformers.json"] = 19,
        ["1_Pooling/config.json"] = 23,
    };

    public void Dispose()
    {
        if (Directory.Exists(_cacheDir))
            Directory.Delete(_cacheDir, recursive: true);
    }

    [Fact]
    public async Task APrefetchedRepositoryTheCatalogDoesNotKnow_ReadsAsDownloaded()
    {
        var options = new EmbedderOptions { CacheDirectory = _cacheDir, Provider = ExecutionProvider.Cpu };
        using var downloader = new HuggingFaceDownloader(_cacheDir, new Hub());

        var directory = await LocalEmbedder.DownloadRepositoryFilesAsync(downloader, Repo, options.Clone(), progress: null, Ct);

        // The files a load reads beside the graph keep their place at the repository root.
        File.Exists(Path.Combine(directory, "onnx", "model.onnx")).Should().BeTrue();
        File.Exists(Path.Combine(directory, "config.json")).Should().BeTrue();
        File.Exists(Path.Combine(directory, "1_Pooling", "config.json")).Should().BeTrue();
        // Asked without the network, the same question a host asks before an offline load.
        (await LocalEmbedder.IsModelDownloadedAsync(Repo, options, Ct)).Should().BeTrue();
    }

    // The load follows a built-in GGUF alias to its repository before anything else; so must the prefetch. Offline on an
    // empty cache, both can only say that the repository is not cached.
    [Fact]
    public async Task ABuiltInGgufAlias_IsFollowedByThePrefetchAsByTheLoad()
    {
        var options = new EmbedderOptions { CacheDirectory = _cacheDir, DisableAutoDownload = true };

        var prefetch = () => LocalEmbedder.DownloadModelAsync("embeddinggemma-2", options, cancellationToken: Ct);

        await prefetch.Should().ThrowAsync<ModelNotFoundException>().WithMessage("*downloads are disabled*");
    }

    // And so must the delete: what a load of the alias cached is what deleting the alias removes.
    [Fact]
    public void ABuiltInGgufAlias_IsFollowedByTheDeleteAsByTheLoad()
    {
        // Where the GGUF download writes when the revision does not resolve to a commit.
        var snapshot = Path.Combine(_cacheDir, "models--ggml-org--embeddinggemma-2-GGUF", "snapshots", "main");
        Directory.CreateDirectory(snapshot);
        File.WriteAllText(Path.Combine(snapshot, "embeddinggemma-2-Q8_0.gguf"), "weights");

        var deleted = LocalEmbedder.DeleteModel("embeddinggemma-2", new EmbedderOptions { CacheDirectory = _cacheDir });

        deleted.Should().BeTrue();
        File.Exists(Path.Combine(snapshot, "embeddinggemma-2-Q8_0.gguf")).Should().BeFalse();
    }

    /// <summary>Serves the listing and every listed file at its listed length.</summary>
    private sealed class Hub : HttpMessageHandler
    {
        protected override Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken)
        {
            var path = Uri.UnescapeDataString(request.RequestUri!.AbsolutePath);

            if (path == $"/api/models/{Repo}/tree/main")
            {
                var json = "[" + string.Join(",", Files.Select(f => $$"""{"path":"{{f.Key}}","type":"file","size":{{f.Value}}}""")) + "]";
                return Respond(HttpStatusCode.OK, System.Text.Encoding.UTF8.GetBytes(json));
            }

            var prefix = $"/{Repo}/resolve/main/";
            if (path.StartsWith(prefix, StringComparison.Ordinal) && Files.TryGetValue(path[prefix.Length..], out var size))
                return Respond(HttpStatusCode.OK, Enumerable.Repeat((byte)'x', size).ToArray());

            return Respond(HttpStatusCode.NotFound, []);
        }

        private static Task<HttpResponseMessage> Respond(HttpStatusCode status, byte[] body) =>
            Task.FromResult(new HttpResponseMessage(status) { Content = new ByteArrayContent(body) });
    }
}
