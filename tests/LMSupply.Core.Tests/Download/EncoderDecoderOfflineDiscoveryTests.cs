using LMSupply.Core.Download;
using System.Net;
using System.Text;
using LMSupply.Download;

namespace LMSupply.Core.Tests.Download;

/// <summary>
/// An encoder-decoder model downloaded in one quantization is found again offline from the download manifest alone.
/// The manifest lists only the files that were fetched (here the int8 pair), so offline discovery sees a repository
/// without the full-precision files — it must still recognise the export as encoder-decoder and pick the same pair the
/// online load picked, or a load with downloads disabled looks for a file that was never downloaded.
/// </summary>
public sealed class EncoderDecoderOfflineDiscoveryTests : IDisposable
{
    private const string Repo = "acme/speech-model";
    private readonly string _cacheDir = Path.Combine(Path.GetTempPath(), "lmsupply-encdec-" + Guid.NewGuid().ToString("N"));

    private static CancellationToken Ct => TestContext.Current.CancellationToken;

    private static readonly (string Path, string Body)[] RepoFiles =
    [
        ("onnx/encoder_model.onnx", "encoder-fp32-graph"),
        ("onnx/encoder_model_int8.onnx", "encoder-int8"),
        ("onnx/decoder_model_merged.onnx", "decoder-merged-fp32-graph"),
        ("onnx/decoder_model_merged_int8.onnx", "decoder-merged-int8"),
        ("config.json", "{}"),
    ];

    private static ModelPreferences Int8 => new()
    {
        PreferredSubfolder = "onnx",
        QuantizationPriority = ModelPreferences.ForQuantizationHint("int8").QuantizationPriority,
        RequireMatchedQuantization = true
    };

    public void Dispose()
    {
        if (Directory.Exists(_cacheDir))
            Directory.Delete(_cacheDir, recursive: true);
    }

    [Fact]
    public async Task WithOnlyTheDownloadManifest_TheOfflineLoadFindsTheQuantizedPairItDownloaded()
    {
        var online = await DownloadOnlineAsync();
        File.Delete(ModelDiscoveryService.GetListingPath(_cacheDir, Repo, "main"));

        var hub = new Hub();
        using var downloader = new HuggingFaceDownloader(_cacheDir, hub, localFilesOnly: true);
        var (dir, offline) = await downloader.DownloadWithDiscoveryAsync(Repo, Int8, cancellationToken: Ct);

        Assert.Equal(ModelArchitecture.EncoderDecoder, offline.Architecture);
        Assert.Equal(online.EncoderFiles, offline.EncoderFiles);
        Assert.Equal(online.DecoderFiles, offline.DecoderFiles);
        Assert.True(File.Exists(offline.GetEncoderPath(dir)), "the encoder the load opens is on disk");
        Assert.True(File.Exists(offline.GetDecoderPath(dir)), "the decoder the load opens is on disk");
        Assert.Equal(0, hub.Count);
    }

    [Fact]
    public void QuantizedOnlyFiles_AreAnEncoderDecoderModel()
    {
        Assert.True(ModelDiscoveryService.IsEncoderDecoderModel(
            ["onnx/encoder_model_int8.onnx", "onnx/decoder_model_merged_int8.onnx"]));
        Assert.True(ModelDiscoveryService.IsEncoderDecoderModel(
            ["onnx/encoder_model_q4f16.onnx", "onnx/decoder_with_past_model_q4f16.onnx"]));
        Assert.False(ModelDiscoveryService.IsEncoderDecoderModel(
            ["onnx/encoder_model_int8.onnx", "onnx/encoder_model.onnx"]), "two encoders and no decoder");
    }

    // Populates the cache the way an online load does and returns what that load chose; also the positive control for
    // the request counter above.
    private async Task<ModelDiscoveryResult> DownloadOnlineAsync()
    {
        var hub = new Hub(servesModel: true);
        using var downloader = new HuggingFaceDownloader(_cacheDir, hub);

        var (_, discovery) = await downloader.DownloadWithDiscoveryAsync(Repo, Int8, cancellationToken: Ct);

        Assert.Equal(["onnx/encoder_model_int8.onnx"], discovery.EncoderFiles);
        Assert.Equal(["onnx/decoder_model_merged_int8.onnx"], discovery.DecoderFiles);
        Assert.DoesNotContain(hub.Paths, p => p.EndsWith("/encoder_model.onnx", StringComparison.Ordinal));
        Assert.Contains(hub.Paths, p => p.StartsWith("/api/models/", StringComparison.Ordinal));
        return discovery;
    }

    private sealed class Hub(bool servesModel = false) : HttpMessageHandler
    {
        private readonly List<string> _paths = [];

        public int Count
        {
            get
            {
                lock (_paths)
                    return _paths.Count;
            }
        }

        public IReadOnlyList<string> Paths
        {
            get
            {
                lock (_paths)
                    return [.. _paths];
            }
        }

        protected override Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken)
        {
            var path = Uri.UnescapeDataString(request.RequestUri!.AbsolutePath);
            lock (_paths)
                _paths.Add(path);

            if (servesModel && path == $"/api/models/{Repo}/tree/main")
            {
                var listing = string.Join(",", RepoFiles.Select(f =>
                    $$"""{"path":"{{f.Path}}","type":"file","size":{{Encoding.UTF8.GetByteCount(f.Body)}}}"""));
                return Ok("[" + listing + "]");
            }

            if (servesModel)
            {
                foreach (var (file, body) in RepoFiles)
                {
                    if (path == $"/{Repo}/resolve/main/{file}")
                        return Ok(body);
                }
            }

            return Task.FromResult(new HttpResponseMessage(HttpStatusCode.NotFound));
        }

        private static Task<HttpResponseMessage> Ok(string body)
            => Task.FromResult(new HttpResponseMessage(HttpStatusCode.OK) { Content = new ByteArrayContent(Encoding.UTF8.GetBytes(body)) });
    }
}
