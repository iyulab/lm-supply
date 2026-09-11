using System.Net;
using System.Text;
using LMSupply.Download;

namespace LMSupply.Core.Tests.Download;

/// <summary>
/// Where <c>DownloadModelAsync</c> puts a subfolder's files, and what it hands back.
///
/// <para>
/// Repositories commonly keep one model per subfolder under the same file names — one recognizer per
/// script (<c>languages/korean/rec.onnx</c>, <c>languages/latin/rec.onnx</c>), one ONNX variant per
/// execution provider. When the subfolder was dropped from the local path, the second subfolder found
/// the first one's <c>rec.onnx</c> already on disk, skipped its own download, and silently ran the
/// first model. These tests pin the layout the callers already assume: each subfolder under its own
/// directory, and that directory returned.
/// </para>
/// </summary>
public sealed class HuggingFaceDownloaderSubfolderTests : IDisposable
{
    private const string Repo = "acme/multi-model";
    private readonly string _cacheDir = Path.Combine(Path.GetTempPath(), "lmsupply-subfolder-" + Guid.NewGuid().ToString("N"));

    public void Dispose()
    {
        if (Directory.Exists(_cacheDir))
            Directory.Delete(_cacheDir, recursive: true);
    }

    [Fact]
    public async Task TwoSubfoldersWithTheSameFileNames_EachKeepsItsOwnBytes()
    {
        var handler = new FakeHub(new()
        {
            ["languages/korean/rec.onnx"] = "korean-model",
            ["languages/korean/dict.txt"] = "korean-dict",
            ["languages/latin/rec.onnx"] = "latin-model",
            ["languages/latin/dict.txt"] = "latin-dict",
        });
        using var downloader = new HuggingFaceDownloader(_cacheDir, handler);
        var ct = TestContext.Current.CancellationToken;

        var koreanDir = await downloader.DownloadModelAsync(Repo, ["rec.onnx", "dict.txt"], subfolder: "languages/korean", cancellationToken: ct);
        var latinDir = await downloader.DownloadModelAsync(Repo, ["rec.onnx", "dict.txt"], subfolder: "languages/latin", cancellationToken: ct);

        Assert.NotEqual(koreanDir, latinDir);
        Assert.Equal("korean-model", File.ReadAllText(Path.Combine(koreanDir, "rec.onnx")));
        Assert.Equal("latin-model", File.ReadAllText(Path.Combine(latinDir, "rec.onnx")));
        Assert.Equal("latin-dict", File.ReadAllText(Path.Combine(latinDir, "dict.txt")));
        Assert.Contains("languages/latin/rec.onnx", handler.Requested);
    }

    [Fact]
    public async Task ASubfolderDownload_ReturnsThatSubfolder_WithItsManifest()
    {
        var handler = new FakeHub(new() { ["onnx/model.onnx"] = "graph" });
        using var downloader = new HuggingFaceDownloader(_cacheDir, handler);

        var dir = await downloader.DownloadModelAsync(Repo, ["model.onnx"], subfolder: "onnx", cancellationToken: TestContext.Current.CancellationToken);

        Assert.Equal("onnx", Path.GetFileName(dir));
        Assert.True(ModelDirectoryValidator.Validate(dir).IsValid, "the manifest must describe the directory handed back");
    }

    [Fact]
    public async Task ARootFallbackFile_LandsInTheReturnedDirectory()
    {
        // Tokenizer assets usually sit at the repository root while the graph sits in onnx/.
        // The returned directory has to hold both, or the caller cannot find its tokenizer.
        var handler = new FakeHub(new()
        {
            ["onnx/model.onnx"] = "graph",
            ["tokenizer.json"] = "{}",
        });
        using var downloader = new HuggingFaceDownloader(_cacheDir, handler);

        var dir = await downloader.DownloadModelAsync(Repo, ["model.onnx", "tokenizer.json"], subfolder: "onnx", cancellationToken: TestContext.Current.CancellationToken);

        Assert.True(File.Exists(Path.Combine(dir, "model.onnx")));
        Assert.True(File.Exists(Path.Combine(dir, "tokenizer.json")));
    }

    [Fact]
    public async Task NoSubfolder_KeepsTheSnapshotRoot()
    {
        var handler = new FakeHub(new() { ["model.onnx"] = "graph" });
        using var downloader = new HuggingFaceDownloader(_cacheDir, handler);

        var dir = await downloader.DownloadModelAsync(Repo, ["model.onnx"], cancellationToken: TestContext.Current.CancellationToken);

        Assert.Equal(CacheManager.GetModelDirectory(_cacheDir, Repo), dir);
    }

    [Theory]
    [InlineData("../escape")]
    [InlineData("onnx/../../escape")]
    public async Task ASubfolderThatLeavesTheSnapshot_IsRefused(string subfolder)
    {
        using var downloader = new HuggingFaceDownloader(_cacheDir, new FakeHub([]));

        await Assert.ThrowsAsync<InvalidOperationException>(() =>
            downloader.DownloadModelAsync(Repo, ["model.onnx"], subfolder: subfolder, cancellationToken: TestContext.Current.CancellationToken));
    }

    /// <summary>Serves repository paths from a map; anything else is 404.</summary>
    private sealed class FakeHub(Dictionary<string, string> files) : HttpMessageHandler
    {
        private readonly List<string> _requested = [];

        public IReadOnlyList<string> Requested
        {
            get
            {
                lock (_requested)
                    return [.. _requested];
            }
        }

        protected override Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken)
        {
            // https://huggingface.co/{org}/{name}/resolve/{revision}/{path}
            var marker = "/resolve/main/";
            var absolute = Uri.UnescapeDataString(request.RequestUri!.AbsolutePath);
            var path = absolute[(absolute.IndexOf(marker, StringComparison.Ordinal) + marker.Length)..];
            lock (_requested)
                _requested.Add(path);

            return Task.FromResult(files.TryGetValue(path, out var body)
                ? new HttpResponseMessage(HttpStatusCode.OK) { Content = new ByteArrayContent(Encoding.UTF8.GetBytes(body)) }
                : new HttpResponseMessage(HttpStatusCode.NotFound));
        }
    }
}
