using System.Net;
using System.Text;
using System.Text.Json;
using AwesomeAssertions;
using LMSupply.Generator.Gguf;
using Xunit;

namespace LMSupply.Generator.Tests;

/// <summary>
/// <see cref="LocalGenerator.DescribeAsync(string, GeneratorOptions?, CancellationToken)"/> answers "what is this id and does it run here" before a
/// download: the facts of a catalog entry for any repository id, and a distinct answer for an id that does not exist, one with
/// nothing to load, and one that could not be checked. Network-free (a fake hub over the listing endpoint).
/// </summary>
public sealed class LocalGeneratorDescribeTests : IDisposable
{
    private const long GB = 1_000_000_000L;
    private readonly string _cache = Path.Combine(Path.GetTempPath(), "lmsupply-describe-" + Guid.NewGuid().ToString("N"));

    public void Dispose()
    {
        if (Directory.Exists(_cache))
            Directory.Delete(_cache, recursive: true);
    }

    private GeneratorOptions Options(bool offline = false) => new() { CacheDirectory = _cache, DisableAutoDownload = offline };

    private Task<ModelProbe> Describe(string id, HttpMessageHandler hub, bool offline = false)
        => LocalGenerator.DescribeAsync(id, Options(offline), hub, TestContext.Current.CancellationToken);

    [Fact]
    public async Task ARepositoryTheCatalogDoesNotList_IsDescribedFromItsListing()
    {
        var hub = new FakeHub().Repo("example-org/Some-Model-GGUF", ("Some-Model-Q4_K_M.gguf", 5 * GB), ("Some-Model-Q8_0.gguf", 9 * GB), ("README.md", 2_000));

        var probe = await Describe("example-org/Some-Model-GGUF", hub);

        probe.Status.Should().Be(ModelProbeStatus.Found);
        probe.Description!.Backend.Should().Be(ModelBackend.Gguf);
        probe.Description.DisplayName.Should().Be("Some-Model", "the -GGUF suffix is not part of a model's name");
        probe.Description.License.Should().BeNull("the catalog does not know it, and a licence is never guessed");
        probe.KnownIssues.Should().BeEmpty();
        new[] { 5 * GB, 9 * GB }.Should().Contain(probe.DownloadBytes, "the file the load picks on this host");
        probe.Judgement.Should().NotBeNull();
        probe.Judgement!.Model.EstimatedSizeBytes.Should().Be(probe.DownloadBytes, "the verdict is for the file the load takes");
    }

    [Fact]
    public async Task ACatalogAlias_IsDescribedWithItsCatalogFacts()
    {
        var info = GgufModelRegistry.Resolve("gguf:qwen3-default")!;
        var hub = new FakeHub().Repo(info.RepoId, (info.DefaultFile, info.EstimatedSizeBytes ?? 3 * GB));

        var probe = await Describe("gguf:qwen3-default", hub);

        probe.Status.Should().Be(ModelProbeStatus.Found);
        probe.Description!.DisplayName.Should().Be(info.DisplayName);
        probe.Description.ResolvedId.Should().Be(info.RepoId);
        probe.Description.CatalogEntry.Should().NotBeNull();
        probe.KnownIssues.Should().Contain(GgufModelKnownIssues.ThinkingEnabledByDefault);
        probe.DownloadBytes.Should().Be(info.EstimatedSizeBytes ?? 3 * GB);
        probe.Judgement!.Model.RepoId.Should().Be(info.RepoId);
    }

    [Fact]
    public async Task ARepositoryThatDoesNotExist_IsNotFound()
    {
        var probe = await Describe("nobody/no-such-model-zz", new FakeHub());

        probe.Status.Should().Be(ModelProbeStatus.NotFound);
        probe.Detail.Should().NotBeNullOrWhiteSpace();
        probe.Description.Should().BeNull();
        probe.DownloadBytes.Should().Be(0);
        probe.Judgement.Should().BeNull();
    }

    [Fact]
    public async Task AHubThatAnswersAnonymousRequestsForAMissingRepositoryWith401_IsNotFound()
    {
        var probe = await Describe("nobody/no-such-model-zz", new FakeHub { Missing = HttpStatusCode.Unauthorized });

        probe.Status.Should().Be(ModelProbeStatus.NotFound);
        probe.Detail.Should().Contain("HF_TOKEN");
    }

    [Fact]
    public async Task ARepositoryWithNothingToLoad_IsNoLoadableFiles_AndStillNamed()
    {
        var hub = new FakeHub().Repo("example-org/Docs-Only-GGUF", ("README.md", 1_000), ("LICENSE", 500));

        var probe = await Describe("example-org/Docs-Only-GGUF", hub);

        probe.Status.Should().Be(ModelProbeStatus.NoLoadableFiles);
        probe.Description!.DisplayName.Should().Be("Docs-Only");
        probe.Detail.Should().Contain("No GGUF files");
    }

    [Fact]
    public async Task AHubThatCannotBeReached_IsUnreachable_NotNotFound()
    {
        var probe = await Describe("example-org/Some-Model-GGUF", new Refusing());

        probe.Status.Should().Be(ModelProbeStatus.Unreachable);
        probe.Detail.Should().NotBeNullOrWhiteSpace();
    }

    [Fact]
    public async Task AHubThatAnswersWithAServerError_IsUnreachable()
    {
        var probe = await Describe("example-org/Some-Model-GGUF", new FakeHub { Missing = HttpStatusCode.InternalServerError });

        probe.Status.Should().Be(ModelProbeStatus.Unreachable);
    }

    [Fact]
    public async Task WithDownloadsDisabled_ARepositoryThatWasNeverListed_IsUnreachableNotNotFound()
    {
        var probe = await Describe("example-org/Some-Model-GGUF", new Refusing(), offline: true);

        probe.Status.Should().Be(ModelProbeStatus.Unreachable, "an empty cache says nothing about the repository");
    }

    [Fact]
    public async Task ACancelledProbe_Throws_InsteadOfAnsweringUnreachable()
    {
        using var cts = new CancellationTokenSource();
        await cts.CancelAsync();

        var act = () => LocalGenerator.DescribeAsync("example-org/Some-Model-GGUF", Options(), new Refusing(), cts.Token);

        await act.Should().ThrowAsync<OperationCanceledException>();
    }

    [Fact]
    public async Task AnOnnxModel_IsDescribedWithoutAJudgement()
    {
        var entry = DefaultGeneratorModels.Phi4Mini;
        var folder = entry.Subfolder!;
        var hub = new FakeHub().Repo(entry.ModelId,
            ($"{folder}/model.onnx", 100_000), ($"{folder}/model.onnx.data", 2 * GB), ($"{folder}/genai_config.json", 2_000),
            ($"{folder}/tokenizer.json", 3_000));

        var probe = await Describe(entry.ModelId, hub);

        probe.Status.Should().Be(ModelProbeStatus.Found, probe.Detail);
        probe.Description!.Backend.Should().Be(ModelBackend.Onnx);
        probe.Description.DisplayName.Should().Be(entry.DisplayName);
        probe.Judgement.Should().BeNull("ONNX loading does not use the GGUF memory rule, and a verdict from another rule is not offered as the same one");
        probe.DownloadBytes.Should().BeGreaterThan(2 * GB);
    }

    [Fact]
    public async Task AGgufFileOnLocalDisk_IsFoundWithNothingToDownload()
    {
        Directory.CreateDirectory(_cache);
        var path = Path.Combine(_cache, "local-model-Q4_K_M.gguf");
        await File.WriteAllBytesAsync(path, new byte[1_000_000], TestContext.Current.CancellationToken);

        var probe = await Describe(path, new Refusing());

        probe.Status.Should().Be(ModelProbeStatus.Found);
        probe.DownloadBytes.Should().Be(0);
        probe.Description!.DisplayName.Should().Be("local-model-Q4_K_M");
        probe.Judgement!.Model.QuantizationType.Should().Be("Q4_K_M");
    }

    // Answers the repository listing the downloaders read: {base}/api/models/{repo}/tree/main
    private sealed class FakeHub : HttpMessageHandler
    {
        private readonly Dictionary<string, string> _listings = new(StringComparer.OrdinalIgnoreCase);

        /// <summary>What a repository the hub does not hold is answered with.</summary>
        public HttpStatusCode Missing { get; init; } = HttpStatusCode.NotFound;

        public FakeHub Repo(string repoId, params (string Path, long Size)[] files)
        {
            // By hand: this test project runs with reflection-based JSON switched off.
            _listings[repoId] = "[" + string.Join(",", files.Select(f =>
                FormattableString.Invariant($"{{\"type\":\"file\",\"path\":\"{JsonEncodedText.Encode(f.Path)}\",\"size\":{f.Size}}}"))) + "]";
            return this;
        }

        protected override Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken)
        {
            var path = request.RequestUri!.AbsolutePath;
            const string marker = "/api/models/";
            var at = path.IndexOf(marker, StringComparison.Ordinal);
            if (at >= 0)
            {
                var rest = path[(at + marker.Length)..];
                var treeAt = rest.IndexOf("/tree/", StringComparison.Ordinal);
                if (treeAt > 0 && _listings.TryGetValue(rest[..treeAt], out var json))
                {
                    return Task.FromResult(new HttpResponseMessage(HttpStatusCode.OK)
                    {
                        Content = new StringContent(json, Encoding.UTF8, "application/json"),
                    });
                }
            }

            return Task.FromResult(new HttpResponseMessage(Missing));
        }
    }

    private sealed class Refusing : HttpMessageHandler
    {
        protected override Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken)
            => throw new HttpRequestException(HttpRequestError.ConnectionError, "No connection could be made because the target machine actively refused it.");
    }
}
