using System.Net;
using System.Text;
using AwesomeAssertions;
using LMSupply.Core.Download;

namespace LMSupply.Core.Tests.Download;

/// <summary>
/// The hub root holds only the Hugging Face layout. Hugging Face's cache scan reads every other entry there as a broken
/// repository ("not a valid HuggingFace cache directory"), so the repository listing this library caches lives in the
/// repository's own private directory, beside the download manifests — where Hugging Face tools do not look, and which
/// they delete together with the repository.
/// </summary>
public sealed class ListingCacheLocationTests : IDisposable
{
    private const string Repo = "acme/listed-model";
    private readonly string _cache = Path.Combine(Path.GetTempPath(), "lmsupply-listing-loc-" + Guid.NewGuid().ToString("N"));

    public void Dispose()
    {
        if (Directory.Exists(_cache))
            Directory.Delete(_cache, recursive: true);
    }

    private sealed class Tree : HttpMessageHandler
    {
        protected override Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken) =>
            Task.FromResult(new HttpResponseMessage(HttpStatusCode.OK)
            {
                Content = new StringContent("""[{"type":"file","path":"model.onnx","size":5}]""", Encoding.UTF8, "application/json"),
            });
    }

    [Fact]
    public async Task AListing_IsCachedInTheRepositorysPrivateDirectory_AndNothingIsAddedAtTheHubRoot()
    {
        using var handler = new Tree();
        using var service = new ModelDiscoveryService(_cache, hfToken: null, handler);

        await service.ListRepositoryFilesAsync(Repo, cancellationToken: TestContext.Current.CancellationToken);

        File.Exists(Path.Combine(_cache, "models--acme--listed-model", ".lmsupply", "listings", "main.json")).Should().BeTrue();
        Directory.EnumerateFileSystemEntries(_cache).Select(Path.GetFileName)
            .Should().Equal(["models--acme--listed-model"], "the hub root gets only the repository directory");
    }

    [Fact]
    public async Task TheHubRootListingCacheOfEarlierVersions_IsRemoved()
    {
        var legacy = Directory.CreateDirectory(Path.Combine(_cache, ModelDiscoveryService.LegacyListingDirectoryName)).FullName;
        File.WriteAllText(Path.Combine(legacy, "acme_listed-model_main.json"), "[]");
        using var handler = new Tree();
        using var service = new ModelDiscoveryService(_cache, hfToken: null, handler);

        await service.ListRepositoryFilesAsync(Repo, cancellationToken: TestContext.Current.CancellationToken);

        Directory.Exists(legacy).Should().BeFalse("it is this library's own directory, and only a cache");
    }

    [Fact]
    public void ARevisionWithSlashes_IsOneFileName() =>
        Path.GetFileName(ModelDiscoveryService.GetListingPath(_cache, Repo, "refs/pr/1")).Should().Be("refs_pr_1.json");
}
