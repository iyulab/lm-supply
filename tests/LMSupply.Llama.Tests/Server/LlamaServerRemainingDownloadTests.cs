using System.Net;
using AwesomeAssertions;
using LMSupply.Llama.Server;
using Xunit;

namespace LMSupply.Llama.Tests.Server;

/// <summary>
/// <see cref="LlamaServerUpdateService.GetRemainingDownloadBytesAsync"/>: what acquiring the server would still fetch — read
/// from the cache and the release listing, never by downloading, and null rather than a guess when the listing cannot tell.
/// </summary>
public sealed class LlamaServerRemainingDownloadTests : IDisposable
{
    private readonly string _dir = Path.Combine(Path.GetTempPath(), "lmsupply-remaining-" + Guid.NewGuid().ToString("N"));

    public LlamaServerRemainingDownloadTests() => Directory.CreateDirectory(_dir);

    public void Dispose()
    {
        try { Directory.Delete(_dir, recursive: true); } catch { /* best-effort temp cleanup */ }
    }

    private static CancellationToken Ct => TestContext.Current.CancellationToken;

    [Fact]
    public async Task An_external_binary_leaves_nothing_to_download()
    {
        var binary = Path.Combine(_dir, FakeReleaseAssets.ServerExecutableName);
        File.WriteAllBytes(binary, "external"u8.ToArray());
        var handler = new ListingHandler(releaseJson: null);
        using var http = new HttpClient(handler);
        await using var service = new LlamaServerUpdateService(new LlamaServerUpdateOptions { CacheDirectory = _dir, ServerBinaryPath = binary }, http);

        (await service.GetRemainingDownloadBytesAsync(LlamaServerBackend.Cpu, Ct)).Should().Be(0);
        handler.Calls.Should().Be(0);
    }

    [Fact]
    public async Task A_cached_build_leaves_nothing_to_download_and_reads_no_listing()
    {
        var versionDir = Path.Combine(_dir, "b9898", "cpu");
        Directory.CreateDirectory(versionDir);
        File.WriteAllBytes(Path.Combine(versionDir, FakeReleaseAssets.ServerExecutableName), "cached"u8.ToArray());
        var handler = new ListingHandler(releaseJson: null);
        using var http = new HttpClient(handler);
        await using var service = new LlamaServerUpdateService(new LlamaServerUpdateOptions { CacheDirectory = _dir, PinnedVersion = "b9898" }, http);

        (await service.GetRemainingDownloadBytesAsync(LlamaServerBackend.Cpu, Ct)).Should().Be(0);
        handler.Calls.Should().Be(0, "a CPU build needs no companion, so the cache answers alone");
    }

    [Fact]
    public async Task A_missing_build_counts_the_archive_the_listing_states()
    {
        var name = FakeReleaseAssets.CpuAssetName("b9898");
        var handler = new ListingHandler($$"""{ "tag_name": "b9898", "assets": [ { "name": "{{name}}", "browser_download_url": "https://fake.local/{{name}}", "size": 31457280 } ] }""");
        using var http = new HttpClient(handler);
        await using var service = new LlamaServerUpdateService(new LlamaServerUpdateOptions { CacheDirectory = _dir, PinnedVersion = "b9898" }, http);

        (await service.GetRemainingDownloadBytesAsync(LlamaServerBackend.Cpu, Ct)).Should().Be(31_457_280);
        Directory.EnumerateFileSystemEntries(_dir).Should().BeEmpty("nothing is downloaded");
    }

    [Fact]
    public async Task A_listing_that_states_no_size_is_unknown_not_zero()
    {
        var name = FakeReleaseAssets.CpuAssetName("b9898");
        var handler = new ListingHandler($$"""{ "tag_name": "b9898", "assets": [ { "name": "{{name}}", "browser_download_url": "https://fake.local/{{name}}" } ] }""");
        using var http = new HttpClient(handler);
        await using var service = new LlamaServerUpdateService(new LlamaServerUpdateOptions { CacheDirectory = _dir, PinnedVersion = "b9898" }, http);

        (await service.GetRemainingDownloadBytesAsync(LlamaServerBackend.Cpu, Ct)).Should().BeNull();
    }

    [Theory]
    [InlineData("b9898")]
    [InlineData(null)] // the latest build: the release listing itself is unreachable
    public async Task An_unreachable_listing_is_unknown(string? pinned)
    {
        using var http = new HttpClient(new ListingHandler(releaseJson: null));
        await using var service = new LlamaServerUpdateService(new LlamaServerUpdateOptions { CacheDirectory = _dir, PinnedVersion = pinned, UpdateOnWarmup = false }, http);

        (await service.GetRemainingDownloadBytesAsync(LlamaServerBackend.Cpu, Ct)).Should().BeNull();
    }

    /// <summary>Serves one release's tag listing, or answers every request 503 without one.</summary>
    private sealed class ListingHandler(string? releaseJson) : HttpMessageHandler
    {
        public int Calls { get; private set; }

        protected override Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken)
        {
            Calls++;
            return Task.FromResult(releaseJson is not null && request.RequestUri!.ToString().Contains("/releases/tags/", StringComparison.Ordinal)
                ? new HttpResponseMessage(HttpStatusCode.OK) { Content = new StringContent(releaseJson) }
                : new HttpResponseMessage(HttpStatusCode.ServiceUnavailable));
        }
    }
}
