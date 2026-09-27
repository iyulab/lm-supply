using System.Net;
using System.Text;
using AwesomeAssertions;
using LMSupply.Core.Download;

namespace LMSupply.Core.Tests.Download;

/// <summary>
/// A repository is listed with one recursive request per page of the tree API, following the <c>Link</c> header — not one
/// request per directory. A voice collection with hundreds of directories used to take about a minute and exhaust the
/// anonymous API quota (500 requests per 5 minutes per IP) in a single load.
/// </summary>
public sealed class RepositoryListingPaginationTests : IDisposable
{
    private readonly string _cache = Path.Combine(Path.GetTempPath(), "lmsupply-listing-" + Guid.NewGuid().ToString("N"));

    public void Dispose()
    {
        if (Directory.Exists(_cache))
            Directory.Delete(_cache, recursive: true);
    }

    private sealed class PagedTree : HttpMessageHandler
    {
        public List<Uri> Requests { get; } = [];

        protected override Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken)
        {
            Requests.Add(request.RequestUri!);
            var second = request.RequestUri!.Query.Contains("cursor=p2", StringComparison.Ordinal);
            var body = second
                ? """[{"type":"file","path":"en/en_US/lessac/medium/voice.onnx","size":63201294}]"""
                : """
                  [{"type":"directory","path":"en","size":0},
                   {"type":"directory","path":"en/en_US","size":0},
                   {"type":"file","path":"README.md","size":10},
                   {"type":"file","path":"en/en_US/lessac/medium/voice.onnx.json","size":4885}]
                  """;
            var response = new HttpResponseMessage(HttpStatusCode.OK)
            {
                Content = new StringContent(body, Encoding.UTF8, "application/json"),
            };
            if (!second)
            {
                response.Headers.TryAddWithoutValidation("Link",
                    "<https://huggingface.co/api/models/org/voices/tree/main?expand=false&recursive=true&limit=1000&cursor=p2>; rel=\"next\"");
            }
            return Task.FromResult(response);
        }
    }

    [Fact]
    public async Task ListsEveryPageWithOneRecursiveRequestEach()
    {
        using var handler = new PagedTree();
        using var service = new ModelDiscoveryService(_cache, hfToken: null, handler);

        var files = await service.ListRepositoryFilesAsync("org/voices", cancellationToken: TestContext.Current.CancellationToken);

        files.Select(f => f.Path).Should().BeEquivalentTo(
            "README.md", "en/en_US/lessac/medium/voice.onnx.json", "en/en_US/lessac/medium/voice.onnx");
        files.Should().OnlyContain(f => f.IsFile, "directories are structure, not files to download");
        handler.Requests.Should().HaveCount(2, "one request per page, none per directory");
        handler.Requests[0].Query.Should().Contain("recursive=true");
        handler.Requests[1].Query.Should().Contain("cursor=p2");
    }

    [Theory]
    [InlineData(null, null)]
    [InlineData("<https://h/a?cursor=x>; rel=\"next\"", "https://h/a?cursor=x")]
    [InlineData("<https://h/prev>; rel=\"prev\", <https://h/next>; rel=\"next\"", "https://h/next")]
    [InlineData("<https://h/prev>; rel=\"prev\"", null)]
    public void NextPageUrl_ReadsOnlyTheNextRelation(string? link, string? expected)
    {
        using var response = new HttpResponseMessage(HttpStatusCode.OK);
        if (link is not null)
            response.Headers.TryAddWithoutValidation("Link", link);

        ModelDiscoveryService.NextPageUrl(response).Should().Be(expected);
    }
}
