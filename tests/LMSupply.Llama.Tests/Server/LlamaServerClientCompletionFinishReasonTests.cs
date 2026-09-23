using System.Net;
using AwesomeAssertions;
using LMSupply.Llama.Server;

namespace LMSupply.Llama.Tests.Server;

/// <summary>
/// The raw <c>/completion</c> stream ends with a chunk that says why it stopped (<c>stop_type</c>). The text-only
/// stream used to break on that chunk and drop the reason; the structured stream carries it as a finish reason.
/// </summary>
public class LlamaServerClientCompletionFinishReasonTests
{
    private static string Sse(params string[] chunks) => string.Concat(chunks.Select(c => $"data: {c}\n\n"));

    private static LlamaServerClient Client(string body) =>
        new("http://localhost:9999", new HttpClient(new FakeHandler(body)) { BaseAddress = new Uri("http://localhost:9999") }, 4096);

    [Theory]
    [InlineData("limit", "length")]
    [InlineData("eos", "stop")]
    [InlineData("word", "stop")]
    public async Task GenerateStreamAsync_LastChunk_CarriesTheFinishReason(string stopType, string expected)
    {
        var client = Client(Sse(
            """{"content":"Hello","stop":false}""",
            """{"content":" world","stop":false}""",
            $$"""{"content":"","stop":true,"stop_type":"{{stopType}}"}"""));

        var chunks = new List<CompletionStreamData>();
        await foreach (var chunk in client.GenerateStreamAsync("prompt", cancellationToken: TestContext.Current.CancellationToken))
            chunks.Add(chunk);

        string.Concat(chunks.Select(c => c.TextDelta)).Should().Be("Hello world");
        chunks[^1].FinishReason.Should().Be(expected);
        chunks.Take(chunks.Count - 1).Should().OnlyContain(c => c.FinishReason == null);
    }

    [Fact]
    public async Task GenerateStreamAsync_AServerThatDoesNotSayWhy_ReportsNoReason_RatherThanGuessing()
    {
        var client = Client(Sse(
            """{"content":"Hi","stop":false}""",
            """{"content":"","stop":true}"""));

        var chunks = new List<CompletionStreamData>();
        await foreach (var chunk in client.GenerateStreamAsync("prompt", cancellationToken: TestContext.Current.CancellationToken))
            chunks.Add(chunk);

        chunks.Should().OnlyContain(c => c.FinishReason == null);
    }

    [Fact]
    public async Task GenerateAsync_TextStream_IsUnchanged()
    {
        var client = Client(Sse(
            """{"content":"Hello","stop":false}""",
            """{"content":" world","stop":false}""",
            """{"content":"","stop":true,"stop_type":"limit"}"""));

        var text = new List<string>();
        await foreach (var token in client.GenerateAsync("prompt", cancellationToken: TestContext.Current.CancellationToken))
            text.Add(token);

        text.Should().Equal("Hello", " world");
    }

    private sealed class FakeHandler(string body) : HttpMessageHandler
    {
        protected override Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken) =>
            Task.FromResult(new HttpResponseMessage(HttpStatusCode.OK) { Content = new StringContent(body) });
    }
}
