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
    public async Task GenerateStreamAsync_LastChunk_CarriesTheServerTimings()
    {
        // b11146's final /completion chunk on a second call with the same prompt: tokens_evaluated counts the whole
        // prompt (5), while the timings say one token came from the cache and four were evaluated.
        var client = Client(Sse(
            """{"content":" the","stop":false}""",
            """{"content":"","stop":true,"stop_type":"limit","tokens_predicted":20,"tokens_evaluated":5,"tokens_cached":24,"timings":{"cache_n":1,"prompt_n":4,"prompt_ms":25.502,"prompt_per_token_ms":6.3755,"prompt_per_second":156.85044310250177,"predicted_n":20,"predicted_ms":153.301,"predicted_per_token_ms":8.068473684210526,"predicted_per_second":123.93917847894014}}"""));

        var chunks = new List<CompletionStreamData>();
        await foreach (var chunk in client.GenerateStreamAsync("prompt", cancellationToken: TestContext.Current.CancellationToken))
            chunks.Add(chunk);

        var last = chunks[^1];
        last.PromptTokens.Should().Be(5);
        last.Timings.Should().NotBeNull();
        last.Timings!.CacheN.Should().Be(1);
        last.Timings.PromptN.Should().Be(4);
        last.Timings.PredictedMs.Should().Be(153.301);
        last.Timings.PredictedPerSecond.Should().BeApproximately(123.939, 0.001);
        chunks.Take(chunks.Count - 1).Should().OnlyContain(c => c.Timings == null);
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
