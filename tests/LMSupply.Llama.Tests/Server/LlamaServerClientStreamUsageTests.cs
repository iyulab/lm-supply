using System.Net;
using System.Text;
using AwesomeAssertions;
using LMSupply.Llama.Server;

namespace LMSupply.Llama.Tests.Server;

/// <summary>
/// A streamed completion's token accounting comes from the server. Until 0.77.0 the streamed result was estimated from
/// the visible text, which cannot see a reasoning model's hidden reasoning: one call reported ~60 completion tokens
/// while the server had generated ~480 and stopped at the limit.
/// </summary>
public class LlamaServerClientStreamUsageTests
{
    private static CancellationToken Ct => TestContext.Current.CancellationToken;

    private sealed class SseHandler(string body) : HttpMessageHandler
    {
        public string? RequestBody { get; private set; }

        protected override async Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken)
        {
            RequestBody = request.Content is null ? null : await request.Content.ReadAsStringAsync(cancellationToken);
            return new HttpResponseMessage(HttpStatusCode.OK) { Content = new StringContent(body, Encoding.UTF8, "text/event-stream") };
        }
    }

    private static async Task<(List<ChatStreamData> Data, SseHandler Handler)> StreamAsync(string sse)
    {
        var handler = new SseHandler(sse);
        var client = new LlamaServerClient("http://localhost:9999", new HttpClient(handler));
        var data = new List<ChatStreamData>();
        await foreach (var d in client.GenerateChatStreamAsync([new ChatCompletionMessage { Role = "user", Content = "hi" }], cancellationToken: Ct))
            data.Add(d);
        return (data, handler);
    }

    [Fact]
    public async Task AStreamedRequest_AsksForUsage()
    {
        var (_, handler) = await StreamAsync("data: [DONE]\n\n");

        handler.RequestBody.Should().Contain("\"stream_options\":{\"include_usage\":true}");
    }

    [Fact]
    public async Task UsageOnAChunkWithoutChoices_IsSurfaced()
    {
        // OpenAI's include_usage shape: a final chunk with an empty choices array and the usage object.
        var (data, _) = await StreamAsync(
            "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\n" +
            "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"length\"}]}\n\n" +
            "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":41,\"completion_tokens\":512,\"total_tokens\":553}}\n\n" +
            "data: [DONE]\n\n");

        var usage = data.Single(d => d.Usage is not null).Usage!;
        usage.PromptTokens.Should().Be(41);
        usage.CompletionTokens.Should().Be(512, "the server's count includes the reasoning the text never shows");
    }

    [Fact]
    public async Task LlamaTimings_AreTheFallback_WhenNoUsageObjectIsSent()
    {
        var (data, _) = await StreamAsync(
            "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}],\"timings\":{\"prompt_n\":30,\"predicted_n\":77}}\n\n" +
            "data: [DONE]\n\n");

        var usage = data.Single(d => d.Usage is not null).Usage!;
        usage.PromptTokens.Should().Be(30);
        usage.CompletionTokens.Should().Be(77);
    }

    [Fact]
    public async Task NoAccountingInTheStream_SurfacesNoUsage()
    {
        var (data, _) = await StreamAsync(
            "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"},\"finish_reason\":\"stop\"}]}\n\n" +
            "data: [DONE]\n\n");

        data.Should().OnlyContain(d => d.Usage == null);
    }
}
