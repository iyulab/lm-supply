using System.Net;
using System.Text;
using System.Text.Json;
using AwesomeAssertions;
using LMSupply.Llama.Server;

namespace LMSupply.Llama.Tests.Server;

/// <summary>
/// A streamed chat asks for the server's prompt-processing progress. The stream below is llama-server b11669's,
/// verbatim apart from the dropped <c>id</c>, <c>created</c>, <c>model</c> and <c>system_fingerprint</c> fields: a
/// 2526-token prompt whose first 1536 tokens were in the prompt cache, read in batches of 512, with a keep-alive comment
/// line the server sends while a batch runs.
/// </summary>
public class LlamaServerClientPromptProgressTests
{
    private static CancellationToken Ct => TestContext.Current.CancellationToken;

    private sealed class FixedResponseHandler(string body) : HttpMessageHandler
    {
        public string? RequestBody { get; private set; }

        protected override async Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken)
        {
            RequestBody = request.Content is null ? null : await request.Content.ReadAsStringAsync(cancellationToken);
            return new HttpResponseMessage(HttpStatusCode.OK) { Content = new StringContent(body, Encoding.UTF8, "text/event-stream") };
        }
    }

    private static readonly ChatCompletionMessage[] Question = [ChatCompletionMessage.User("Say OK.")];

    private const string ChatStream =
        "data: {\"choices\":[{\"finish_reason\":null,\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":null}}],\"object\":\"chat.completion.chunk\",\"prompt_progress\":{\"total\":2526,\"cache\":1536,\"processed\":1536,\"time_ms\":0}}\n\n" +
        ":\n\n" +
        "data: {\"choices\":[{\"finish_reason\":null,\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":null}}],\"object\":\"chat.completion.chunk\",\"prompt_progress\":{\"total\":2526,\"cache\":1536,\"processed\":2014,\"time_ms\":43159}}\n\n" +
        "data: {\"choices\":[{\"finish_reason\":null,\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":null}}],\"object\":\"chat.completion.chunk\",\"prompt_progress\":{\"total\":2526,\"cache\":1536,\"processed\":2522,\"time_ms\":79998}}\n\n" +
        "data: {\"choices\":[{\"finish_reason\":null,\"index\":0,\"delta\":{\"content\":\"OK\"}}],\"object\":\"chat.completion.chunk\"}\n\n" +
        "data: {\"choices\":[{\"finish_reason\":\"stop\",\"index\":0,\"delta\":{}}],\"object\":\"chat.completion.chunk\",\"timings\":{\"cache_n\":1536,\"prompt_n\":990,\"prompt_ms\":80956.27,\"prompt_per_second\":12.228824277600733,\"predicted_n\":2,\"predicted_ms\":237.385,\"predicted_per_second\":4.212566084630453}}\n\n" +
        "data: [DONE]\n\n";

    private static async Task<(List<ChatStreamData> Data, FixedResponseHandler Handler)> StreamChatAsync()
    {
        var handler = new FixedResponseHandler(ChatStream);
        var client = new LlamaServerClient("http://localhost:9999", new HttpClient(handler));
        var data = new List<ChatStreamData>();
        await foreach (var d in client.GenerateChatStreamAsync(Question, new ChatCompletionOptions(), Ct))
            data.Add(d);
        return (data, handler);
    }

    [Fact]
    public async Task StreamedChat_AsksForPromptProgress()
    {
        var (_, handler) = await StreamChatAsync();

        JsonDocument.Parse(handler.RequestBody!).RootElement.GetProperty("return_progress").GetBoolean().Should().BeTrue();
    }

    [Fact]
    public void NonStreamedChat_DoesNotAskForPromptProgress()
    {
        var request = LlamaServerClient.BuildChatRequest(Question, new ChatCompletionOptions(), stream: false);

        request.ReturnProgress.Should().BeNull("the option only applies to a streamed completion");
    }

    [Fact]
    public async Task EachProgressReport_IsYieldedOnItsOwn_BeforeTheFirstToken()
    {
        var (data, _) = await StreamChatAsync();

        var progress = data.TakeWhile(d => d.TextDelta is null).Select(d => d.PromptProgress).ToList();
        progress.Should().HaveCount(3).And.OnlyContain(p => p != null);
        progress.Select(p => (p!.Total, p.Cache, p.Processed, p.TimeMs))
            .Should().Equal((2526, 1536, 1536, 0.0), (2526, 1536, 2014, 43159.0), (2526, 1536, 2522, 79998.0));
        data.Where(d => d.PromptProgress is not null)
            .Should().OnlyContain(d => d.TextDelta == null && d.FinishReason == null && d.Usage == null && d.Timings == null);
        data.Skip(3).Should().OnlyContain(d => d.PromptProgress == null);
        string.Concat(data.Select(d => d.TextDelta)).Should().Be("OK");
    }
}
