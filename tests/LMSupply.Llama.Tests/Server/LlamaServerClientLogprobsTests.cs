using System.Net;
using System.Text;
using System.Text.Json;
using AwesomeAssertions;
using LMSupply.Llama.Server;

namespace LMSupply.Llama.Tests.Server;

/// <summary>
/// Per-token log probabilities on both llama-server paths. The response bodies are llama-server b11429's, verbatim apart
/// from the dropped <c>id</c>, <c>created</c>, <c>model</c>, <c>usage</c>, <c>timings</c> and <c>generation_settings</c>
/// fields: a Qwen3.5 2B model answering a two-option question with thinking off (chat), and continuing
/// "One, two, three," (<c>/completion</c>). The server leaves the values before sampling unless
/// <c>post_sampling_probs</c> is sent, which then renames the fields (<c>prob</c>, <c>top_probs</c>) — so the client
/// must send no such flag and read only <c>logprob</c> / <c>top_logprobs</c>.
/// </summary>
public class LlamaServerClientLogprobsTests
{
    private static CancellationToken Ct => TestContext.Current.CancellationToken;

    private sealed class FixedResponseHandler(string body) : HttpMessageHandler
    {
        public string? RequestBody { get; private set; }
        public string? RequestPath { get; private set; }

        protected override async Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken)
        {
            RequestPath = request.RequestUri?.AbsolutePath;
            RequestBody = request.Content is null ? null : await request.Content.ReadAsStringAsync(cancellationToken);
            return new HttpResponseMessage(HttpStatusCode.OK) { Content = new StringContent(body, Encoding.UTF8, "application/json") };
        }
    }

    private static readonly ChatCompletionMessage[] Question = [ChatCompletionMessage.User("Which planet is red? A) Venus B) Mars. Answer with one letter.")];

    // Non-streamed chat, top_logprobs 2: the answer token and the end-of-generation token, which has no text.
    private const string ChatResponse =
        "{\"choices\":[{\"finish_reason\":\"stop\",\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":\"B\"}," +
        "\"logprobs\":{\"content\":[" +
        "{\"id\":33,\"token\":\"B\",\"bytes\":[66],\"logprob\":-0.034990593791007996,\"top_logprobs\":[" +
        "{\"id\":33,\"token\":\"B\",\"bytes\":[66],\"logprob\":-0.034990593791007996}," +
        "{\"id\":44,\"token\":\"M\",\"bytes\":[77],\"logprob\":-3.7268946170806885}]}," +
        "{\"id\":248046,\"token\":\"\",\"bytes\":[],\"logprob\":-0.00039603206096217036,\"top_logprobs\":[" +
        "{\"id\":248046,\"token\":\"\",\"bytes\":[],\"logprob\":-0.00039603206096217036}," +
        "{\"id\":13,\"token\":\".\",\"bytes\":[46],\"logprob\":-8.3460111618042}]}]}}]," +
        "\"system_fingerprint\":\"b11429-d81235049\",\"object\":\"chat.completion\"}";

    // Streamed chat, top_logprobs 2: the logprobs object sits beside the delta, one token per chunk.
    private const string ChatStream =
        "data: {\"choices\":[{\"finish_reason\":null,\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":null}}],\"object\":\"chat.completion.chunk\"}\n\n" +
        "data: {\"choices\":[{\"finish_reason\":null,\"index\":0,\"delta\":{\"content\":\"B\"},\"logprobs\":{\"content\":[" +
        "{\"id\":33,\"token\":\"B\",\"bytes\":[66],\"logprob\":-0.034990593791007996,\"top_logprobs\":[" +
        "{\"id\":33,\"token\":\"B\",\"bytes\":[66],\"logprob\":-0.034990593791007996}," +
        "{\"id\":44,\"token\":\"M\",\"bytes\":[77],\"logprob\":-3.7268946170806885}]}]}}],\"object\":\"chat.completion.chunk\"}\n\n" +
        "data: {\"choices\":[{\"finish_reason\":\"stop\",\"index\":0,\"delta\":{}}],\"object\":\"chat.completion.chunk\"}\n\n" +
        "data: [DONE]\n\n";

    // Streamed chat with thinking on, top_logprobs 1: reasoning tokens carry log probabilities too.
    private const string ReasoningStream =
        "data: {\"choices\":[{\"finish_reason\":null,\"index\":0,\"delta\":{\"reasoning_content\":\"Thinking\"},\"logprobs\":{\"content\":[" +
        "{\"id\":90700,\"token\":\"Thinking\",\"bytes\":[84,104,105,110,107,105,110,103],\"logprob\":-0.37451669573783875,\"top_logprobs\":[" +
        "{\"id\":90700,\"token\":\"Thinking\",\"bytes\":[84,104,105,110,107,105,110,103],\"logprob\":-0.37451669573783875}]}]}}],\"object\":\"chat.completion.chunk\"}\n\n" +
        "data: [DONE]\n\n";

    // Streamed /completion, n_probs 2: completion_probabilities on every token chunk, none on the final one.
    private const string CompletionStream =
        "data: {\"index\":0,\"content\":\" four\",\"tokens\":[2943],\"stop\":false,\"id_slot\":-1,\"tokens_predicted\":1,\"tokens_evaluated\":6," +
        "\"completion_probabilities\":[{\"id\":2943,\"token\":\" four\",\"bytes\":[32,102,111,117,114],\"logprob\":-0.3311412036418915,\"top_logprobs\":[" +
        "{\"id\":2943,\"token\":\" four\",\"bytes\":[32,102,111,117,114],\"logprob\":-0.3311412036418915}," +
        "{\"id\":321,\"token\":\" and\",\"bytes\":[32,97,110,100],\"logprob\":-3.3919026851654053}]}]}\n\n" +
        "data: {\"index\":0,\"content\":\",\",\"tokens\":[11],\"stop\":false,\"id_slot\":-1,\"tokens_predicted\":2,\"tokens_evaluated\":6," +
        "\"completion_probabilities\":[{\"id\":11,\"token\":\",\",\"bytes\":[44],\"logprob\":-0.21445505321025848,\"top_logprobs\":[" +
        "{\"id\":11,\"token\":\",\",\"bytes\":[44],\"logprob\":-0.21445505321025848}," +
        "{\"id\":13,\"token\":\".\",\"bytes\":[46],\"logprob\":-2.5312294960021973}]}]}\n\n" +
        "data: {\"index\":0,\"content\":\"\",\"tokens\":[],\"id_slot\":0,\"stop\":true,\"tokens_predicted\":2,\"tokens_evaluated\":6," +
        "\"prompt\":\"One, two, three,\",\"has_new_line\":false,\"truncated\":false,\"stop_type\":\"limit\",\"stopping_word\":\"\",\"tokens_cached\":7}\n\n";

    private static JsonElement Body(FixedResponseHandler handler) => JsonDocument.Parse(handler.RequestBody!).RootElement;

    private static async Task<(List<ChatStreamData> Data, FixedResponseHandler Handler)> StreamChatAsync(string sse, ChatCompletionOptions options)
    {
        var handler = new FixedResponseHandler(sse);
        var client = new LlamaServerClient("http://localhost:9999", new HttpClient(handler));
        var data = new List<ChatStreamData>();
        await foreach (var d in client.GenerateChatStreamAsync(Question, options, Ct))
            data.Add(d);
        return (data, handler);
    }

    private static async Task<(List<CompletionStreamData> Data, FixedResponseHandler Handler)> StreamCompletionAsync(CompletionOptions options)
    {
        var handler = new FixedResponseHandler(CompletionStream);
        var client = new LlamaServerClient("http://localhost:9999", new HttpClient(handler));
        var data = new List<CompletionStreamData>();
        await foreach (var d in client.GenerateStreamAsync("One, two, three,", options, Ct))
            data.Add(d);
        return (data, handler);
    }

    [Fact]
    public async Task ChatRequest_AsksForLogprobs_WithTheCandidateCount()
    {
        var handler = new FixedResponseHandler(ChatResponse);
        var client = new LlamaServerClient("http://localhost:9999", new HttpClient(handler));

        await client.GenerateChatWithToolsAsync(Question, new ChatCompletionOptions { TopLogprobs = 2 }, Ct);

        var body = Body(handler);
        body.GetProperty("logprobs").GetBoolean().Should().BeTrue("the server refuses top_logprobs without logprobs: true");
        body.GetProperty("top_logprobs").GetInt32().Should().Be(2);
        body.TryGetProperty("post_sampling_probs", out _).Should().BeFalse("the values must stay the distribution before sampling");
    }

    [Fact]
    public async Task ChatRequest_WithoutTopLogprobs_AsksForNone()
    {
        var handler = new FixedResponseHandler(ChatResponse);
        var client = new LlamaServerClient("http://localhost:9999", new HttpClient(handler));

        await client.GenerateChatWithToolsAsync(Question, new ChatCompletionOptions(), Ct);

        var body = Body(handler);
        body.TryGetProperty("logprobs", out _).Should().BeFalse();
        body.TryGetProperty("top_logprobs", out _).Should().BeFalse();
    }

    [Fact]
    public async Task TopLogprobsZero_AsksForOneCandidate_AndReturnsNoCandidates()
    {
        // top_logprobs 0 makes llama-server compute no probabilities at all; the sampled token's own value needs 1.
        var handler = new FixedResponseHandler(ChatResponse);
        var client = new LlamaServerClient("http://localhost:9999", new HttpClient(handler));

        var response = await client.GenerateChatWithToolsAsync(Question, new ChatCompletionOptions { TopLogprobs = 0 }, Ct);

        Body(handler).GetProperty("top_logprobs").GetInt32().Should().Be(1);
        var content = response.Choices![0].Logprobs!.Content!;
        content.Should().HaveCount(2);
        content[0].Logprob.Should().Be(-0.034990593791007996);
        content.Should().OnlyContain(t => t.TopLogprobs != null && t.TopLogprobs.Count == 0);
    }

    [Fact]
    public void NegativeTopLogprobs_IsRefused()
    {
        var build = () => LlamaServerClient.BuildChatRequest(Question, new ChatCompletionOptions { TopLogprobs = -1 }, stream: false);

        build.Should().Throw<ArgumentOutOfRangeException>();
    }

    [Fact]
    public void StreamedRequestWithToolsAndLogprobs_IsRefusedBeforeSending()
    {
        var options = new ChatCompletionOptions
        {
            TopLogprobs = 1,
            Tools = [new LlamaToolDefinition { Function = new FunctionDefinition { Name = "f" } }]
        };

        var streamed = () => LlamaServerClient.BuildChatRequest(Question, options, stream: true);
        var whole = () => LlamaServerClient.BuildChatRequest(Question, options, stream: false);

        streamed.Should().Throw<ArgumentException>();
        whole.Should().NotThrow("llama-server returns log probabilities with tools on a non-streamed completion");
    }

    [Fact]
    public async Task ChatResponse_LogprobsAreRead_TokenForToken()
    {
        var handler = new FixedResponseHandler(ChatResponse);
        var client = new LlamaServerClient("http://localhost:9999", new HttpClient(handler));

        var response = await client.GenerateChatWithToolsAsync(Question, new ChatCompletionOptions { TopLogprobs = 2 }, Ct);

        var content = response.Choices![0].Logprobs!.Content!;
        content.Should().HaveCount(2);

        var answer = content[0];
        answer.Id.Should().Be(33);
        answer.Token.Should().Be("B");
        answer.Bytes.Should().Equal(66);
        answer.Logprob.Should().Be(-0.034990593791007996);
        answer.TopLogprobs!.Select(t => (t.Id, t.Token, t.Logprob)).Should().Equal(
            (33, "B", -0.034990593791007996),
            (44, "M", -3.7268946170806885));
        answer.TopLogprobs.Should().OnlyContain(t => t.TopLogprobs == null);

        var end = content[1];
        end.Id.Should().Be(248046);
        end.Token.Should().BeEmpty("an end-of-generation token has no text");
        end.Bytes.Should().BeEmpty();
    }

    [Fact]
    public async Task ChatStream_LogprobsRideTheChunkOfTheirToken()
    {
        var (data, handler) = await StreamChatAsync(ChatStream, new ChatCompletionOptions { TopLogprobs = 2 });

        Body(handler).GetProperty("top_logprobs").GetInt32().Should().Be(2);
        var carrier = data.Single(d => d.Logprobs is not null);
        carrier.TextDelta.Should().Be("B");
        carrier.Logprobs!.Single().Logprob.Should().Be(-0.034990593791007996);
        carrier.Logprobs!.Single().TopLogprobs!.Select(t => t.Token).Should().Equal("B", "M");
    }

    [Fact]
    public async Task ChatStream_ReasoningTokensCarryLogprobs()
    {
        var (data, _) = await StreamChatAsync(ReasoningStream, new ChatCompletionOptions { TopLogprobs = 1 });

        var carrier = data.Single();
        carrier.TextDelta.Should().BeNull();
        carrier.ReasoningDelta.Should().Be("Thinking");
        carrier.Logprobs!.Single().Id.Should().Be(90700);
        carrier.Logprobs!.Single().Bytes.Should().Equal("Thinking"u8.ToArray().Select(b => (int)b));
    }

    [Fact]
    public async Task CompletionRequest_AsksForNProbs()
    {
        var (_, handler) = await StreamCompletionAsync(new CompletionOptions { TopLogprobs = 2 });

        handler.RequestPath.Should().Be("/completion");
        Body(handler).GetProperty("n_probs").GetInt32().Should().Be(2);
        Body(handler).TryGetProperty("post_sampling_probs", out _).Should().BeFalse();
    }

    [Fact]
    public async Task CompletionRequest_WithoutTopLogprobs_AsksForNone()
    {
        var (_, handler) = await StreamCompletionAsync(new CompletionOptions());

        Body(handler).TryGetProperty("n_probs", out _).Should().BeFalse();
    }

    [Fact]
    public async Task CompletionStream_LogprobsRideTheChunkOfTheirToken()
    {
        var (data, _) = await StreamCompletionAsync(new CompletionOptions { TopLogprobs = 2 });

        var tokens = data.Where(d => d.Logprobs is not null).SelectMany(d => d.Logprobs!).ToList();
        tokens.Select(t => (t.Id, t.Token, t.Logprob)).Should().Equal(
            (2943, " four", -0.3311412036418915),
            (11, ",", -0.21445505321025848));
        tokens[0].Bytes.Should().Equal(32, 102, 111, 117, 114);
        tokens[0].TopLogprobs!.Select(t => t.Logprob).Should().Equal(-0.3311412036418915, -3.3919026851654053);

        var final = data.Last();
        final.FinishReason.Should().Be("length");
        final.Logprobs.Should().BeNull("the final chunk carries no token");
    }

    [Fact]
    public async Task CompletionStream_TopLogprobsZero_ReturnsNoCandidates()
    {
        var (data, handler) = await StreamCompletionAsync(new CompletionOptions { TopLogprobs = 0 });

        Body(handler).GetProperty("n_probs").GetInt32().Should().Be(1);
        var tokens = data.Where(d => d.Logprobs is not null).SelectMany(d => d.Logprobs!).ToList();
        tokens.Should().HaveCount(2);
        tokens.Should().OnlyContain(t => t.TopLogprobs != null && t.TopLogprobs.Count == 0);
    }
}
