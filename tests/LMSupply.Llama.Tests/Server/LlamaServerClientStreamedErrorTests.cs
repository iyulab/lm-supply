using System.Net;
using System.Text;
using AwesomeAssertions;
using LMSupply.Exceptions;
using LMSupply.Llama.Server;

namespace LMSupply.Llama.Tests.Server;

/// <summary>
/// llama-server reports a failure that happens after the response headers went out (<c>Compute error.</c> when the machine
/// runs short of memory) as an ordinary <c>data:</c> event <c>{"error":{...}}</c> on a <c>200 text/event-stream</c>, not as
/// an HTTP status. A reader that looks only for <c>choices</c> ended the stream as if the model had said nothing: the caller
/// could not tell «the model said nothing» from «the server failed». The failure is the exception the same failure gets
/// without streaming.
/// </summary>
public class LlamaServerClientStreamedErrorTests
{
    private const string ComputeError = """{"error":{"code":500,"message":"Compute error.","type":"server_error"}}""";

    private static readonly ChatCompletionMessage[] Messages = [new() { Role = "user", Content = "hello" }];

    private static LlamaServerClient ClientAnswering(string sseBody)
    {
        var httpClient = new HttpClient(new SseHandler(sseBody)) { BaseAddress = new Uri("http://localhost:9999") };
        return new LlamaServerClient("http://localhost:9999", httpClient, 4096);
    }

    private static string Sse(params string[] events) => string.Concat(events.Select(e => $"data: {e}\n\n"));

    [Fact]
    public async Task A_structured_chat_stream_that_carries_only_an_error_event_throws_with_the_servers_failure()
    {
        var client = ClientAnswering(Sse(ComputeError));

        var act = async () =>
        {
            await foreach (var _ in client.GenerateChatStreamAsync(Messages, cancellationToken: TestContext.Current.CancellationToken))
            { }
        };

        var ex = await act.Should().ThrowAsync<InferenceBackendException>();
        ex.Which.StatusCode.Should().Be(HttpStatusCode.InternalServerError);
        ex.Which.ResponseBody.Should().Be(ComputeError);
        ex.Which.Message.Should().Contain("Compute error.");
    }

    [Fact]
    public async Task A_failure_after_some_text_is_thrown_after_that_text_was_delivered()
    {
        var client = ClientAnswering(Sse(
            """{"choices":[{"index":0,"delta":{"content":"Hel"}}]}""",
            """{"choices":[{"index":0,"delta":{"content":"lo"}}]}""",
            ComputeError));
        var text = new StringBuilder();

        var act = async () =>
        {
            await foreach (var data in client.GenerateChatStreamAsync(Messages, cancellationToken: TestContext.Current.CancellationToken))
                text.Append(data.TextDelta);
        };

        await act.Should().ThrowAsync<InferenceBackendException>();
        text.ToString().Should().Be("Hello", because: "what the model already said is not taken back");
    }

    [Fact]
    public async Task A_text_chat_stream_throws_for_an_error_event_too()
    {
        var client = ClientAnswering(Sse(ComputeError));

        var act = async () =>
        {
            await foreach (var _ in client.GenerateChatAsync(Messages, cancellationToken: TestContext.Current.CancellationToken))
            { }
        };

        var ex = await act.Should().ThrowAsync<InferenceBackendException>();
        ex.Which.ResponseBody.Should().Be(ComputeError);
    }

    [Fact]
    public async Task A_raw_completion_stream_throws_for_an_error_event_too()
    {
        var client = ClientAnswering(Sse(ComputeError));

        var act = async () =>
        {
            await foreach (var _ in client.GenerateStreamAsync("Once upon a time", cancellationToken: TestContext.Current.CancellationToken))
            { }
        };

        var ex = await act.Should().ThrowAsync<InferenceBackendException>();
        ex.Which.StatusCode.Should().Be(HttpStatusCode.InternalServerError);
    }

    [Fact]
    public async Task An_error_event_without_a_status_code_is_a_500_with_the_event_as_its_body()
    {
        // The Anthropic-style form of the same server: «event: error» then a data line with a typed error object.
        const string body = """{"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}""";
        var client = ClientAnswering($"event: error\ndata: {body}\n\n");

        var act = async () =>
        {
            await foreach (var _ in client.GenerateChatStreamAsync(Messages, cancellationToken: TestContext.Current.CancellationToken))
            { }
        };

        var ex = await act.Should().ThrowAsync<InferenceBackendException>();
        ex.Which.StatusCode.Should().Be(HttpStatusCode.InternalServerError);
        ex.Which.ResponseBody.Should().Be(body);
    }

    [Fact]
    public async Task A_streamed_context_overflow_is_the_context_length_exception()
    {
        const string body = """{"error":{"code":400,"message":"the request exceeds the available context size","type":"exceed_context_size_error","n_prompt_tokens":5000,"n_ctx":4096}}""";
        var client = ClientAnswering(Sse(body));

        var act = async () =>
        {
            await foreach (var _ in client.GenerateChatStreamAsync(Messages, cancellationToken: TestContext.Current.CancellationToken))
            { }
        };

        var ex = await act.Should().ThrowAsync<ContextLengthExceededException>();
        ex.Which.TokenCount.Should().Be(5000);
        ex.Which.MaxContextLength.Should().Be(4096);
    }

    [Fact]
    public async Task A_stream_without_an_error_event_is_read_as_before()
    {
        var client = ClientAnswering(Sse(
            """{"choices":[{"index":0,"delta":{"content":"Hi"}}]}""",
            """{"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}""") + "data: [DONE]\n\n");
        var text = new StringBuilder();
        string? finish = null;

        await foreach (var data in client.GenerateChatStreamAsync(Messages, cancellationToken: TestContext.Current.CancellationToken))
        {
            text.Append(data.TextDelta);
            finish ??= data.FinishReason;
        }

        text.ToString().Should().Be("Hi");
        finish.Should().Be("stop");
    }

    private sealed class SseHandler(string body) : HttpMessageHandler
    {
        protected override Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken) =>
            Task.FromResult(new HttpResponseMessage(HttpStatusCode.OK)
            {
                Content = new StringContent(body, Encoding.UTF8, "text/event-stream"),
            });
    }
}
