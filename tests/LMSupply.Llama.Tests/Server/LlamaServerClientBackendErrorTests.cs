using System.Net;
using AwesomeAssertions;
using LMSupply.Exceptions;
using LMSupply.Llama.Server;

namespace LMSupply.Llama.Tests.Server;

/// <summary>
/// Verifies that a non-success response that isn't a recognized context-overflow shape surfaces
/// the backend's actual status code and body (regression: LlamaServerClient discarding the
/// already-read response body and falling through to a generic EnsureSuccessStatusCode()).
/// </summary>
public class LlamaServerClientBackendErrorTests
{
    private static HttpClient MakeFakeClient(HttpStatusCode status, string body)
    {
        var handler = new FakeHandler(status, body);
        return new HttpClient(handler) { BaseAddress = new Uri("http://localhost:9999") };
    }

    [Fact]
    public async Task GenerateChatWithToolsAsync_NonContextServerError_ThrowsWithStatusCodeAndBody()
    {
        const string body = "{\"error\":{\"message\":\"unexpected chat template state\"}}";
        using var httpClient = MakeFakeClient(HttpStatusCode.InternalServerError, body);
        var client = new LlamaServerClient("http://localhost:9999", httpClient, 4096);

        var act = async () => await client.GenerateChatWithToolsAsync(
            [new ChatCompletionMessage { Role = "user", Content = "hello" }]);

        var ex = await act.Should().ThrowAsync<InferenceBackendException>();
        ex.Which.StatusCode.Should().Be(HttpStatusCode.InternalServerError);
        ex.Which.ResponseBody.Should().Be(body);
        ex.Which.Message.Should().Contain(body, because: "the caller should see what the backend itself reported");
    }

    [Fact]
    public async Task GenerateChatStreamAsync_NonContextServerError_ThrowsWithStatusCodeAndBody()
    {
        const string body = "{\"error\":{\"message\":\"internal server error\"}}";
        using var httpClient = MakeFakeClient(HttpStatusCode.InternalServerError, body);
        var client = new LlamaServerClient("http://localhost:9999", httpClient, 4096);

        var act = async () =>
        {
            await foreach (var _ in client.GenerateChatStreamAsync(
                [new ChatCompletionMessage { Role = "user", Content = "hello" }]))
            { }
        };

        var ex = await act.Should().ThrowAsync<InferenceBackendException>();
        ex.Which.StatusCode.Should().Be(HttpStatusCode.InternalServerError);
        ex.Which.ResponseBody.Should().Be(body);
    }

    [Fact]
    public async Task GenerateChatWithToolsAsync_BadRequestWithoutContextOverflowShape_ThrowsWithBody()
    {
        // BadRequest alone isn't sufficient to classify as context overflow — the body must
        // actually match the heuristic. A BadRequest for an unrelated reason must still surface.
        const string body = "{\"error\":\"invalid role: system2\"}";
        using var httpClient = MakeFakeClient(HttpStatusCode.BadRequest, body);
        var client = new LlamaServerClient("http://localhost:9999", httpClient, 4096);

        var act = async () => await client.GenerateChatWithToolsAsync(
            [new ChatCompletionMessage { Role = "user", Content = "hello" }]);

        var ex = await act.Should().ThrowAsync<InferenceBackendException>();
        ex.Which.StatusCode.Should().Be(HttpStatusCode.BadRequest);
        ex.Which.ResponseBody.Should().Be(body);
    }

    [Fact]
    public async Task GenerateEmbeddingsBatchAsync_InputLargerThanTheBatch_NamesTheLimit()
    {
        const string body = """{"error":{"code":500,"message":"input (12639 tokens) is too large to process. increase the physical batch size (current batch size: 8192)","type":"server_error"}}""";
        using var httpClient = MakeFakeClient(HttpStatusCode.InternalServerError, body);
        var client = new LlamaServerClient("http://localhost:9999", httpClient, 8192);

        var act = async () => await client.GenerateEmbeddingsBatchAsync(["a very long text"]);

        var ex = await act.Should().ThrowAsync<ContextLengthExceededException>();
        ex.Which.TokenCount.Should().Be(12639);
        ex.Which.MaxContextLength.Should().Be(8192);
    }

    [Fact]
    public async Task GenerateEmbeddingsBatchAsync_OtherServerError_ThrowsWithStatusCodeAndBody()
    {
        const string body = "{\"error\":{\"message\":\"model is not loaded\"}}";
        using var httpClient = MakeFakeClient(HttpStatusCode.ServiceUnavailable, body);
        var client = new LlamaServerClient("http://localhost:9999", httpClient, 8192);

        var act = async () => await client.GenerateEmbeddingsBatchAsync(["text"]);

        var ex = await act.Should().ThrowAsync<InferenceBackendException>();
        ex.Which.StatusCode.Should().Be(HttpStatusCode.ServiceUnavailable);
        ex.Which.Message.Should().Contain(body);
    }

    [Fact]
    public async Task RerankAsync_ServerError_ThrowsWithStatusCodeAndBody()
    {
        const string body = "{\"error\":{\"message\":\"pooling type is not rank\"}}";
        using var httpClient = MakeFakeClient(HttpStatusCode.InternalServerError, body);
        var client = new LlamaServerClient("http://localhost:9999", httpClient, 8192);

        var act = async () => await client.RerankAsync("query", ["doc"]);

        var ex = await act.Should().ThrowAsync<InferenceBackendException>();
        ex.Which.ResponseBody.Should().Be(body);
    }

    [Fact]
    public async Task CountTokensAsync_ServerError_ThrowsWithStatusCodeAndBody()
    {
        const string body = "{\"error\":{\"message\":\"tokenizer unavailable\"}}";
        using var httpClient = MakeFakeClient(HttpStatusCode.InternalServerError, body);
        var client = new LlamaServerClient("http://localhost:9999", httpClient, 8192);

        var act = async () => await client.CountTokensAsync("text");

        var ex = await act.Should().ThrowAsync<InferenceBackendException>();
        ex.Which.ResponseBody.Should().Be(body);
    }

    private sealed class FakeHandler(HttpStatusCode status, string body) : HttpMessageHandler
    {
        protected override Task<HttpResponseMessage> SendAsync(
            HttpRequestMessage request,
            CancellationToken cancellationToken)
        {
            var response = new HttpResponseMessage(status)
            {
                Content = new StringContent(body)
            };
            return Task.FromResult(response);
        }
    }
}
