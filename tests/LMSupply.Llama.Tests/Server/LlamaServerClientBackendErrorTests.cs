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
