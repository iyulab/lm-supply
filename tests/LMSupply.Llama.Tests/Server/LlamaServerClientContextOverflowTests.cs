using System.Net;
using AwesomeAssertions;
using LMSupply.Exceptions;
using LMSupply.Llama.Server;

namespace LMSupply.Llama.Tests.Server;

/// <summary>
/// Verifies that ContextLengthExceededException carries the actual model context length,
/// not a hardcoded 0 (regression: LlamaServerClient.cs context overflow misreport).
/// </summary>
public class LlamaServerClientContextOverflowTests
{
    private static HttpClient MakeFakeClient(HttpStatusCode status, string body)
    {
        var handler = new FakeHandler(status, body);
        return new HttpClient(handler) { BaseAddress = new Uri("http://localhost:9999") };
    }

    [Theory]
    [InlineData("the prompt is too long")]
    [InlineData("input exceeds context limit")]
    [InlineData("context overflow detected")]
    public async Task GenerateChatWithToolsAsync_ContextOverflow_ExceptionCarriesActualMaxContextLength(string errorBody)
    {
        const int expectedContextLength = 4096;
        using var httpClient = MakeFakeClient(HttpStatusCode.BadRequest, $"{{\"error\":\"{errorBody}\"}}");
        var client = new LlamaServerClient("http://localhost:9999", httpClient, expectedContextLength);

        var act = async () => await client.GenerateChatWithToolsAsync(
            [new ChatCompletionMessage { Role = "user", Content = "hello" }]);

        var ex = await act.Should().ThrowAsync<ContextLengthExceededException>();
        ex.Which.MaxContextLength.Should().Be(expectedContextLength);
    }

    [Theory]
    [InlineData("the prompt is too long")]
    [InlineData("context overflow")]
    public async Task GenerateChatStreamAsync_ContextOverflow_ExceptionCarriesActualMaxContextLength(string errorBody)
    {
        const int expectedContextLength = 8192;
        using var httpClient = MakeFakeClient(HttpStatusCode.BadRequest, $"{{\"error\":\"{errorBody}\"}}");
        var client = new LlamaServerClient("http://localhost:9999", httpClient, expectedContextLength);

        var act = async () =>
        {
            await foreach (var _ in client.GenerateChatStreamAsync(
                [new ChatCompletionMessage { Role = "user", Content = "hello" }]))
            { }
        };

        var ex = await act.Should().ThrowAsync<ContextLengthExceededException>();
        ex.Which.MaxContextLength.Should().Be(expectedContextLength);
    }

    [Theory]
    [InlineData("the prompt is too long")]
    [InlineData("context overflow")]
    public async Task GenerateChatAsync_ContextOverflow_ExceptionCarriesActualMaxContextLength(string errorBody)
    {
        const int expectedContextLength = 32768;
        using var httpClient = MakeFakeClient(HttpStatusCode.BadRequest, $"{{\"error\":\"{errorBody}\"}}");
        var client = new LlamaServerClient("http://localhost:9999", httpClient, expectedContextLength);

        var act = async () =>
        {
            await foreach (var _ in client.GenerateChatAsync(
                [new ChatCompletionMessage { Role = "user", Content = "hello" }]))
            { }
        };

        var ex = await act.Should().ThrowAsync<ContextLengthExceededException>();
        ex.Which.MaxContextLength.Should().Be(expectedContextLength);
    }

    [Fact]
    public async Task GenerateChatWithToolsAsync_ContextOverflow_MaxContextLength_IsNeverZero()
    {
        using var httpClient = MakeFakeClient(HttpStatusCode.BadRequest, "{\"error\":\"the prompt is too long\"}");
        var client = new LlamaServerClient("http://localhost:9999", httpClient, 512);

        var act = async () => await client.GenerateChatWithToolsAsync(
            [new ChatCompletionMessage { Role = "user", Content = "hello" }]);

        var ex = await act.Should().ThrowAsync<ContextLengthExceededException>();
        ex.Which.MaxContextLength.Should().BeGreaterThan(0, because: "hardcoded 0 is a misreport");
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
