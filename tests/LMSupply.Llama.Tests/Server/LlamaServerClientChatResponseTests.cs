using System.Net;
using AwesomeAssertions;
using LMSupply.Llama.Server;

namespace LMSupply.Llama.Tests.Server;

/// <summary>
/// The non-streaming chat response carries two things a consumer cannot get any other way: the
/// reasoning the model produced before its answer (reasoning_content, separated from content by
/// llama-server b8994+) and the server's token accounting (the OpenAI-compatible usage object). Before
/// 0.66.0 the client deserialized reasoning_content but no caller read it, and usage was not
/// deserialized at all, so an empty answer with finish_reason=length was unexplainable and a bridge
/// could not report token counts. These facts pin both fields on a recorded body.
/// </summary>
public class LlamaServerClientChatResponseTests
{
    private const string Body = """
        {
          "id": "chatcmpl-1",
          "object": "chat.completion",
          "model": "gemma-4-E4B-it",
          "choices": [
            {
              "index": 0,
              "message": {
                "role": "assistant",
                "content": "",
                "reasoning_content": "The user wants an essay. First outline the sections..."
              },
              "finish_reason": "length"
            }
          ],
          "usage": { "prompt_tokens": 42, "completion_tokens": 64, "total_tokens": 106 }
        }
        """;

    [Fact]
    public async Task GenerateChatWithToolsAsync_CarriesReasoningContentAndUsage()
    {
        using var httpClient = new HttpClient(new FakeHandler(HttpStatusCode.OK, Body))
        {
            BaseAddress = new Uri("http://localhost:9999")
        };
        var client = new LlamaServerClient("http://localhost:9999", httpClient, 4096);

        var response = await client.GenerateChatWithToolsAsync(
            [new ChatCompletionMessage { Role = "user", Content = "Write a 400-word essay." }],
            cancellationToken: TestContext.Current.CancellationToken);

        var choice = response.Choices.Should().ContainSingle().Subject;
        choice.FinishReason.Should().Be("length");
        choice.Message!.Content.Should().BeEmpty("the budget went to reasoning, which is exactly the case the caller must be able to see");
        choice.Message.ReasoningContent.Should().StartWith("The user wants an essay.");
        response.Usage.Should().NotBeNull();
        response.Usage!.PromptTokens.Should().Be(42);
        response.Usage.CompletionTokens.Should().Be(64);
        response.Usage.TotalTokens.Should().Be(106);
    }

    [Fact]
    public async Task GenerateChatWithToolsAsync_UsageAbsent_IsNull()
    {
        const string noUsage = """{"choices":[{"index":0,"message":{"role":"assistant","content":"hi"},"finish_reason":"stop"}]}""";
        using var httpClient = new HttpClient(new FakeHandler(HttpStatusCode.OK, noUsage))
        {
            BaseAddress = new Uri("http://localhost:9999")
        };
        var client = new LlamaServerClient("http://localhost:9999", httpClient, 4096);

        var response = await client.GenerateChatWithToolsAsync(
            [new ChatCompletionMessage { Role = "user", Content = "hi" }],
            cancellationToken: TestContext.Current.CancellationToken);

        response.Usage.Should().BeNull();
        response.Choices![0].Message!.ReasoningContent.Should().BeNull();
    }

    private sealed class FakeHandler(HttpStatusCode status, string body) : HttpMessageHandler
    {
        protected override Task<HttpResponseMessage> SendAsync(
            HttpRequestMessage request,
            CancellationToken cancellationToken)
        {
            var response = new HttpResponseMessage(status)
            {
                Content = new StringContent(body, System.Text.Encoding.UTF8, "application/json")
            };
            return Task.FromResult(response);
        }
    }
}
