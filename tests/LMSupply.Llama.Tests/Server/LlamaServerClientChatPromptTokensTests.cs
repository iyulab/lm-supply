using System.Net;
using System.Text.Json.Nodes;
using AwesomeAssertions;
using LMSupply.Llama.Server;

namespace LMSupply.Llama.Tests.Server;

/// <summary>
/// A chat request's prompt is counted as the server renders it: the request body goes to <c>/apply-template</c> —
/// the same body <c>/v1/chat/completions</c> gets, tools included — and the rendered prompt to <c>/tokenize</c>.
/// </summary>
public class LlamaServerClientChatPromptTokensTests
{
    private static readonly ChatCompletionMessage[] Messages = [new() { Role = "user", Content = "What is in README.md?" }];

    private static ChatCompletionOptions WithTool() => new()
    {
        Tools =
        [
            new ToolDefinition
            {
                Function = new FunctionDefinition { Name = "read_file", Description = "Read a file." },
            },
        ],
        EnableThinking = false,
    };

    [Fact]
    public async Task RendersThroughApplyTemplate_WithTheChatRequestBody_ThenTokenizesThePrompt()
    {
        var handler = new RoutingHandler(applyTemplate: _ => (HttpStatusCode.OK, """{"prompt":"<rendered with tools>"}"""));
        using var http = new HttpClient(handler);
        var client = new LlamaServerClient("http://localhost:9999", http, 4096);

        var count = await client.CountChatPromptTokensAsync(Messages, WithTool(), TestContext.Current.CancellationToken);

        count.Should().Be(3, "the stub tokenizer returns three tokens");
        var sent = JsonNode.Parse(handler.ApplyTemplateBody!)!;
        sent["tools"]!.AsArray().Should().HaveCount(1, "the tools reach the template");
        sent["tools"]![0]!["function"]!["name"]!.GetValue<string>().Should().Be("read_file");
        sent["chat_template_kwargs"]!["enable_thinking"]!.GetValue<bool>().Should().BeFalse();
        handler.TokenizedContent.Should().Be("<rendered with tools>", "the rendered prompt is what gets counted");
        handler.TokenizeAddSpecial.Should().BeTrue("generation counts the prompt with the model's BOS");
    }

    [Fact]
    public async Task ServerWithoutApplyTemplate_ReturnsNull()
    {
        var handler = new RoutingHandler(applyTemplate: _ => (HttpStatusCode.NotFound, "Not Found"));
        using var http = new HttpClient(handler);
        var client = new LlamaServerClient("http://localhost:9999", http, 4096);

        var count = await client.CountChatPromptTokensAsync(Messages, WithTool(), TestContext.Current.CancellationToken);

        count.Should().BeNull();
        handler.TokenizedContent.Should().BeNull();
    }

    private sealed class RoutingHandler(Func<string, (HttpStatusCode Status, string Body)> applyTemplate) : HttpMessageHandler
    {
        public string? ApplyTemplateBody { get; private set; }
        public string? TokenizedContent { get; private set; }
        public bool? TokenizeAddSpecial { get; private set; }

        protected override async Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken)
        {
            var body = request.Content is null ? string.Empty : await request.Content.ReadAsStringAsync(cancellationToken);
            switch (request.RequestUri!.AbsolutePath)
            {
                case "/apply-template":
                    ApplyTemplateBody = body;
                    var (status, text) = applyTemplate(body);
                    return new HttpResponseMessage(status) { Content = new StringContent(text) };
                case "/tokenize":
                    var tokenize = JsonNode.Parse(body)!;
                    TokenizedContent = tokenize["content"]!.GetValue<string>();
                    TokenizeAddSpecial = tokenize["add_special"]?.GetValue<bool>();
                    return new HttpResponseMessage(HttpStatusCode.OK) { Content = new StringContent("""{"tokens":[1,2,3]}""") };
                default:
                    return new HttpResponseMessage(HttpStatusCode.NotFound);
            }
        }
    }
}
