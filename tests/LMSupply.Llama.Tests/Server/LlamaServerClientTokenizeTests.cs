using System.Net;
using System.Text;
using System.Text.Json;
using AwesomeAssertions;
using LMSupply.Llama.Server;

namespace LMSupply.Llama.Tests.Server;

/// <summary>
/// <see cref="LlamaServerClient.TokenizeAsync"/> returns the ids <c>/tokenize</c> gives a text, so a caller can find the id
/// of a token whose log probability it reads. The response is llama-server b11429's for <c>"A &lt;think&gt;"</c> on a
/// Qwen3.5 model: the special-token text became the special token.
/// </summary>
public class LlamaServerClientTokenizeTests
{
    private static CancellationToken Ct => TestContext.Current.CancellationToken;

    private sealed class TokenizeHandler(string body) : HttpMessageHandler
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

    [Fact]
    public async Task ReturnsTheServersIds_WithoutTheModelsPromptTokens()
    {
        var handler = new TokenizeHandler("{\"tokens\":[32,220,248068]}");
        var client = new LlamaServerClient("http://localhost:9999", new HttpClient(handler));

        var ids = await client.TokenizeAsync("A <think>", cancellationToken: Ct);

        ids.Should().Equal(32, 220, 248068);
        handler.RequestPath.Should().Be("/tokenize");
        var body = JsonDocument.Parse(handler.RequestBody!).RootElement;
        body.GetProperty("content").GetString().Should().Be("A <think>");
        body.TryGetProperty("add_special", out _).Should().BeFalse("a label's id must not come with a BOS token");
    }

    [Fact]
    public async Task AddSpecial_AsksForTheModelsPromptTokens()
    {
        var handler = new TokenizeHandler("{\"tokens\":[1,32]}");
        var client = new LlamaServerClient("http://localhost:9999", new HttpClient(handler));

        await client.TokenizeAsync("A", addSpecial: true, Ct);

        JsonDocument.Parse(handler.RequestBody!).RootElement.GetProperty("add_special").GetBoolean().Should().BeTrue();
    }

    [Fact]
    public async Task AResponseWithoutTokens_IsEmpty()
    {
        var client = new LlamaServerClient("http://localhost:9999", new HttpClient(new TokenizeHandler("{}")));

        (await client.TokenizeAsync("", cancellationToken: Ct)).Should().BeEmpty();
    }
}
