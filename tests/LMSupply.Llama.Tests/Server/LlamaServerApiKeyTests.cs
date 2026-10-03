using System.Net;
using System.Net.Http.Headers;
using System.Text;
using AwesomeAssertions;
using LMSupply.Llama.Server;

namespace LMSupply.Llama.Tests.Server;

/// <summary>
/// The server listens on loopback, which keeps other machines out but not other processes or other users of the same
/// machine. A launched server therefore requires a per-process key, and every request LMSupply sends carries it.
/// </summary>
public class LlamaServerApiKeyTests
{
    [Fact]
    public async Task Client_WithKey_SendsBearerOnEveryRequest_IncludingTimeoutViews()
    {
        var handler = new CapturingHandler();
        using var httpClient = new HttpClient(handler);
        var client = new LlamaServerClient("http://127.0.0.1:9999", httpClient, 4096, apiKey: "k-123");

        await client.GenerateChatWithToolsAsync(
            [new ChatCompletionMessage { Role = "user", Content = "hi" }],
            cancellationToken: TestContext.Current.CancellationToken);
        await client.WithRequestTimeout(TimeSpan.FromSeconds(30)).GenerateChatWithToolsAsync(
            [new ChatCompletionMessage { Role = "user", Content = "hi" }],
            cancellationToken: TestContext.Current.CancellationToken);

        handler.Authorizations.Should().HaveCount(2).And.AllBeEquivalentTo(new AuthenticationHeaderValue("Bearer", "k-123"));
    }

    [Fact]
    public async Task Client_WithoutKey_SendsNoAuthorization()
    {
        var handler = new CapturingHandler();
        using var httpClient = new HttpClient(handler);
        var client = new LlamaServerClient("http://127.0.0.1:9999", httpClient, 4096);

        await client.GenerateChatWithToolsAsync(
            [new ChatCompletionMessage { Role = "user", Content = "hi" }],
            cancellationToken: TestContext.Current.CancellationToken);

        handler.Authorizations.Should().ContainSingle().Which.Should().BeNull();
    }

    [Theory]
    [InlineData(new[] { "--api-key", "x" }, "--api-key")]
    [InlineData(new[] { "--pooling", "mean", "--api-key-file", "keys.txt" }, "--api-key-file")]
    [InlineData(new[] { "--api-key=x" }, "--api-key")]
    [InlineData(new[] { "--pooling", "mean" }, null)]
    [InlineData(new[] { "--api-key-prefix" }, null)]
    public void FindApiKeyArgument_NamesOnlyTheKeyOptions(string[] args, string? expected)
        => LlamaServerProcess.FindApiKeyArgument(args).Should().Be(expected);

    [Fact]
    public async Task StartAsync_KeyArgumentWhileRequired_IsRefusedBeforeLaunch()
    {
        var config = new LlamaServerConfig { ModelPath = "model.gguf", AdditionalArgs = ["--api-key", "mine"] };

        var act = () => LlamaServerProcess.StartAsync(
            Path.Combine(Path.GetTempPath(), "no-such-llama-server.exe"), config, LlamaServerBackend.Cpu,
            TestContext.Current.CancellationToken);

        (await act.Should().ThrowAsync<ArgumentException>())
            .Which.Message.Should().Contain("--api-key").And.Contain("RequireApiKey");
    }

    [Fact]
    public async Task StartAsync_KeyArgumentWhenNotRequired_GoesOnToLaunch()
    {
        // Positive control for the refusal above: the same arguments pass validation once the host manages the key, and
        // the start then fails only because the binary does not exist.
        var config = new LlamaServerConfig { ModelPath = "model.gguf", AdditionalArgs = ["--api-key", "mine"], RequireApiKey = false };

        var act = () => LlamaServerProcess.StartAsync(
            Path.Combine(Path.GetTempPath(), "no-such-llama-server.exe"), config, LlamaServerBackend.Cpu,
            TestContext.Current.CancellationToken);

        var thrown = await act.Should().ThrowAsync<Exception>();
        thrown.Which.Should().NotBeOfType<ArgumentException>();
    }

    private sealed class CapturingHandler : HttpMessageHandler
    {
        public List<AuthenticationHeaderValue?> Authorizations { get; } = [];

        protected override Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken)
        {
            Authorizations.Add(request.Headers.Authorization);
            const string body = """{"choices":[{"index":0,"message":{"role":"assistant","content":"hi"},"finish_reason":"stop"}]}""";
            return Task.FromResult(new HttpResponseMessage(HttpStatusCode.OK)
            {
                Content = new StringContent(body, Encoding.UTF8, "application/json"),
            });
        }
    }
}

/// <summary>
/// Against a real llama-server: a keyed server refuses a caller without the key, and the same request succeeds with it
/// and on a server started with <see cref="LlamaServerConfig.RequireApiKey"/> off. Needs a cached GGUF model and the
/// server binary (downloaded on first use), so it is excluded from CI.
/// </summary>
[Trait("Category", "Integration")]
public sealed class LlamaServerApiKeyLiveTests
{
    private static readonly string ModelPath = Path.Combine(
        Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
        ".cache", "huggingface", "hub", "models--unsloth--Qwen3.5-2B-GGUF", "snapshots", "main", "Qwen3.5-2B-Q4_K_M.gguf");

    [Fact]
    public async Task KeyedServer_RefusesACallerWithoutTheKey_AndServesOneWithIt()
    {
        Assert.SkipUnless(File.Exists(ModelPath), $"model not cached: {ModelPath}");
        var ct = TestContext.Current.CancellationToken;
        using var downloader = new LlamaServerDownloader();
        var serverPath = await downloader.EnsureServerAsync(preferredBackend: LlamaServerBackend.Cpu, cancellationToken: ct);

        await using (var keyed = await LlamaServerProcess.StartAsync(serverPath, Config(requireKey: true), LlamaServerBackend.Cpu, ct))
        {
            keyed.ApiKey.Should().NotBeNullOrEmpty();
            keyed.Info!.ToString().Should().NotContain(keyed.ApiKey!, "the record's ToString is what ends up in logs");

            (await ChatStatusAsync(keyed.Info.BaseUrl, apiKey: null, ct)).Should().Be(HttpStatusCode.Unauthorized);
            (await ChatStatusAsync(keyed.Info.BaseUrl, "not-the-key", ct)).Should().Be(HttpStatusCode.Unauthorized);
            (await ChatStatusAsync(keyed.Info.BaseUrl, keyed.ApiKey, ct)).Should().Be(HttpStatusCode.OK);

            var client = new LlamaServerClient(keyed.Info.BaseUrl, maxContextLength: 512, apiKey: keyed.ApiKey);
            var response = await client.GenerateChatWithToolsAsync(
                [new ChatCompletionMessage { Role = "user", Content = "Say hi." }],
                new ChatCompletionOptions { MaxTokens = 2 },
                cancellationToken: ct);
            response.Choices.Should().NotBeEmpty();
        }

        await using var open = await LlamaServerProcess.StartAsync(serverPath, Config(requireKey: false), LlamaServerBackend.Cpu, ct);
        open.ApiKey.Should().BeNull();
        (await ChatStatusAsync(open.Info!.BaseUrl, apiKey: null, ct)).Should().Be(HttpStatusCode.OK);
    }

    private static LlamaServerConfig Config(bool requireKey) =>
        new() { ModelPath = ModelPath, ContextSize = 512, GpuLayers = 0, RequireApiKey = requireKey };

    private static async Task<HttpStatusCode> ChatStatusAsync(string baseUrl, string? apiKey, CancellationToken ct)
    {
        using var http = new HttpClient();
        using var request = new HttpRequestMessage(HttpMethod.Post, $"{baseUrl}/v1/chat/completions")
        {
            Content = new StringContent("""{"messages":[{"role":"user","content":"hi"}],"max_tokens":1}""", Encoding.UTF8, "application/json"),
        };
        if (apiKey is not null)
            request.Headers.Authorization = new AuthenticationHeaderValue("Bearer", apiKey);
        using var response = await http.SendAsync(request, ct);
        return response.StatusCode;
    }
}
