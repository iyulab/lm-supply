using System.Net;
using System.Text.Json.Nodes;
using AwesomeAssertions;
using LMSupply.Exceptions;
using LMSupply.Llama.Server;

namespace LMSupply.Llama.Tests.Server;

/// <summary>
/// The two calls a decision client makes beyond chat: <c>GET /v1/models</c>, whose <c>architecture.output_modalities</c>
/// says whether the model is a native decision model, and <c>POST /v1/systemone</c>, which answers typed questions.
/// </summary>
public class LlamaServerClientDecisionsTests
{
    [Fact]
    public async Task OutputModalities_OfADecisionModel_IncludeDecisions()
    {
        const string body = """
            {"object":"list","data":[{"id":"m","object":"model","owned_by":"llamacpp",
              "architecture":{"input_modalities":["text"],"output_modalities":["decisions"]},"meta":null}]}
            """;
        var handler = new RecordingHandler(HttpStatusCode.OK, body);
        var client = NewClient(handler);

        var modalities = await client.GetOutputModalitiesAsync(TestContext.Current.CancellationToken);

        modalities.Should().Equal("decisions");
        handler.Method.Should().Be(HttpMethod.Get);
        handler.Path.Should().Be("/v1/models");
    }

    [Fact]
    public async Task OutputModalities_OfATextModel_AreText()
    {
        const string body = """
            {"data":[{"id":"m","architecture":{"input_modalities":["text","image"],"output_modalities":["text"]}}]}
            """;
        var client = NewClient(new RecordingHandler(HttpStatusCode.OK, body));

        var modalities = await client.GetOutputModalitiesAsync(TestContext.Current.CancellationToken);

        modalities.Should().Equal("text");
    }

    [Fact]
    public async Task OutputModalities_WithoutArchitecture_AreNull()
    {
        // Builds before the field omit architecture; the caller keeps its legacy behavior.
        var client = NewClient(new RecordingHandler(HttpStatusCode.OK, """{"data":[{"id":"m","meta":{"n_ctx_train":4096}}]}"""));

        var modalities = await client.GetOutputModalitiesAsync(TestContext.Current.CancellationToken);

        modalities.Should().BeNull();
    }

    [Fact]
    public async Task Decide_PostsTheRequestToSystemOne_AndReturnsTheBody()
    {
        const string body = """{"answers":{"q":{"type":"noul","noul":0.75}},"usage":{"input_tokens":12,"output_tokens":0}}""";
        var handler = new RecordingHandler(HttpStatusCode.OK, body);
        var client = NewClient(handler);
        var request = new JsonObject
        {
            ["state"] = "s",
            ["questions"] = new JsonObject { ["q"] = new JsonObject { ["type"] = "noul", ["instructions"] = "i" } },
        };

        var response = await client.DecideAsync(request, TestContext.Current.CancellationToken);

        handler.Method.Should().Be(HttpMethod.Post);
        handler.Path.Should().Be("/v1/systemone");
        handler.Body.Should().Be("""{"state":"s","questions":{"q":{"type":"noul","instructions":"i"}}}""");
        response["answers"]!["q"]!["noul"]!.GetValue<double>().Should().Be(0.75);
    }

    [Fact]
    public async Task Decide_OnAModelThatIsNotADecisionModel_ThrowsWithTheServerBody()
    {
        const string error = """{"error":{"code":501,"message":"not a decision model","type":"not_supported_error"}}""";
        var client = NewClient(new RecordingHandler(HttpStatusCode.NotImplemented, error));

        var act = () => client.DecideAsync(new JsonObject(), TestContext.Current.CancellationToken);

        var thrown = await act.Should().ThrowAsync<InferenceBackendException>();
        thrown.Which.StatusCode.Should().Be(HttpStatusCode.NotImplemented);
        thrown.Which.ResponseBody.Should().Contain("not a decision model");
    }

    private static LlamaServerClient NewClient(HttpMessageHandler handler) =>
        new("http://localhost:9999", new HttpClient(handler) { BaseAddress = new Uri("http://localhost:9999") }, 4096);

    private sealed class RecordingHandler(HttpStatusCode status, string body) : HttpMessageHandler
    {
        public HttpMethod? Method { get; private set; }
        public string? Path { get; private set; }
        public string? Body { get; private set; }

        protected override async Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken)
        {
            Method = request.Method;
            Path = request.RequestUri?.AbsolutePath;
            Body = request.Content is null ? null : await request.Content.ReadAsStringAsync(cancellationToken);
            return new HttpResponseMessage(status) { Content = new StringContent(body, System.Text.Encoding.UTF8, "application/json") };
        }
    }
}
