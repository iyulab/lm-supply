using System.Net;
using System.Text;
using System.Text.Json.Nodes;
using AwesomeAssertions;
using LMSupply.Decider.Internal;
using LMSupply.Llama.Server;

namespace LMSupply.Decider.Tests;

/// <summary>
/// The decider against a server that answers from a script: which path a model gets from <c>/v1/models</c>, what each
/// path sends, and what a tokenizer that spells an answer label as two tokens does.
/// </summary>
public class FakeServerTests
{
    private static CancellationToken Ct => TestContext.Current.CancellationToken;

    private const string TextModel = """{"input_modalities":["text"],"output_modalities":["text"]}""";
    private const string DecisionModel = """{"input_modalities":["text"],"output_modalities":["decisions"]}""";
    private const string OneTokenResponse = """{"tokens":[7]}""";
    private const string TwoTokenResponse = """{"tokens":[7,8]}""";

    private const string NativeResponse = """
        {"answers":{
          "route":{"type":"choice","choice":"billing","probabilities":{"billing":0.7,"shipping":0.2,"none of these":0.1}},
          "angry":{"type":"noul","noul":0.25}},
         "usage":{"input_tokens":30,"output_tokens":0}}
        """;

    private sealed class ScriptedServer(string modalitiesJson, Func<string, string> tokenize, string? chatOverride = null)
        : HttpMessageHandler
    {
        public List<(string Path, string Body)> Requests { get; } = [];

        protected override async Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken)
        {
            var path = request.RequestUri!.AbsolutePath;
            var body = request.Content is null ? "" : await request.Content.ReadAsStringAsync(cancellationToken);
            Requests.Add((path, body));

            var json = path switch
            {
                "/v1/models" => "{\"data\":[{\"id\":\"m\",\"architecture\":" + modalitiesJson + "}]}",
                "/tokenize" => tokenize(JsonNode.Parse(body)!["content"]!.GetValue<string>()),
                "/v1/systemone" => NativeResponse,
                "/v1/chat/completions" => chatOverride ?? ChatResponse(body),
                _ => throw new InvalidOperationException($"Unexpected request to {path}"),
            };
            return new HttpResponseMessage(HttpStatusCode.OK) { Content = new StringContent(json, Encoding.UTF8, "application/json") };
        }
    }

    // The first token's candidates: for a choice the labels 1, 2, 0 and a stray word; for yes/no both words and a stray one.
    private static string ChatResponse(string requestBody)
    {
        var yesNo = requestBody.Contains("Answer Yes or No.", StringComparison.Ordinal);
        var candidates = yesNo
            ? "{\"token\":\"Yes\",\"logprob\":-0.2231435513},{\"token\":\"No\",\"logprob\":-1.6094379124},{\"token\":\"The\",\"logprob\":-5.0}"
            : "{\"token\":\"1\",\"logprob\":-0.3566749439},{\"token\":\"2\",\"logprob\":-1.2039728043},{\"token\":\"0\",\"logprob\":-4.6051701860},{\"token\":\"Well\",\"logprob\":-6.0}";
        var first = yesNo ? "Yes" : "1";
        return "{\"choices\":[{\"finish_reason\":\"length\",\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":\"" + first + "\"}," +
               "\"logprobs\":{\"content\":[{\"token\":\"" + first + "\",\"logprob\":-0.3,\"top_logprobs\":[" + candidates + "]}]}}]," +
               "\"timings\":{\"cache_n\":0,\"prompt_n\":40}}";
    }

    private static Task<LlamaServerDeciderModel> Create(ScriptedServer server)
    {
        var client = new LlamaServerClient("http://fake", new HttpClient(server));
        return LlamaServerDeciderModel.CreateAsync("m", _ => ValueTask.FromResult(client), null, null, Ct);
    }

    [Fact]
    public async Task AModelWithTheDecisionsModality_IsNative_AndSkipsTheTokenizerCheck()
    {
        var server = new ScriptedServer(DecisionModel, _ => OneTokenResponse);

        await using var decider = await Create(server);

        decider.Path.Should().Be(DecisionPath.Native);
        server.Requests.Select(r => r.Path).Should().Equal("/v1/models");
    }

    [Fact]
    public async Task ANativeCall_SendsOneRequestForAllQuestions_AndReadsTheServersProbabilities()
    {
        var server = new ScriptedServer(DecisionModel, _ => OneTokenResponse);
        await using var decider = await Create(server);

        var result = await decider.DecideAsync(
            "Customer: charged twice.",
            new Dictionary<string, DecisionQuestion>
            {
                ["route"] = DecisionQuestion.Choice("Which team?", ["billing", "shipping"], allowNone: true),
                ["angry"] = DecisionQuestion.YesNo("Angry?"),
            },
            Ct);

        server.Requests.Count(r => r.Path == "/v1/systemone").Should().Be(1);
        result["route"].Choice.Should().Be("billing");
        result["route"].NoneProbability.Should().BeApproximately(0.1, 1e-9);
        result["route"].Path.Should().Be(DecisionPath.Native);
        result["route"].LabelMass.Should().Be(1);
        result["angry"].Probability.Should().BeApproximately(0.25, 1e-9);
        result["angry"].IsTrue.Should().BeFalse();
    }

    [Fact]
    public async Task ATextModel_IsEmulated_AndChecksEveryLabelOnce()
    {
        var server = new ScriptedServer(TextModel, _ => OneTokenResponse);

        await using var decider = await Create(server);

        decider.Path.Should().Be(DecisionPath.Emulated);
        server.Requests.Count(r => r.Path == "/tokenize").Should().Be(10 + 26 + 2);
        decider.Capabilities.MaxChoices.Should().Be(EmulatedPrompt.MaxLetterChoices);
        decider.Capabilities.SupportsScore.Should().BeTrue();
    }

    [Fact]
    public async Task AServerThatReportsNoArchitecture_IsEmulated()
    {
        var server = new ScriptedServer("null", _ => OneTokenResponse);

        await using var decider = await Create(server);

        decider.Path.Should().Be(DecisionPath.Emulated);
    }

    [Fact]
    public async Task AnEmulatedCall_SendsOneChatRequestPerQuestion_InOrder_AndRenormalizesOverTheLabels()
    {
        var server = new ScriptedServer(TextModel, _ => OneTokenResponse);
        await using var decider = await Create(server);

        var result = await decider.DecideAsync(
            "Customer: charged twice.",
            new Dictionary<string, DecisionQuestion>
            {
                ["route"] = DecisionQuestion.Choice("Which team?", ["billing", "shipping"], allowNone: true),
                ["angry"] = DecisionQuestion.YesNo("Angry?"),
            },
            Ct);

        var chats = server.Requests.Where(r => r.Path == "/v1/chat/completions").Select(r => JsonNode.Parse(r.Body)!).ToList();
        chats.Should().HaveCount(2);
        chats[0]["messages"]![0]!["content"]!.GetValue<string>().Should().Contain("1. billing").And.StartWith("<message>\nCustomer: charged twice.\n</message>");
        chats[1]["messages"]![0]!["content"]!.GetValue<string>().Should().Contain("Answer Yes or No.").And.StartWith("<message>\nCustomer: charged twice.\n</message>");
        chats[0]["max_tokens"]!.GetValue<int>().Should().Be(1);
        chats[0]["logprobs"]!.GetValue<bool>().Should().BeTrue();
        chats[0]["top_logprobs"]!.GetValue<int>().Should().Be(3 + EmulatedPrompt.TopLogprobsMargin);
        chats[0]["chat_template_kwargs"]!["enable_thinking"]!.GetValue<bool>().Should().BeFalse();

        // 1, 2, 0 carry exp(-0.3567)=0.7, exp(-1.2040)=0.3, exp(-4.6052)=0.01; "Well" is off the labels.
        var route = result["route"];
        route.Choice.Should().Be("billing");
        route.LabelMass.Should().BeApproximately(1.01, 1e-3);
        route.Probabilities["billing"].Should().BeApproximately(0.7 / 1.01, 1e-3);
        route.NoneProbability.Should().BeApproximately(0.01 / 1.01, 1e-3);
        result["angry"].IsTrue.Should().BeTrue();
        result["angry"].Probability.Should().BeApproximately(0.8 / 1.0, 1e-3);
        ((LlamaServerDeciderModel)decider).LastTimings.Should().HaveCount(2);
    }

    [Fact]
    public async Task ATokenizerThatSplitsALabelInTwo_RefusesTheQuestion_WithTheTokenCount()
    {
        // Yes and No are two tokens; the digits stay one.
        var server = new ScriptedServer(TextModel, text => text is "Yes" or "No" ? TwoTokenResponse : OneTokenResponse);
        await using var decider = await Create(server);

        decider.Capabilities.SupportsYesNo.Should().BeFalse();
        var act = () => decider.DecideAsync("s", "angry", DecisionQuestion.YesNo("Angry?"), Ct);

        var thrown = await act.Should().ThrowAsync<NotSupportedException>();
        thrown.Which.Message.Should().Contain("'Yes'").And.Contain("2 tokens");
        server.Requests.Should().NotContain(r => r.Path == "/v1/chat/completions");
    }

    [Fact]
    public async Task ADigitQuestion_StillWorks_WhenOnlyTheYesNoLabelsAreSplit()
    {
        var server = new ScriptedServer(TextModel, text => text is "Yes" or "No" ? TwoTokenResponse : OneTokenResponse);
        await using var decider = await Create(server);

        var decision = await decider.DecideAsync("s", "route", DecisionQuestion.Choice("Which?", ["a", "b"], allowNone: true), Ct);

        decision.Choice.Should().Be("a");
    }

    [Fact]
    public async Task AFirstTokenThatIsNoLabel_ThrowsADecisionException_NamingIt()
    {
        const string chat = "{\"choices\":[{\"finish_reason\":\"length\",\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":\"\"}," +
                            "\"logprobs\":{\"content\":[{\"token\":\"<|channel>\",\"logprob\":-0.01,\"top_logprobs\":[" +
                            "{\"token\":\"<|channel>\",\"logprob\":-0.01},{\"token\":\"The\",\"logprob\":-5}]}]}}]}";
        var server = new ScriptedServer(TextModel, _ => OneTokenResponse, chat);
        await using var decider = await Create(server);

        var act = () => decider.DecideAsync("s", "q", DecisionQuestion.YesNo("Q?"), Ct);

        var thrown = await act.Should().ThrowAsync<DecisionException>();
        thrown.Which.Message.Should().Contain("<|channel>");
    }
}
