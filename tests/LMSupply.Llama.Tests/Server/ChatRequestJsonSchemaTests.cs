using System.Text.Json;
using AwesomeAssertions;
using LMSupply.Llama.Server;
using Xunit;

namespace LMSupply.Llama.Tests.Server;

/// <summary>
/// Pins the wire form of structured-output (JSON schema) on the chat request body.
/// <para>
/// Regression guard for upstream-021: <c>GenerationOptions.JsonSchema</c> is a JSON <i>string</i>.
/// Serializing that string as a root-level <c>json_schema</c> field emitted a quoted JSON string,
/// which llama-server's <c>/v1/chat/completions</c> rejects with HTTP 400. The fix routes the schema
/// through the OpenAI-compatible <c>response_format.json_schema.schema</c> as a JSON <b>object</b>.
/// </para>
/// HW-free: builds the request and inspects the serialized JSON.
/// </summary>
public class ChatRequestJsonSchemaTests
{
    // Mirrors LlamaServerClient.JsonOptions.
    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower,
        DefaultIgnoreCondition = System.Text.Json.Serialization.JsonIgnoreCondition.WhenWritingNull
    };

    // The exact array schema from the Forge ForgeLens repro (upstream-021).
    private const string ArraySchema =
        """{"type":"array","items":{"enum":["VA","NNVA","NVA"]}}""";

    private const string ObjectSchema =
        """{"type":"object","properties":{"answer":{"type":"string"}},"required":["answer"]}""";

    private static JsonDocument SerializeChat(ChatCompletionOptions options)
    {
        var request = LlamaServerClient.BuildChatRequest(
            new[] { new ChatCompletionMessage { Role = "user", Content = "hi" } },
            options,
            stream: true);
        return JsonDocument.Parse(JsonSerializer.Serialize(request, JsonOptions));
    }

    [Fact]
    public void JsonSchema_EmitsResponseFormat_AsObject_NotRootString()
    {
        using var doc = SerializeChat(new ChatCompletionOptions { JsonSchema = ObjectSchema });
        var root = doc.RootElement;

        // The broken behavior was a root-level, quoted-string "json_schema". It must be gone.
        root.TryGetProperty("json_schema", out _).Should().BeFalse(
            "the chat endpoint must not carry a root-level json_schema field");

        root.TryGetProperty("response_format", out var rf).Should().BeTrue(
            "a set JsonSchema must be sent via OpenAI-compatible response_format");
        rf.GetProperty("type").GetString().Should().Be("json_schema");

        var schema = rf.GetProperty("json_schema").GetProperty("schema");
        schema.ValueKind.Should().Be(JsonValueKind.Object,
            "the schema must be an embedded JSON object, not a quoted string");
        schema.GetProperty("type").GetString().Should().Be("object");
        schema.GetProperty("properties").GetProperty("answer").GetProperty("type").GetString().Should().Be("string");
    }

    [Fact]
    public void JsonSchema_ArrayRoot_PreservedAsObject()
    {
        // The Forge repro used an array-rooted schema; ensure a non-object root still embeds structurally.
        using var doc = SerializeChat(new ChatCompletionOptions { JsonSchema = ArraySchema });

        var schema = doc.RootElement.GetProperty("response_format").GetProperty("json_schema").GetProperty("schema");
        schema.ValueKind.Should().Be(JsonValueKind.Object);
        schema.GetProperty("type").GetString().Should().Be("array");
        schema.GetProperty("items").GetProperty("enum").EnumerateArray()
            .Select(e => e.GetString()).Should().ContainInOrder("VA", "NNVA", "NVA");
    }

    [Fact]
    public void NoJsonSchema_OmitsResponseFormat()
    {
        using var doc = SerializeChat(new ChatCompletionOptions());

        doc.RootElement.TryGetProperty("response_format", out _).Should().BeFalse(
            "requests without a schema must not carry response_format (preserves free-form generation)");
    }

    [Fact]
    public void Grammar_StillSerializesAsString()
    {
        // Grammar flows the same way but is CORRECTLY a string on the wire (GBNF text) — must not regress.
        using var doc = SerializeChat(new ChatCompletionOptions { Grammar = "root ::= (\"yes\" | \"no\")" });

        var grammar = doc.RootElement.GetProperty("grammar");
        grammar.ValueKind.Should().Be(JsonValueKind.String);
        grammar.GetString().Should().Be("root ::= (\"yes\" | \"no\")");
    }

    [Fact]
    public void GrammarAndJsonSchema_Together_Throws()
    {
        var act = () => SerializeChat(new ChatCompletionOptions { Grammar = "root ::= \"x\"", JsonSchema = ObjectSchema });

        act.Should().Throw<ArgumentException>()
            .WithMessage("*both*", "llama-server rejects a request specifying both grammar and json_schema");
    }

    [Fact]
    public void InvalidJsonSchema_Throws_WithClearMessage()
    {
        var act = () => SerializeChat(new ChatCompletionOptions { JsonSchema = "{ not valid json" });

        act.Should().Throw<ArgumentException>()
            .WithMessage("*not valid JSON*", "an unparseable schema must surface at call time, not as an opaque HTTP 400");
    }

    [Fact]
    public void ParseStructuredSchema_NativeCompletionPath_YieldsObjectNode()
    {
        // The native /completion path assigns this node to the root json_schema field.
        var node = LlamaServerClient.ParseStructuredSchema(grammar: null, jsonSchema: ObjectSchema);

        node.Should().NotBeNull();
        var json = JsonDocument.Parse(node!.ToJsonString());
        json.RootElement.ValueKind.Should().Be(JsonValueKind.Object);
        json.RootElement.GetProperty("type").GetString().Should().Be("object");
    }

    [Fact]
    public void ParseStructuredSchema_NoSchema_ReturnsNull()
    {
        LlamaServerClient.ParseStructuredSchema(grammar: null, jsonSchema: null).Should().BeNull();
        LlamaServerClient.ParseStructuredSchema(grammar: null, jsonSchema: "   ").Should().BeNull();
    }

    [Fact]
    public void CompletionRequest_NativePath_JsonSchema_SerializesAsObject()
    {
        // The native /completion endpoint (GenerateAsync path) carries a root json_schema, which
        // llama-server expects to be an object — verified live (request E) to constrain output. This
        // pins the CompletionRequest.JsonSchema (JsonNode) wire form so it can't regress to a string.
        var req = new CompletionRequest
        {
            Prompt = "classify",
            JsonSchema = LlamaServerClient.ParseStructuredSchema(grammar: null, jsonSchema: ObjectSchema)
        };

        using var doc = JsonDocument.Parse(JsonSerializer.Serialize(req, JsonOptions));

        doc.RootElement.TryGetProperty("json_schema", out var js).Should().BeTrue();
        js.ValueKind.Should().Be(JsonValueKind.Object,
            "the native /completion endpoint expects json_schema as an object, not a quoted string");
        js.GetProperty("type").GetString().Should().Be("object");
        js.GetProperty("properties").GetProperty("answer").GetProperty("type").GetString().Should().Be("string");
    }
}
