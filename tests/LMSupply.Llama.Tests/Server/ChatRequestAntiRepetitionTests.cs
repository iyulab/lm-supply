using System.Text.Json;
using AwesomeAssertions;
using LMSupply.Llama.Server;
using Xunit;

namespace LMSupply.Llama.Tests.Server;

/// <summary>
/// Pins the serialization contract for anti-repetition sampler options on the chat request body
/// (Issue 2 + Issue 4). Standard llama-server samplers (DRY family, repeat_last_n) must serialize to
/// their exact snake_case keys when set, and must be omitted when unset so the server default applies
/// — i.e. setting nothing leaves the request byte-identical to before these options existed. Also pins
/// the pre-existing repeat_penalty gate (1.0 => omitted). HW-free: builds the request, inspects JSON.
/// </summary>
public class ChatRequestAntiRepetitionTests
{
    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower,
        DefaultIgnoreCondition = System.Text.Json.Serialization.JsonIgnoreCondition.WhenWritingNull
    };

    private static JsonDocument Serialize(ChatCompletionOptions options)
    {
        var request = LlamaServerClient.BuildChatRequest(
            new[] { new ChatCompletionMessage { Role = "user", Content = "hi" } },
            options,
            stream: true);
        return JsonDocument.Parse(JsonSerializer.Serialize(request, JsonOptions));
    }

    [Fact]
    public void DrySamplerOptions_WhenSet_SerializeToExactSnakeCaseKeys()
    {
        using var doc = Serialize(new ChatCompletionOptions
        {
            DryMultiplier = 0.8f,
            DryBase = 1.75f,
            DryAllowedLength = 2,
            DryPenaltyLastN = -1,
            RepeatLastN = 64
        });
        var root = doc.RootElement;

        root.GetProperty("dry_multiplier").GetSingle().Should().BeApproximately(0.8f, 0.0001f);
        root.GetProperty("dry_base").GetSingle().Should().BeApproximately(1.75f, 0.0001f);
        root.GetProperty("dry_allowed_length").GetInt32().Should().Be(2);
        root.GetProperty("dry_penalty_last_n").GetInt32().Should().Be(-1);
        root.GetProperty("repeat_last_n").GetInt32().Should().Be(64);
    }

    [Fact]
    public void AntiRepetitionOptions_WhenUnset_AreOmittedFromRequest()
    {
        using var doc = Serialize(new ChatCompletionOptions());
        var root = doc.RootElement;

        root.TryGetProperty("dry_multiplier", out _).Should().BeFalse("unset DRY must not alter the request");
        root.TryGetProperty("dry_base", out _).Should().BeFalse();
        root.TryGetProperty("dry_allowed_length", out _).Should().BeFalse();
        root.TryGetProperty("dry_penalty_last_n", out _).Should().BeFalse();
        root.TryGetProperty("repeat_last_n", out _).Should().BeFalse();
    }

    [Fact]
    public void NoRepeatNgramSize_IsNotSentToLlamaServer_Unsupported()
    {
        // no_repeat_ngram_size is an ONNX-only option; llama-server does not support it, so the chat
        // request must never carry it (ChatCompletionOptions has no such field by design).
        using var doc = Serialize(new ChatCompletionOptions { DryMultiplier = 0.8f });

        doc.RootElement.TryGetProperty("no_repeat_ngram_size", out _).Should().BeFalse();
    }

    [Fact]
    public void RepeatPenalty_AtOne_IsOmitted_GatePreserved()
    {
        // Pre-existing gate (Issue 4 regression pin): repeat_penalty == 1.0 means "disabled" and is
        // omitted so the server default applies; any other value is sent verbatim.
        using var doc = Serialize(new ChatCompletionOptions { RepeatPenalty = 1.0f });

        doc.RootElement.TryGetProperty("repeat_penalty", out _).Should().BeFalse();
    }

    [Fact]
    public void RepeatPenalty_AboveOne_IsSent()
    {
        using var doc = Serialize(new ChatCompletionOptions { RepeatPenalty = 1.2f });

        doc.RootElement.GetProperty("repeat_penalty").GetSingle().Should().BeApproximately(1.2f, 0.0001f);
    }
}
