using System.Text;
using System.Text.Json;
using AwesomeAssertions;
using LMSupply.Generator.Abstractions;
using LMSupply.Generator.Internal;
using LMSupply.Generator.Models;
using Xunit;

namespace LMSupply.Generator.Tests.Gguf;

/// <summary>
/// Decisive experiment for upstream triage (umbrella 2026-06-16):
/// Filer ISSUE-219 — local GGUF chat final answer arrives as a single text delta.
/// Source tracing proved the IronHive adapter + Filer's ResilientChatClient are 1:1
/// per-chunk pass-throughs, so the collapse must originate inside
/// LMSupply.Generator.GenerateChatStreamAsync (or the llama-server beneath it).
///
/// Leading hypothesis: when GenerationOptions.Tools is populated, the llama-server
/// buffers content and emits it as one chunk, whereas the tools-absent path streams
/// per token. This test measures text-chunk counts for the SAME prompt with tools
/// ON vs OFF on the exact repro model (gguf:qwen3-fast = Qwen3.5-2B-Q4_K_M).
///
/// Run with: dotnet test --filter "FullyQualifiedName~StreamingToolBuffering"
/// </summary>
[Trait("Category", "Integration")]
public class StreamingToolBufferingExperimentTests
{
    private readonly ITestOutputHelper _output;

    public StreamingToolBufferingExperimentTests(ITestOutputHelper output)
    {
        _output = output;
    }

    [Fact]
    public async Task GenerateChatStreamAsync_ToolsOn_vs_ToolsOff_TextChunkCount()
    {
        await using var model = await LocalGenerator.LoadAsync("gguf:qwen3-fast", new GeneratorOptions { MaxContextLength = 4096 }, cancellationToken: TestContext.Current.CancellationToken);

        // A prompt that elicits a multi-sentence plain-text answer (no tool actually needed),
        // mirroring Filer's "final answer" turn (finishReason=stop, multi-sentence).
        var messages = new[]
        {
            ChatMessage.System("You are a helpful assistant. Answer in 3-4 full sentences."),
            ChatMessage.User("Briefly explain what a vector database is and why RAG systems use one.")
        };

        // tools-off
        var off = await CountTextChunksAsync(model, messages, tools: null);
        _output.WriteLine($"[tools-OFF] textChunks={off.TextChunks} totalChunks={off.TotalChunks} " +
                          $"chars={off.TotalChars} finish={off.FinishReason}");

        // tools-on: a dummy tool the model has no reason to call on this prompt.
        var weatherSchema = JsonSerializer.Deserialize<JsonElement>(
            """{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}""");
        var tools = new List<ChatToolDefinition>
        {
            new("get_weather", "Get the current weather for a city.", weatherSchema)
        };
        var on = await CountTextChunksAsync(model, messages, tools);
        _output.WriteLine($"[tools-ON ] textChunks={on.TextChunks} totalChunks={on.TotalChunks} " +
                          $"chars={on.TotalChars} finish={on.FinishReason} toolChunks={on.ToolChunks}");

        _output.WriteLine("");
        _output.WriteLine("=== VERDICT ===");
        if (off.TextChunks > 1 && on.TextChunks <= 1)
        {
            _output.WriteLine("CONFIRMED tool-presence buffering: tools-off streams, tools-on collapses to 1 chunk.");
        }
        else if (off.TextChunks <= 1 && on.TextChunks <= 1)
        {
            _output.WriteLine("NOT tool-specific: both collapse to 1 chunk -> server/config emits non-incremental content regardless of tools.");
        }
        else if (off.TextChunks > 1 && on.TextChunks > 1)
        {
            _output.WriteLine("NO COLLAPSE in LMSupply: both stream multiple chunks -> single-delta originates ABOVE this boundary; re-examine consumer chain.");
        }
        else
        {
            _output.WriteLine($"UNEXPECTED: off={off.TextChunks} on={on.TextChunks}");
        }

        // Don't hard-assert a mechanism we haven't confirmed; the experiment's value is the
        // logged counts. Only assert the run actually produced text (guards a dead run).
        Assert.True(off.TotalChars > 0 || on.TotalChars > 0,
            "Model produced no text in either configuration; experiment inconclusive (check model/server provisioning).");
    }

    /// <summary>
    /// Closer repro of Filer ISSUE-219 call-3: the final-answer turn arrives after a
    /// multi-turn history of tool-call + tool-result messages (ReadFile x2, WriteFile),
    /// with tools still present in options. This is the one structural difference between
    /// the clean 2-message test above and Filer's real run. If the symptom (finish=stop +
    /// a single text chunk on a multi-sentence answer) reproduces here, the trigger is the
    /// tool-result history, not the tool definitions.
    /// </summary>
    [Fact]
    public async Task GenerateChatStreamAsync_MultiTurnToolHistory_FinalAnswerChunkCount()
    {
        await using var model = await LocalGenerator.LoadAsync("gguf:qwen3-fast", new GeneratorOptions { MaxContextLength = 4096 }, cancellationToken: TestContext.Current.CancellationToken);

        var weatherSchema = JsonSerializer.Deserialize<JsonElement>(
            """{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}""");
        var tools = new List<ChatToolDefinition>
        {
            new("read_file", "Read a file's contents.", weatherSchema),
            new("write_file", "Write content to a file.", weatherSchema)
        };

        // Mirror Filer's call-3 history: 2x ReadFile round-trips + 1x WriteFile, then ask
        // for the final combined summary (the turn that produced finish=stop + 1 text delta).
        var messages = new List<ChatMessage>
        {
            ChatMessage.System("You are a file assistant. Use tools, then summarize what you did."),
            ChatMessage.User("Read a.txt and b.txt, then write a combined summary to c.txt."),
            ChatMessage.AssistantToolCalls(new[] { new ChatToolCall("call_1", "read_file", "{\"path\":\"a.txt\"}") }),
            ChatMessage.ToolResult("call_1", "a.txt contents: The quarterly revenue grew by 12 percent."),
            ChatMessage.AssistantToolCalls(new[] { new ChatToolCall("call_2", "read_file", "{\"path\":\"b.txt\"}") }),
            ChatMessage.ToolResult("call_2", "b.txt contents: Customer churn dropped to 3 percent this quarter."),
            ChatMessage.AssistantToolCalls(new[] { new ChatToolCall("call_3", "write_file", "{\"path\":\"c.txt\"}") }),
            ChatMessage.ToolResult("call_3", "write_file: ok, 2 sentences written to c.txt."),
            ChatMessage.User("Now tell me in 3-4 sentences what you found and did.")
        };

        var stats = await CountTextChunksAsync(model, messages, tools);
        _output.WriteLine($"[multi-turn final] textChunks={stats.TextChunks} totalChunks={stats.TotalChunks} " +
                          $"chars={stats.TotalChars} finish={stats.FinishReason} toolChunks={stats.ToolChunks}");
        _output.WriteLine($"first 200 chars: {stats.Preview}");
        _output.WriteLine("");
        _output.WriteLine("=== VERDICT ===");
        if (stats.FinishReason == "stop" && stats.TextChunks <= 1 && stats.TotalChars > 40)
        {
            _output.WriteLine("REPRODUCED Filer 219: stop + single text chunk on a multi-sentence answer. " +
                              "Trigger = tool-result history. Locus confirmed in LMSupply/server boundary.");
        }
        else if (stats.FinishReason == "stop" && stats.TextChunks > 1)
        {
            _output.WriteLine($"NOT reproduced: streamed {stats.TextChunks} chunks. Trigger is NOT tool-result history; " +
                              "request Filer's exact call-3 messages+ChatOptions.");
        }
        else
        {
            _output.WriteLine($"INCONCLUSIVE: finish={stats.FinishReason} textChunks={stats.TextChunks} chars={stats.TotalChars}");
        }

        Assert.True(stats.TotalChunks > 0, "No chunks produced; experiment inconclusive.");
    }

    /// <summary>
    /// Mechanism probe (umbrella 2026-06-21): does GenerationOptions.EnableThinking=false actually
    /// suppress thinking for a thinking-default-on model (Qwen3) on the CHAT path
    /// (GenerateChatStreamAsync -> /v1/chat/completions), and does a "/no_think" soft switch in the
    /// last user message suppress it? Measures Text vs ReasoningDelta separately.
    /// Run: dotnet test --filter "FullyQualifiedName~ChatThinking_Probe"
    /// </summary>
    [Fact]
    public async Task ChatThinking_Probe_Qwen3_EnableThinkingToggle_and_NoThinkInjection()
    {
        await using var model = await LocalGenerator.LoadAsync("gguf:qwen3-fast", new GeneratorOptions { MaxContextLength = 4096 }, cancellationToken: TestContext.Current.CancellationToken);

        string sys = "You are a helpful assistant.";
        string ask = "What is 2+2? Answer in one short sentence.";

        var baseMsgs = new[] { ChatMessage.System(sys), ChatMessage.User(ask) };

        var off = await ProbeThinkingAsync(model, baseMsgs, ThinkingMode.Off);
        _output.WriteLine($"[Thinking.Off ] reasoningChars={off.ReasoningChars} textChars={off.TextChars} finish={off.FinishReason}");
        _output.WriteLine($"    textPreview: {off.TextPreview}");

        var auto = await ProbeThinkingAsync(model, baseMsgs, ThinkingMode.Auto);
        _output.WriteLine($"[Thinking.Auto] reasoningChars={auto.ReasoningChars} textChars={auto.TextChars} finish={auto.FinishReason}");

        _output.WriteLine("");
        _output.WriteLine("=== VERDICT ===");
        _output.WriteLine($"Thinking.Off suppresses Qwen3 reasoning? {(off.ReasoningChars == 0 ? "YES" : "NO (reasoning still emitted)")}");
        _output.WriteLine($"Thinking.Auto keeps model default (Qwen3 thinks)? {(auto.ReasoningChars > 0 ? "YES" : "NO")}");

        // The fix's contract: Off must suppress thinking while still producing a direct answer.
        off.ReasoningChars.Should().Be(0, "Thinking.Off must suppress the reasoning block on Qwen3");
        off.TextChars.Should().BeGreaterThan(0, "Thinking.Off must still yield a direct answer");
        auto.ReasoningChars.Should().BeGreaterThan(0, "Thinking.Auto must preserve Qwen3's default (thinking on)");
    }

    private static async Task<ThinkStats> ProbeThinkingAsync(
        IGeneratorModel model, IEnumerable<ChatMessage> messages, ThinkingMode thinking)
    {
        var opts = new GenerationOptions
        {
            MaxTokens = 1024,            // large enough for Qwen3 to close </think> and emit an answer
            Temperature = 0.2f,
            Thinking = thinking,
            ExtractReasoningTokens = true // surface reasoning separately so we can measure think vs answer
        };
        var text = new StringBuilder();
        var reasoning = new StringBuilder();
        string? finish = null;
        await foreach (var chunk in model.GenerateChatStreamAsync(messages, opts))
        {
            if (chunk.Text is not null) text.Append(chunk.Text);
            if (chunk.ReasoningDelta is not null) reasoning.Append(chunk.ReasoningDelta);
            if (chunk.FinishReason is not null) finish = chunk.FinishReason;
        }
        return new ThinkStats
        {
            TextChars = text.Length,
            ReasoningChars = reasoning.Length,
            FinishReason = finish,
            TextPreview = text.Length > 0 ? text.ToString(0, Math.Min(120, text.Length)) : "(no text)"
        };
    }

    private sealed class ThinkStats
    {
        public int TextChars { get; set; }
        public int ReasoningChars { get; set; }
        public string? FinishReason { get; set; }
        public string TextPreview { get; set; } = "";
    }

    private static async Task<StreamStats> CountTextChunksAsync(
        IGeneratorModel model,
        IEnumerable<ChatMessage> messages,
        IReadOnlyList<ChatToolDefinition>? tools)
    {
        var opts = new GenerationOptions
        {
            MaxTokens = 256,
            Temperature = 0.3f,
            Tools = tools
        };

        var stats = new StreamStats();
        var sb = new StringBuilder();

        await foreach (var chunk in model.GenerateChatStreamAsync(messages, opts))
        {
            stats.TotalChunks++;
            if (chunk.Text is not null)
            {
                stats.TextChunks++;
                sb.Append(chunk.Text);
            }
            if (chunk.ToolCalls is { Count: > 0 })
            {
                stats.ToolChunks++;
            }
            if (chunk.FinishReason is not null)
            {
                stats.FinishReason = chunk.FinishReason;
            }
        }

        stats.TotalChars = sb.Length;
        stats.Preview = sb.Length > 0 ? sb.ToString(0, Math.Min(200, sb.Length)) : "(no text)";
        return stats;
    }

    private sealed class StreamStats
    {
        public int TotalChunks { get; set; }
        public int TextChunks { get; set; }
        public int ToolChunks { get; set; }
        public int TotalChars { get; set; }
        public string? FinishReason { get; set; }
        public string Preview { get; set; } = "";
    }
}
