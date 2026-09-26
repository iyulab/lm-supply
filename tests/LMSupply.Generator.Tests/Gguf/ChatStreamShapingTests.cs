using AwesomeAssertions;
using LMSupply.Generator.Abstractions;
using LMSupply.Generator.Internal.Llama;
using LMSupply.Generator.Models;
using LMSupply.Llama.Server;
using Xunit;

namespace LMSupply.Generator.Tests.Gguf;

/// <summary>
/// The streamed chat turn ends with one chunk that carries the finish reason and the server's usage. llama-server sends
/// usage on its own chunk after <c>finish_reason</c> (<c>stream_options.include_usage</c>), and the parser/filter flushes
/// can still release text after the server's last delta — a consumer that stops at <c>FinishReason</c> must lose neither.
/// </summary>
public class ChatStreamShapingTests
{
    private static CancellationToken Ct => TestContext.Current.CancellationToken;

    private static async IAsyncEnumerable<ChatStreamData> Stream(params ChatStreamData[] data)
    {
        foreach (var d in data)
        {
            await Task.Yield();
            yield return d;
        }
    }

    private static readonly ChatCompletionUsage ServerUsage = new() { PromptTokens = 41, CompletionTokens = 512, TotalTokens = 553 };

    private static readonly LlamaServerTimings ServerTimings = new()
    {
        CacheN = 9, PromptN = 32, PromptMs = 21.5, PromptPerSecond = 185.7, PredictedN = 512, PredictedMs = 4300.0, PredictedPerSecond = 119.0,
    };

    private static async Task<List<ChatStreamChunk>> ShapeAsync(
        IAsyncEnumerable<ChatStreamData> source, GenerationOptions? options = null, IToolCallStreamParser? parser = null)
    {
        var chunks = new List<ChatStreamChunk>();
        await foreach (var c in LlamaServerGeneratorModel.ShapeChatStream(
            source, options ?? new GenerationOptions(), parser, suppressServerCallsWhenParserActive: false, Ct))
            chunks.Add(c);
        return chunks;
    }

    [Fact]
    public async Task UsageAfterFinish_IsCarriedOnTheSingleFinalChunk()
    {
        var chunks = await ShapeAsync(Stream(
            new ChatStreamData { TextDelta = "yes" },
            new ChatStreamData { FinishReason = "stop" },
            new ChatStreamData { Usage = ServerUsage }));

        chunks.Count(c => c.FinishReason is not null).Should().Be(1);
        var last = chunks[^1];
        last.FinishReason.Should().Be("stop");
        last.Usage.Should().BeEquivalentTo(new ChatTokenUsage { PromptTokens = 41, CompletionTokens = 512, TotalTokens = 553 });
        chunks.Take(chunks.Count - 1).Should().OnlyContain(c => c.Usage == null && c.FinishReason == null);
        string.Concat(chunks.Select(c => c.Text)).Should().Be("yes");
    }

    [Fact]
    public async Task ServerTimings_RideTheSameFinalChunk_AsTheServerReportedThem()
    {
        var chunks = await ShapeAsync(Stream(
            new ChatStreamData { TextDelta = "yes" },
            new ChatStreamData { FinishReason = "stop" },
            new ChatStreamData { Usage = ServerUsage, Timings = ServerTimings }));

        var last = chunks[^1];
        last.FinishReason.Should().Be("stop");
        last.Timings.Should().BeEquivalentTo(new GenerationTimings
        {
            CachedPromptTokens = 9,
            PromptTokensEvaluated = 32,
            PromptDuration = TimeSpan.FromMilliseconds(21.5),
            PromptTokensPerSecond = 185.7,
            CompletionDuration = TimeSpan.FromMilliseconds(4300.0),
            CompletionTokensPerSecond = 119.0,
        });
        chunks.Take(chunks.Count - 1).Should().OnlyContain(c => c.Timings == null);
    }

    [Fact]
    public async Task TextReleasedByTheParserFlush_ArrivesBeforeTheFinishReason()
    {
        var chunks = await ShapeAsync(
            Stream(new ChatStreamData { TextDelta = "Hello" }, new ChatStreamData { FinishReason = "stop" }, new ChatStreamData { Usage = ServerUsage }),
            parser: new HoldingParser());

        chunks.Select(c => c.Text).Where(t => t is not null).Should().Equal("He", "llo");
        chunks[^1].FinishReason.Should().Be("stop", "the flushed tail belongs to the turn the finish reason ends");
        chunks[^1].Usage.Should().NotBeNull();
    }

    [Fact]
    public async Task NoUsageFromTheBackend_FinalChunkHasTheFinishReasonAndNullUsage()
    {
        var chunks = await ShapeAsync(Stream(new ChatStreamData { TextDelta = "hi" }, new ChatStreamData { FinishReason = "length" }));

        chunks[^1].FinishReason.Should().Be("length");
        chunks[^1].Usage.Should().BeNull();
    }

    [Fact]
    public async Task TheClientSideSafetyLimit_EndsWithLengthAndNoUsage()
    {
        var chunks = await ShapeAsync(
            Stream(
                new ChatStreamData { TextDelta = "a" },
                new ChatStreamData { TextDelta = "b" },
                new ChatStreamData { TextDelta = "c" },
                new ChatStreamData { FinishReason = "stop" },
                new ChatStreamData { Usage = ServerUsage, Timings = ServerTimings }),
            new GenerationOptions { MaxTokens = 2 });

        string.Concat(chunks.Select(c => c.Text)).Should().Be("ab");
        chunks[^1].FinishReason.Should().Be("length");
        chunks[^1].Usage.Should().BeNull("the stream was cut before the server reported its count");
        chunks.Should().OnlyContain(c => c.Timings == null, "the server's timings describe a completion the consumer did not get");
    }

    /// <summary>Holds the last three characters of every delta and releases them on flush.</summary>
    private sealed class HoldingParser : IToolCallStreamParser
    {
        private string _held = "";

        public ToolCallStreamResult Feed(string textDelta)
        {
            var all = _held + textDelta;
            var cut = Math.Max(0, all.Length - 3);
            _held = all[cut..];
            return new ToolCallStreamResult { Text = cut == 0 ? null : all[..cut] };
        }

        public ToolCallStreamResult Flush()
        {
            var tail = _held;
            _held = "";
            return new ToolCallStreamResult { Text = tail.Length == 0 ? null : tail };
        }
    }
}
