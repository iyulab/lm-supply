using AwesomeAssertions;
using LMSupply.Generator.Abstractions;
using LMSupply.Generator.Internal.Llama;
using LMSupply.Generator.Models;
using LMSupply.Llama.Server;
using NSubstitute;
using Xunit;

namespace LMSupply.Generator.Tests.Gguf;

/// <summary>
/// <see cref="GenerationOptions.TopLogprobs"/> on the generator surface: llama-server's per-token log probabilities become
/// <see cref="TokenLogprob"/> entries on results and stream chunks, and a generator that cannot report them refuses the
/// option instead of ignoring it.
/// </summary>
public class TokenLogprobsTests
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

    private static LlamaTokenLogprob Token(int id, string text, double logprob, params LlamaTokenLogprob[] top) => new()
    {
        Id = id,
        Token = text,
        Bytes = System.Text.Encoding.UTF8.GetBytes(text).Select(b => (int)b).ToList(),
        Logprob = logprob,
        TopLogprobs = top.ToList(),
    };

    private static LlamaTokenLogprob Candidate(int id, string text, double logprob) => new()
    {
        Id = id,
        Token = text,
        Bytes = System.Text.Encoding.UTF8.GetBytes(text).Select(b => (int)b).ToList(),
        Logprob = logprob,
    };

    private static async Task<List<ChatStreamChunk>> ShapeAsync(IAsyncEnumerable<ChatStreamData> source, GenerationOptions options)
    {
        var chunks = new List<ChatStreamChunk>();
        await foreach (var c in LlamaServerGeneratorModel.ShapeChatStream(
            source, options, toolStreamParser: null, Ct))
            chunks.Add(c);
        return chunks;
    }

    [Fact]
    public void ServerLogprobs_MapFieldForField()
    {
        var mapped = LlamaServerGeneratorModel.ToTokenLogprobs(
        [
            Token(33, "B", -0.034990593791007996, Candidate(33, "B", -0.034990593791007996), Candidate(44, "M", -3.7268946170806885)),
            new LlamaTokenLogprob { Id = 248046, Token = "", Bytes = [], Logprob = -0.00039603206096217036, TopLogprobs = [] },
        ])!;

        mapped.Should().HaveCount(2);
        var answer = mapped[0];
        answer.TokenId.Should().Be(33);
        answer.Token.Should().Be("B");
        answer.Logprob.Should().Be(-0.034990593791007996);
        answer.Bytes.Should().Equal((byte)'B');
        answer.TopLogprobs.Select(c => (c.TokenId, c.Token, c.Logprob)).Should().Equal(
            (33, "B", -0.034990593791007996),
            (44, "M", -3.7268946170806885));
        answer.TopLogprobs[1].Bytes.Should().Equal((byte)'M');

        mapped[1].Token.Should().BeEmpty();
        mapped[1].Bytes.Should().BeEmpty();
        mapped[1].TopLogprobs.Should().BeEmpty();
    }

    [Fact]
    public void ABytesValueAbove127_SurvivesAsAByte()
    {
        // A token holding part of a multi-byte character: the first byte of "é" (0xC3).
        var mapped = LlamaServerGeneratorModel.ToTokenLogprobs([new LlamaTokenLogprob { Token = "", Bytes = [195], Logprob = -1 }])!;

        mapped[0].Bytes.Should().Equal((byte)0xC3);
        mapped[0].TokenId.Should().BeNull("the server did not send one");
    }

    [Fact]
    public void NoServerLogprobs_MapToNull()
    {
        LlamaServerGeneratorModel.ToTokenLogprobs(null).Should().BeNull();
        LlamaServerGeneratorModel.ToTokenLogprobs([]).Should().BeNull();
    }

    [Fact]
    public async Task StreamedChunks_CarryTheLogprobsOfTheirTokens()
    {
        var chunks = await ShapeAsync(Stream(
            new ChatStreamData { TextDelta = "B", Logprobs = [Token(33, "B", -0.03, Candidate(33, "B", -0.03))] },
            new ChatStreamData { FinishReason = "stop" }), new GenerationOptions { TopLogprobs = 1 });

        var carrier = chunks.Single(c => c.Logprobs is not null);
        carrier.Text.Should().Be("B");
        carrier.Logprobs!.Single().TokenId.Should().Be(33);
        chunks[^1].FinishReason.Should().Be("stop");
    }

    [Fact]
    public async Task ReasoningTokens_LogprobsLeaveEvenWhenTheirTextIsNotShown()
    {
        // ExtractReasoningTokens is off, so the reasoning text is dropped — its tokens were still generated.
        var chunks = await ShapeAsync(Stream(
            new ChatStreamData { ReasoningDelta = "Thinking", Logprobs = [Token(90700, "Thinking", -0.37)] },
            new ChatStreamData { TextDelta = "B", Logprobs = [Token(33, "B", -0.03)] }), new GenerationOptions { TopLogprobs = 0 });

        chunks.SelectMany(c => c.Logprobs ?? []).Select(t => t.TokenId).Should().Equal(90700, 33);
        var reasoning = chunks.First();
        reasoning.Text.Should().BeNull();
        reasoning.ReasoningDelta.Should().BeNull();
    }

    [Fact]
    public async Task WithoutLogprobs_NoChunkCarriesThem()
    {
        var chunks = await ShapeAsync(Stream(
            new ChatStreamData { TextDelta = "B" },
            new ChatStreamData { FinishReason = "stop" }), new GenerationOptions());

        chunks.Should().OnlyContain(c => c.Logprobs == null);
    }

    [Fact]
    public void TopLogprobs_IsOffByDefault()
        => new GenerationOptions().TopLogprobs.Should().BeNull();

    [Fact]
    public void AGeneratorThatCannotReportLogprobs_RefusesTheOption()
    {
        var unset = () => new GenerationOptions().ThrowIfTopLogprobsUnsupported("Test generator");
        var set = () => new GenerationOptions { TopLogprobs = 0 }.ThrowIfTopLogprobsUnsupported("Test generator");

        unset.Should().NotThrow();
        set.Should().Throw<NotSupportedException>().WithMessage("Test generator*");
    }

    [Fact]
    public async Task SpeculativeDecoding_RefusesTopLogprobs()
    {
        var draft = Substitute.For<IGeneratorModel>();
        var target = Substitute.For<IGeneratorModel>();
        using var decoder = new SpeculativeDecoder(draft, target);

        var generate = async () =>
        {
            await foreach (var _ in decoder.GenerateAsync("Hi", new GenerationOptions { TopLogprobs = 1 }, Ct)) { }
        };

        await generate.Should().ThrowAsync<NotSupportedException>();
        draft.DidNotReceiveWithAnyArgs().GenerateAsync(default!, default, Ct);
    }

    [Fact]
    public void Clone_CarriesEveryOption()
    {
        // A copy that lists the options it carries dropped TopLogprobs (and others) on the ONNX chat path and in the
        // reasoning helpers; the clone must carry every settable property, including ones added later.
        var source = new GenerationOptions
        {
            TopLogprobs = 3, Seed = 7, MinP = 0.2f, NoRepeatNgramSize = 4, Grammar = "root ::= \"a\"",
            Thinking = ThinkingMode.Off, StopSequences = ["x"], MaxTokens = 9,
        };

        var clone = source.Clone();

        clone.Should().NotBeSameAs(source);
        foreach (var property in typeof(GenerationOptions).GetProperties().Where(p => p.CanWrite))
            property.GetValue(clone).Should().Be(property.GetValue(source), property.Name);
    }

    [Fact]
    public async Task ReasoningHelpers_RefuseTopLogprobs_AndForwardTheOtherOptions()
    {
        var generator = Substitute.For<ITextGenerator>();
        GenerationOptions? sent = null;
        generator.GenerateChatAsync(Arg.Any<IEnumerable<ChatMessage>>(), Arg.Do<GenerationOptions?>(o => sent = o), Arg.Any<CancellationToken>())
            .Returns(Stream<string>("hi"));

        var refused = () => generator.GenerateChatWithReasoningAsync([ChatMessage.User("q")], new GenerationOptions { TopLogprobs = 1 }, Ct);
        await refused.Should().ThrowAsync<NotSupportedException>();

        await generator.GenerateChatWithReasoningAsync([ChatMessage.User("q")], new GenerationOptions { Seed = 7, Thinking = ThinkingMode.Off }, Ct);
        sent!.Seed.Should().Be(7);
        sent.Thinking.Should().Be(ThinkingMode.Off);
        sent.ExtractReasoningTokens.Should().BeTrue();
    }

    private static async IAsyncEnumerable<T> Stream<T>(params T[] items)
    {
        foreach (var item in items)
        {
            await Task.Yield();
            yield return item;
        }
    }
}
