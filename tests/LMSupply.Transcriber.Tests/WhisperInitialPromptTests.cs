using AwesomeAssertions;
using LMSupply.Transcriber.Decoding;

namespace LMSupply.Transcriber.Tests;

/// <summary>
/// <see cref="TranscribeOptions.InitialPrompt"/> was accepted and never read: the tokenizer could only
/// decode, so there was no way to put text in front of the audio. The tokenizer now encodes with GPT-2
/// byte-level BPE from the model's merges, and the decoder lays the prompt out as Whisper's reference
/// decoder does — <c>&lt;|startofprev|&gt;</c>, the prompt, then the start-of-transcript sequence.
/// </summary>
public sealed class WhisperInitialPromptTests : IDisposable
{
    private const int StartOfPrev = 50361;
    private const int StartOfTranscript = 50258;
    private const int Transcribe = 50359;
    private const int NoTimestamps = 50363;
    private static readonly int[] Sot = [StartOfTranscript, Transcribe, NoTimestamps];

    // A byte-level vocabulary (ids 0..255, one per byte) plus " the" built by three merges.
    private const int SpaceT = 256;
    private const int SpaceTh = 257;
    private const int SpaceThe = 258;

    private readonly string _dir = Path.Combine(Path.GetTempPath(), "lmsupply-bpe-" + Guid.NewGuid().ToString("N"));

    public void Dispose()
    {
        if (Directory.Exists(_dir))
            Directory.Delete(_dir, recursive: true);
    }

    [Fact]
    public void Encode_AppliesMergesInRankOrder()
    {
        Tokenizer().Encode(" the").Should().Equal(SpaceThe);
    }

    [Fact]
    public void Encode_MergesEachPreTokenizedWordOnItsOwn()
    {
        Tokenizer().Encode(" the the").Should().Equal(SpaceThe, SpaceThe);
        Tokenizer().Encode("the").Should().Equal(Bytes("the"), "without the leading space no merge applies");
    }

    [Theory]
    [InlineData("Attendees: Meena, Joon.")]
    [InlineData(" 회의록 — 3분기 OKR")]
    [InlineData("It's 10:30, isn't it?")]
    public void Decode_OfEncode_GivesTheTextBack(string text)
    {
        var tokenizer = Tokenizer();

        tokenizer.Decode(tokenizer.Encode(text)).Should().Be(text);
    }

    [Fact]
    public void WithoutMerges_EncodingSaysWhatIsMissing()
    {
        var tokenizer = WhisperTokenizer.CreateDefault();

        tokenizer.CanEncode.Should().BeFalse();
        var act = () => tokenizer.Encode("x");
        act.Should().Throw<NotSupportedException>().WithMessage("*merges*InitialPrompt*");
    }

    [Fact]
    public void NoPrompt_LeavesTheStartOfTranscriptSequenceAlone()
    {
        WhisperDecoder.BuildInitialTokens(Tokenizer(), null, Sot, 448).Should().Equal(Sot);
        WhisperDecoder.BuildInitialTokens(Tokenizer(), "   ", Sot, 448).Should().Equal(Sot);
    }

    [Fact]
    public void APrompt_GoesBehindStartOfPrev_BeforeTheStartOfTranscriptSequence()
    {
        var tokens = WhisperDecoder.BuildInitialTokens(Tokenizer(), "  the  ", Sot, 448);

        tokens.Should().Equal([StartOfPrev, SpaceThe, .. Sot], "the prompt is trimmed and encoded with one leading space");
    }

    [Fact]
    public void ALongPrompt_KeepsItsLastHalfContextMinusOneTokens()
    {
        var prompt = string.Join(" ", Enumerable.Repeat("the", 300));

        var tokens = WhisperDecoder.BuildInitialTokens(Tokenizer(), prompt, Sot, 448);

        tokens.Should().HaveCount(1 + 223 + Sot.Length);
        tokens[0].Should().Be(StartOfPrev);
        tokens[^Sot.Length..].Should().Equal(Sot);
    }

    [Fact]
    public async Task Load_ReadsMergesTxt()
    {
        WriteVocab();
        File.WriteAllLines(Path.Combine(_dir, "merges.txt"), ["#version: 0.2", "Ġ t", "Ġt h", "Ġth e"]);

        using var tokenizer = await WhisperTokenizer.LoadAsync(_dir, TestContext.Current.CancellationToken);

        tokenizer.Encode(" the").Should().Equal(SpaceThe);
    }

    [Fact]
    public async Task Load_FallsBackToTheMergesInTokenizerJson()
    {
        WriteVocab();
        File.WriteAllText(Path.Combine(_dir, "tokenizer.json"), """
            { "model": { "type": "BPE", "merges": [ "Ġ t", ["Ġt", "h"], "Ġth e" ] } }
            """);

        using var tokenizer = await WhisperTokenizer.LoadAsync(_dir, TestContext.Current.CancellationToken);

        tokenizer.Encode(" the").Should().Equal(SpaceThe);
    }

    private static WhisperTokenizer Tokenizer()
    {
        var (idToToken, tokenToId) = Vocabulary();
        return WhisperTokenizer.CreateForTesting(idToToken, tokenToId, [("Ġ", "t"), ("Ġt", "h"), ("Ġth", "e")]);
    }

    private static (Dictionary<int, string>, Dictionary<string, int>) Vocabulary()
    {
        var idToToken = new Dictionary<int, string>();
        for (var b = 0; b < 256; b++)
        {
            idToToken[b] = WhisperTokenizer.ByteToUnicode((byte)b);
        }

        idToToken[SpaceT] = "Ġt";
        idToToken[SpaceTh] = "Ġth";
        idToToken[SpaceThe] = "Ġthe";
        return (idToToken, idToToken.ToDictionary(kv => kv.Value, kv => kv.Key, StringComparer.Ordinal));
    }

    private void WriteVocab()
    {
        Directory.CreateDirectory(_dir);
        var (_, tokenToId) = Vocabulary();
        File.WriteAllText(Path.Combine(_dir, "vocab.json"), System.Text.Json.JsonSerializer.Serialize(tokenToId));
    }

    private static int[] Bytes(string ascii) => [.. ascii.Select(c => (int)c)];
}
