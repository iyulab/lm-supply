using AwesomeAssertions;
using LMSupply.Text;

namespace LMSupply.Text.Tests;

/// <summary>
/// Regression tests for tokenizer padding semantics: <c>maxLength</c> is a truncation cap, a single
/// sequence is sized to its real content, and a batch is padded only to its longest member.
///
/// Background: every encoder padded to <c>maxLength</c>, and the embedder promotes its default cap
/// to the model maximum (8192 for bge-m3), so a 10-token sentence ran a full 8192-token pass —
/// measured at ~39s per sentence on CPU. See the issue draft
/// "encode-sequence-pads-to-max-length" in the umbrella's lm-supply issue directory.
/// </summary>
public class DynamicPaddingTests : IDisposable
{
    private readonly string _tempDir;

    public DynamicPaddingTests()
    {
        _tempDir = Path.Combine(Path.GetTempPath(), $"tk-pad-tests-{Guid.NewGuid():N}");
        Directory.CreateDirectory(_tempDir);
        // Minimal BERT-style vocab: ids 0..6 in file order.
        File.WriteAllText(Path.Combine(_tempDir, "vocab.txt"), "[PAD]\n[UNK]\n[CLS]\n[SEP]\nhello\nworld\nagain\n");
    }

    public void Dispose()
    {
        if (Directory.Exists(_tempDir))
            Directory.Delete(_tempDir, true);
        GC.SuppressFinalize(this);
    }

    [Fact]
    public async Task EncodeSequence_ShortText_IsSizedToContentNotCap()
    {
        var tokenizer = await TokenizerFactory.CreateAutoSequenceAsync(_tempDir, maxSequenceLength: 64);

        var encoded = tokenizer.EncodeSequence("hello world");

        // [CLS] hello world [SEP] — four tokens, not 64.
        encoded.InputIds.Length.Should().Be(4);
        encoded.AttentionMask.Length.Should().Be(4);
        encoded.Length.Should().Be(4);
        encoded.AttentionMask.Should().AllSatisfy(m => m.Should().Be(1));
    }

    [Fact]
    public async Task EncodeSequence_LongText_IsTruncatedToCap()
    {
        var tokenizer = await TokenizerFactory.CreateAutoSequenceAsync(_tempDir, maxSequenceLength: 8);

        var encoded = tokenizer.EncodeSequence(string.Join(' ', Enumerable.Repeat("hello", 20)));

        // 6 content tokens + [CLS] + [SEP] — the cap still bounds the sequence.
        encoded.InputIds.Length.Should().Be(8);
        encoded.Length.Should().Be(8);
    }

    [Fact]
    public async Task EncodeBatch_PadsToLongestMember_NotToCap()
    {
        var tokenizer = await TokenizerFactory.CreateAutoSequenceAsync(_tempDir, maxSequenceLength: 64);

        var batch = tokenizer.EncodeBatch(["hello", "hello world again"]);

        // Longest member is [CLS] hello world again [SEP] = 5.
        batch.SequenceLength.Should().Be(5);
        batch.BatchSize.Should().Be(2);

        // First row: [CLS] hello [SEP] + two pad slots (pad id, mask 0).
        batch.AttentionMask[0, 0].Should().Be(1);
        batch.AttentionMask[0, 2].Should().Be(1);
        batch.AttentionMask[0, 3].Should().Be(0);
        batch.AttentionMask[0, 4].Should().Be(0);
        batch.InputIds[0, 3].Should().Be(tokenizer.PadTokenId);
        batch.InputIds[0, 4].Should().Be(tokenizer.PadTokenId);

        // Second row fully attended.
        for (int j = 0; j < 5; j++)
            batch.AttentionMask[1, j].Should().Be(1);
    }

    [Fact]
    public async Task EncodePairBatch_PadsToLongestPair_NotToCap()
    {
        var tokenizer = await TokenizerFactory.CreateAutoPairAsync(_tempDir, maxSequenceLength: 64);

        var batch = tokenizer.EncodePairBatch("hello", ["world", "world again"]);

        // Longest pair: [CLS] hello [SEP] world again [SEP] = 6.
        batch.SequenceLength.Should().Be(6);
        batch.AttentionMask[0, 5].Should().Be(0, "the shorter pair is right-padded");
        batch.InputIds[0, 5].Should().Be(tokenizer.PadTokenId);
        batch.TokenTypeIds[0, 5].Should().Be(0);
        batch.AttentionMask[1, 5].Should().Be(1);
    }

    [Fact]
    public void EncodedBatch_SetSequence_ShorterThanBatch_RightPadsWithGivenPadId()
    {
        var batch = new EncodedBatch(1, 4);

        batch.SetSequence(0, new EncodedSequence([7, 8], [1, 1], 2), padTokenId: 99);

        batch.InputIds[0, 0].Should().Be(7);
        batch.InputIds[0, 1].Should().Be(8);
        batch.InputIds[0, 2].Should().Be(99);
        batch.InputIds[0, 3].Should().Be(99);
        batch.AttentionMask[0, 2].Should().Be(0);
        batch.AttentionMask[0, 3].Should().Be(0);
    }
}
