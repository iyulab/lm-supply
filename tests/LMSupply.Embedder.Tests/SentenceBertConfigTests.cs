using AwesomeAssertions;
using LMSupply.Embedder.Utils;

namespace LMSupply.Embedder.Tests;

/// <summary>
/// A sentence-transformers model says where it truncates in <c>sentence_bert_config.json</c>.
/// Reading it is what makes a model loaded by repository id embed long inputs the way the
/// reference implementation does; an explicit caller value still wins.
/// </summary>
public sealed class SentenceBertConfigTests : IDisposable
{
    private readonly string _dir = Path.Combine(Path.GetTempPath(), $"sbert-{Guid.NewGuid():N}");

    public SentenceBertConfigTests() => Directory.CreateDirectory(_dir);

    public void Dispose()
    {
        if (Directory.Exists(_dir)) Directory.Delete(_dir, true);
    }

    private void Write(string json) => File.WriteAllText(Path.Combine(_dir, SentenceBertConfig.FileName), json);

    [Fact]
    public void TheDeclaredLength_IsRead()
    {
        Write("""{ "max_seq_length": 256, "do_lower_case": false }""");

        SentenceBertConfig.TryReadMaxSequenceLength(_dir).Should().Be(256);
    }

    [Fact]
    public void TheFirstDirectoryThatHasTheFile_Decides()
    {
        Write("""{ "max_seq_length": 384 }""");

        SentenceBertConfig.TryReadMaxSequenceLength(null, Path.Combine(_dir, "onnx"), _dir).Should().Be(384);
    }

    [Theory]
    [InlineData("""{ "do_lower_case": true }""")]
    [InlineData("""{ "max_seq_length": 0 }""")]
    [InlineData("""{ "max_seq_length": "256" }""")]
    [InlineData("""not json""")]
    public void AMissingOrUnusableValue_IsTheSameAsNoFile(string json)
    {
        Write(json);

        SentenceBertConfig.TryReadMaxSequenceLength(_dir).Should().BeNull();
    }

    [Fact]
    public void NoFile_IsNull() => SentenceBertConfig.TryReadMaxSequenceLength(_dir).Should().BeNull();

    [Theory]
    // caller left it unset: the model's declaration, then the catalog, then the default
    [InlineData(null, 256, 8192, 256)]
    [InlineData(null, null, 8192, 8192)]
    [InlineData(null, null, null, 512)]
    // caller chose a value: it wins over both — including exactly 512, which an int could not express
    [InlineData(128, 256, 8192, 128)]
    [InlineData(1024, 256, null, 1024)]
    [InlineData(512, 256, 8192, 512)]
    public void Precedence(int? caller, int? declared, int? catalog, int expected) =>
        SentenceBertConfig.ResolveMaxSequenceLength(caller, declared, catalog).Should().Be(expected);
}
