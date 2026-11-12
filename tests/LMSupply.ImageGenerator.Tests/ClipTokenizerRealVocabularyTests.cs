using System.Net.Http;
using AwesomeAssertions;
using LMSupply.ImageGenerator.Tokenizers;

namespace LMSupply.ImageGenerator.Tests;

/// <summary>
/// Checks the encoder against ids produced by the reference CLIP tokenizer (the original byte-pair implementation with its text
/// cleaning library) on the real vocabulary. The files are small and come from the public model repository, so the fact
/// needs the network and runs only in the live-web set.
/// </summary>
[Trait("Category", "Integration")]
[Trait("Category", "LiveWeb")]
public sealed class ClipTokenizerRealVocabularyTests : IAsyncLifetime
{
    private const string Repository = "https://huggingface.co/openai/clip-vit-base-patch32/resolve/main/";

    private string _directory = string.Empty;
    private ClipTokenizer _tokenizer = null!;

    public async ValueTask InitializeAsync()
    {
        _directory = Path.Combine(Path.GetTempPath(), "clip-real-vocabulary-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(_directory);
        using var http = new HttpClient { Timeout = TimeSpan.FromMinutes(2) };
        foreach (var file in new[] { "vocab.json", "merges.txt" })
        {
            var bytes = await http.GetByteArrayAsync(Repository + file, TestContext.Current.CancellationToken);
            await File.WriteAllBytesAsync(Path.Combine(_directory, file), bytes, TestContext.Current.CancellationToken);
        }

        _tokenizer = ClipTokenizer.FromDirectory(_directory);
    }

    public ValueTask DisposeAsync()
    {
        _tokenizer.Dispose();
        Directory.Delete(_directory, recursive: true);
        return ValueTask.CompletedTask;
    }

    // Text and the ids the reference tokenizer gives it (no start or end marker).
    private static readonly (string Text, int[] Ids)[] References =
    [
        ("a photo of a cat", [320, 1125, 539, 320, 2368]),
        ("A Photograph of an Astronaut riding a horse, 4k, highly detailed", [320, 8853, 539, 550, 18376, 6765, 320, 4558, 267, 275, 330, 267, 5302, 12609]),
        ("\ud55c\uad6d\uc5b4 \ubb38\uc7a5\ub3c4 \ub4e4\uc5b4\uac04\ub2e4", [15197, 250, 31871, 255, 31625, 368, 167, 105, 116, 20849, 98, 31972, 482, 30850, 97, 31625, 112, 21122, 25170, 353]),
        ("na\u00efve caf\u00e9 \u2014 \u201cquotes\u201d & \u2018single\u2019 numbers 12345", [1097, 35689, 563, 15304, 2005, 257, 5808, 257, 261, 568, 17005, 262, 6121, 272, 273, 274, 275, 276]),
        ("", []),
        ("   leading and trailing   ", [3833, 537, 37427]),
        ("a\nnewline\tand tab", [320, 1218, 1148, 537, 14724]),
        ("emoji \U0001f600 test", [16327, 7334, 1628]),
        ("supercalifragilisticexpialidocious", [1642, 2857, 13093, 2076, 5868, 26850, 835, 639, 38466]),
        ("it's we're they'll don't 100%", [585, 568, 649, 982, 889, 1342, 847, 713, 272, 271, 271, 260]),
        ("snake_case and CamelCase", [8798, 318, 2068, 537, 27005, 2068]),
        ("a<|startoftext|>b<|endoftext|>c", [320, 49406, 321, 49407, 322]),
        ("fullwidth \uff21\uff22\uff23 \uff11\uff12\uff13 and a \ufb01ne \ufb02ower", [9407, 23571, 5334, 272, 273, 274, 537, 320, 3797, 4055]),
        ("&amp;lt;b&amp;gt; &amp;amp; entities", [283, 321, 285, 261, 38134]),
        ("cafe\u0301 composed", [15304, 19916]),
        ("a very long prompt a very long prompt a very long prompt of repeated words", [320, 1070, 1538, 20198, 320, 1070, 1538, 20198, 320, 1070, 1538, 20198, 539, 27904, 2709]),
        ("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx", [32035, 32035, 32035, 32035, 32035, 32035, 32035, 32035, 32035, 22819]),
        ("a\u2014b\u2026c", [320, 2005, 321, 959, 322]),
        ("1.5x faster!!! (really?)", [272, 269, 276, 343, 8835, 995, 263, 1414, 9626]),
        ("\u65e5\u672c\u8a9e\u306e\u30d7\u30ed\u30f3\u30d7\u30c8", [39121, 44353, 34002, 252, 21575, 2429, 245, 35827, 25914, 245, 2429, 486]),
    ];

    public static TheoryData<string, int[]> ReferenceIds
    {
        get
        {
            var data = new TheoryData<string, int[]>();
            foreach (var (text, ids) in References)
            {
                data.Add(text, ids);
            }

            return data;
        }
    }

    [Theory]
    [MemberData(nameof(ReferenceIds))]
    public void Encode_WithoutSpecialTokens_GivesTheReferenceIds(string text, int[] expected)
    {
        _tokenizer.Encode(text, addSpecialTokens: false).Should().Equal(expected);
    }

    [Fact]
    public void Encode_WithSpecialTokens_IsStartIdsEndThenPaddedToTheMaximumLength()
    {
        var ids = _tokenizer.Encode("a photo of a cat");

        ids.Should().HaveCount(ClipTokenizer.DefaultMaxLength);
        ids.Take(7).Should().Equal(49406, 320, 1125, 539, 320, 2368, 49407);
        ids.Skip(7).Should().OnlyContain(id => id == 49407);
    }
}
