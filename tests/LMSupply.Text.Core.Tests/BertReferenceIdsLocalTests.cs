namespace LMSupply.Text.Tests;

/// <summary>
/// The same comparison against a real 30 522-piece BERT vocabulary: the token ids must be the ones the
/// reference implementation produces from the model's own <c>tokenizer.json</c>. Runs only where the
/// model is already in the HuggingFace cache — it never downloads — so it is a local check, not a CI one.
/// </summary>
[Trait("Category", "LocalOnly")]
public class BertReferenceIdsLocalTests
{
    private const string Repo = "models--cross-encoder--ms-marco-MiniLM-L-6-v2";

    // Produced by the HuggingFace `tokenizers` library from this model's tokenizer.json
    // (BertNormalizer: clean_text, handle_chinese_chars, lowercase; BertPreTokenizer), without special tokens.
    public static TheoryData<string, int[]> Reference => new()
    {
        { "hello world", [7592, 2088] },
        { "The QUICK brown fox — didn't jump over the lazy dog?!",
            [1996, 4248, 2829, 4419, 1517, 2134, 1005, 1056, 5376, 2058, 1996, 13971, 3899, 1029, 999] },
        { "Café crème brûlée in São Paulo, naïve résumé",
            [7668, 13675, 21382, 7987, 9307, 2063, 1999, 7509, 9094, 1010, 15743, 13746] },
        { "東京タワーの夜景 \U0001F680 is beautiful",
            [1879, 1755, 1709, 30262, 30265, 30197, 100, 100, 100, 2003, 3376] },
        { "line one\nline two\tTabbed $5 + 3 = 8",
            [2240, 2028, 2240, 2048, 21628, 8270, 1002, 1019, 1009, 1017, 1027, 1022] },
        { "Paris is the capital of France.", [3000, 2003, 1996, 3007, 1997, 2605, 1012] },
    };

    [Theory]
    [MemberData(nameof(Reference))]
    public async Task TokenIds_MatchTheReferenceImplementation(string text, int[] expected)
    {
        var modelDir = FindCachedSnapshot();
        if (modelDir is null)
        {
            Assert.Skip($"{Repo} is not in the local HuggingFace cache; this test never downloads.");
        }

        var tokenizer = await TokenizerFactory.CreateWordPieceAsync(modelDir, maxSequenceLength: 256);

        tokenizer.Encode(text, addSpecialTokens: false).Should().Equal(expected);
    }

    private static string? FindCachedSnapshot()
    {
        var snapshots = Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".cache", "huggingface", "hub", Repo, "snapshots");
        if (!Directory.Exists(snapshots))
            return null;

        return Directory.EnumerateDirectories(snapshots)
            .FirstOrDefault(d => File.Exists(Path.Combine(d, "vocab.txt")));
    }
}
