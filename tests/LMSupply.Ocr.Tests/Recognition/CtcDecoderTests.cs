using AwesomeAssertions;
using LMSupply.Ocr.Recognition;

namespace LMSupply.Ocr.Tests.Recognition;

public class CtcDecoderTests : IDisposable
{
    private readonly string _tempDir;

    public CtcDecoderTests()
    {
        _tempDir = Path.Combine(Path.GetTempPath(), "lm-supply-ocr-tests", Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(_tempDir);
    }

    public void Dispose()
    {
        if (Directory.Exists(_tempDir))
            Directory.Delete(_tempDir, recursive: true);
        GC.SuppressFinalize(this);
    }

    private CharacterDictionary CreateDict(params string[] characters)
    {
        var path = Path.Combine(_tempDir, $"dict-{Guid.NewGuid():N}.txt");
        File.WriteAllLines(path, characters);
        return new CharacterDictionary(path, useSpace: false);
    }

    // --- GreedyDecode 2D ---

    [Fact]
    public void GreedyDecode_2D_SingleTimestep_ReturnsCharacter()
    {
        var dict = CreateDict("a", "b", "c");
        // blank=0, a=1, b=2, c=3

        // 1 timestep, 4 vocab: highest logit at index 2 (b)
        var logits = new float[1, 4];
        logits[0, 0] = -1f; // blank
        logits[0, 1] = 0f;  // a
        logits[0, 2] = 5f;  // b (highest)
        logits[0, 3] = 1f;  // c

        var (text, confidence) = CtcDecoder.GreedyDecode(logits, dict);

        text.Should().Be("b");
        confidence.Should().BeGreaterThan(0f);
    }

    [Fact]
    public void GreedyDecode_2D_MultipleTimesteps_DecodesCorrectly()
    {
        var dict = CreateDict("h", "i");
        // blank=0, h=1, i=2

        // 3 timesteps, 3 vocab: sequence h, blank, i -> "hi"
        var logits = new float[3, 3];
        // timestep 0: h is highest
        logits[0, 0] = -1f; logits[0, 1] = 5f; logits[0, 2] = 0f;
        // timestep 1: blank is highest
        logits[1, 0] = 5f; logits[1, 1] = 0f; logits[1, 2] = 0f;
        // timestep 2: i is highest
        logits[2, 0] = -1f; logits[2, 1] = 0f; logits[2, 2] = 5f;

        var (text, _) = CtcDecoder.GreedyDecode(logits, dict);

        text.Should().Be("hi");
    }

    [Fact]
    public void GreedyDecode_2D_DuplicateCharacters_Deduplicated()
    {
        var dict = CreateDict("a");
        // blank=0, a=1

        // 3 timesteps: a, a, a -> "a" (CTC dedup)
        var logits = new float[3, 2];
        logits[0, 0] = -5f; logits[0, 1] = 5f; // a
        logits[1, 0] = -5f; logits[1, 1] = 5f; // a (duplicate)
        logits[2, 0] = -5f; logits[2, 1] = 5f; // a (duplicate)

        var (text, _) = CtcDecoder.GreedyDecode(logits, dict);

        text.Should().Be("a");
    }

    [Fact]
    public void GreedyDecode_2D_AllBlanks_ReturnsEmptyText()
    {
        var dict = CreateDict("a", "b");
        // blank=0, a=1, b=2

        // 2 timesteps: all blanks
        var logits = new float[2, 3];
        logits[0, 0] = 10f; logits[0, 1] = -5f; logits[0, 2] = -5f;
        logits[1, 0] = 10f; logits[1, 1] = -5f; logits[1, 2] = -5f;

        var (text, confidence) = CtcDecoder.GreedyDecode(logits, dict);

        text.Should().BeEmpty();
        confidence.Should().Be(0f); // No valid scores
    }

    [Fact]
    public void GreedyDecode_2D_ConfidenceIsAverageOfNonBlankProbs()
    {
        var dict = CreateDict("a");
        // blank=0, a=1

        // 2 timesteps: a, blank -> only "a"'s probability is used
        var logits = new float[2, 2];
        logits[0, 0] = 0f; logits[0, 1] = 10f; // a with high logit
        logits[1, 0] = 10f; logits[1, 1] = 0f;  // blank

        var (_, confidence) = CtcDecoder.GreedyDecode(logits, dict);

        // softmax(10) / (softmax(0) + softmax(10)) ≈ 0.99995
        confidence.Should().BeGreaterThan(0.99f);
        confidence.Should().BeLessThanOrEqualTo(1f);
    }

    // --- GreedyDecode 3D ---

    [Fact]
    public void GreedyDecode_3D_ExtractsBatchZero()
    {
        var dict = CreateDict("x", "y");
        // blank=0, x=1, y=2

        // Batch=1, T=2, V=3
        var logits = new float[1, 2, 3];
        logits[0, 0, 0] = -1f; logits[0, 0, 1] = 5f; logits[0, 0, 2] = 0f; // x
        logits[0, 1, 0] = -1f; logits[0, 1, 1] = 0f; logits[0, 1, 2] = 5f; // y

        var (text, _) = CtcDecoder.GreedyDecode(logits, dict);

        text.Should().Be("xy");
    }

    [Fact]
    public void GreedyDecode_3D_ConsistentWith2D()
    {
        var dict2d = CreateDict("a", "b");
        var dict3d = CreateDict("a", "b");

        var logits2D = new float[2, 3];
        logits2D[0, 0] = 0f; logits2D[0, 1] = 5f; logits2D[0, 2] = 1f;
        logits2D[1, 0] = 0f; logits2D[1, 1] = 1f; logits2D[1, 2] = 5f;

        var logits3D = new float[1, 2, 3];
        logits3D[0, 0, 0] = 0f; logits3D[0, 0, 1] = 5f; logits3D[0, 0, 2] = 1f;
        logits3D[0, 1, 0] = 0f; logits3D[0, 1, 1] = 1f; logits3D[0, 1, 2] = 5f;

        var result2D = CtcDecoder.GreedyDecode(logits2D, dict2d);
        var result3D = CtcDecoder.GreedyDecode(logits3D, dict3d);

        result2D.text.Should().Be(result3D.text);
        result2D.confidence.Should().BeApproximately(result3D.confidence, 1e-6f);
    }

    // --- Softmax numerical stability ---

    [Fact]
    public void GreedyDecode_2D_LargeLogits_DoesNotOverflow()
    {
        var dict = CreateDict("a");
        // blank=0, a=1

        var logits = new float[1, 2];
        logits[0, 0] = -1000f;
        logits[0, 1] = 1000f; // Very large logit

        var (text, confidence) = CtcDecoder.GreedyDecode(logits, dict);

        text.Should().Be("a");
        confidence.Should().BeGreaterThan(0f);
        float.IsNaN(confidence).Should().BeFalse();
        float.IsInfinity(confidence).Should().BeFalse();
    }

    [Fact]
    public void GreedyDecode_2D_NegativeLogits_WorksCorrectly()
    {
        var dict = CreateDict("a", "b");
        // blank=0, a=1, b=2

        // All negative logits; b is "least negative"
        var logits = new float[1, 3];
        logits[0, 0] = -10f;
        logits[0, 1] = -5f;
        logits[0, 2] = -2f; // b is highest

        var (text, _) = CtcDecoder.GreedyDecode(logits, dict);

        text.Should().Be("b");
    }
}
