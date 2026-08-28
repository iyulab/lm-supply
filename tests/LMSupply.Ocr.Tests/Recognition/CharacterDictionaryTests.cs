using AwesomeAssertions;
using LMSupply.Ocr.Recognition;

namespace LMSupply.Ocr.Tests.Recognition;

public class CharacterDictionaryTests : IDisposable
{
    private readonly string _tempDir;

    public CharacterDictionaryTests()
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

    private string CreateDictFile(params string[] characters)
    {
        var path = Path.Combine(_tempDir, "dict.txt");
        File.WriteAllLines(path, characters);
        return path;
    }

    // --- Constructor & Count ---

    [Fact]
    public void Constructor_LoadsCharactersFromFile()
    {
        var path = CreateDictFile("a", "b", "c");

        var dict = new CharacterDictionary(path);

        // blank(0) + a(1) + b(2) + c(3) + space(4)
        dict.Count.Should().Be(5);
    }

    [Fact]
    public void Constructor_WithoutSpace_ExcludesSpace()
    {
        var path = CreateDictFile("a", "b");

        var dict = new CharacterDictionary(path, useSpace: false);

        // blank(0) + a(1) + b(2)
        dict.Count.Should().Be(3);
        dict.Contains(" ").Should().BeFalse();
    }

    [Fact]
    public void Constructor_SpaceAlreadyInDict_DoesNotDuplicate()
    {
        var path = CreateDictFile("a", " ", "b");

        var dict = new CharacterDictionary(path);

        // blank(0) + a(1) + space(2) + b(3) — space already present, not duplicated
        dict.Count.Should().Be(4);
        dict.Contains(" ").Should().BeTrue();
    }

    [Fact]
    public void Constructor_EmptyFile_OnlyBlankAndSpace()
    {
        var path = CreateDictFile();

        var dict = new CharacterDictionary(path);

        // blank(0) + space(1)
        dict.Count.Should().Be(2);
    }

    [Fact]
    public void Constructor_SkipsEmptyLines()
    {
        var path = CreateDictFile("a", "", "b", "", "c");

        var dict = new CharacterDictionary(path);

        // blank(0) + a(1) + b(2) + c(3) + space(4) — empty lines skipped
        dict.Count.Should().Be(5);
    }

    // --- BlankIndex ---

    [Fact]
    public void BlankIndex_IsAlwaysZero()
    {
        var path = CreateDictFile("x", "y");

        var dict = new CharacterDictionary(path);

        dict.BlankIndex.Should().Be(0);
    }

    // --- GetCharacter ---

    [Fact]
    public void GetCharacter_ValidIndex_ReturnsCharacter()
    {
        var path = CreateDictFile("a", "b", "c");

        var dict = new CharacterDictionary(path);

        dict.GetCharacter(0).Should().Be("<blank>");
        dict.GetCharacter(1).Should().Be("a");
        dict.GetCharacter(2).Should().Be("b");
        dict.GetCharacter(3).Should().Be("c");
    }

    [Fact]
    public void GetCharacter_NegativeIndex_ReturnsEmpty()
    {
        var path = CreateDictFile("a");

        var dict = new CharacterDictionary(path);

        dict.GetCharacter(-1).Should().BeEmpty();
    }

    [Fact]
    public void GetCharacter_OutOfRange_ReturnsEmpty()
    {
        var path = CreateDictFile("a");

        var dict = new CharacterDictionary(path);

        dict.GetCharacter(999).Should().BeEmpty();
    }

    // --- GetIndex ---

    [Fact]
    public void GetIndex_ExistingCharacter_ReturnsIndex()
    {
        var path = CreateDictFile("a", "b", "c");

        var dict = new CharacterDictionary(path);

        dict.GetIndex("b").Should().Be(2);
    }

    [Fact]
    public void GetIndex_NonExisting_ReturnsBlankIndex()
    {
        var path = CreateDictFile("a", "b");

        var dict = new CharacterDictionary(path);

        dict.GetIndex("z").Should().Be(dict.BlankIndex);
    }

    // --- Contains ---

    [Fact]
    public void Contains_ExistingCharacter_ReturnsTrue()
    {
        var path = CreateDictFile("a", "b");

        var dict = new CharacterDictionary(path);

        dict.Contains("a").Should().BeTrue();
        dict.Contains("b").Should().BeTrue();
    }

    [Fact]
    public void Contains_NonExisting_ReturnsFalse()
    {
        var path = CreateDictFile("a");

        var dict = new CharacterDictionary(path);

        dict.Contains("z").Should().BeFalse();
    }

    // --- Decode (CTC) ---

    [Fact]
    public void Decode_SimpleSequence_ReturnsText()
    {
        var path = CreateDictFile("h", "e", "l", "o");
        var dict = new CharacterDictionary(path, useSpace: false);
        // blank=0, h=1, e=2, l=3, o=4

        var indices = new[] { 1, 2, 3, 3, 4 };

        var result = dict.Decode(indices);

        // CTC decoding: removes duplicates (two consecutive 3's -> one 'l')
        result.Should().Be("helo");
    }

    [Fact]
    public void Decode_WithBlanks_SkipsBlanks()
    {
        var path = CreateDictFile("a", "b");
        var dict = new CharacterDictionary(path, useSpace: false);
        // blank=0, a=1, b=2

        var indices = new[] { 0, 1, 0, 2, 0 };

        var result = dict.Decode(indices);

        result.Should().Be("ab");
    }

    [Fact]
    public void Decode_RepeatsWithBlanksBetween_PreservesBoth()
    {
        var path = CreateDictFile("l");
        var dict = new CharacterDictionary(path, useSpace: false);
        // blank=0, l=1

        // "ll" is encoded as: l, blank, l (blank separates same character)
        var indices = new[] { 1, 0, 1 };

        var result = dict.Decode(indices);

        result.Should().Be("ll");
    }

    [Fact]
    public void Decode_AllBlanks_ReturnsEmpty()
    {
        var path = CreateDictFile("a", "b");
        var dict = new CharacterDictionary(path, useSpace: false);

        var indices = new[] { 0, 0, 0, 0 };

        var result = dict.Decode(indices);

        result.Should().BeEmpty();
    }

    [Fact]
    public void Decode_EmptySequence_ReturnsEmpty()
    {
        var path = CreateDictFile("a");
        var dict = new CharacterDictionary(path, useSpace: false);

        var result = dict.Decode(Array.Empty<int>());

        result.Should().BeEmpty();
    }
}
