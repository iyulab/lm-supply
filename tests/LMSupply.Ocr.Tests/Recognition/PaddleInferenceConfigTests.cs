using AwesomeAssertions;
using LMSupply.Ocr.Recognition;

namespace LMSupply.Ocr.Tests.Recognition;

public class PaddleInferenceConfigTests : IDisposable
{
    private readonly string _tempDir;

    public PaddleInferenceConfigTests()
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

    private string Write(string yaml)
    {
        var path = Path.Combine(_tempDir, "inference.yml");
        File.WriteAllText(path, yaml);
        return path;
    }

    private static string Config(params string[] entries) =>
        "Global:\n  model_name: x\nPostProcess:\n  name: CTCLabelDecode\n  character_dict:\n"
        + string.Concat(entries.Select(e => "  - " + e + "\n"))
        + "  use_space_char: true\n";

    [Fact]
    public void Reads_PlainAndSingleQuotedAndDoubleQuotedEntries()
    {
        var path = Write(Config("a", "$", "'!'", "''''", "'\"'", "\"\\\"\"", "\"\\\\\"", "\"\\u0e01\"", "\"\\x41\"", "\"\\U0001F600\""));

        PaddleInferenceConfig.ReadCharacterDictionary(path)
            .Should().Equal("a", "$", "!", "'", "\"", "\"", "\\", "\u0e01", "A", "\U0001F600");
    }

    [Fact]
    public void Reads_AQuotedSpaceAsASpace_AndKeepsOrder()
    {
        var path = Write(Config("'0'", "' '", "z"));

        PaddleInferenceConfig.ReadCharacterDictionary(path).Should().Equal("0", " ", "z");
    }

    [Fact]
    public void Reads_TheEscapesYamlDefines()
    {
        var path = Write(Config("\"\\a\"", "\"\\b\"", "\"\\e\"", "\"\\f\"", "\"\\v\"", "\"\\_\"", "\"\\N\"", "\"\\L\"", "\"\\P\""));

        PaddleInferenceConfig.ReadCharacterDictionary(path)
            .Should().Equal("\a", "\b", "\u001b", "\f", "\v", "\u00a0", "\u0085", "\u2028", "\u2029");
    }

    [Fact]
    public void ABlankLineOrACommentBetweenEntries_DoesNotEndTheList()
    {
        var path = Write("PostProcess:\n  character_dict:\n  - a\n\n  # a comment\n  - b\n  use_space_char: true\n");

        PaddleInferenceConfig.ReadCharacterDictionary(path).Should().Equal("a", "b");
    }

    [Fact]
    public void APlainEntry_LosesItsTrailingSpaces_AsYamlReadsIt()
    {
        var path = Write(Config("b   "));

        PaddleInferenceConfig.ReadCharacterDictionary(path).Should().Equal("b");
    }

    [Fact]
    public void StopsAtTheFirstLineThatIsNotAnEntry()
    {
        var path = Write(Config("a", "b"));

        PaddleInferenceConfig.ReadCharacterDictionary(path).Should().Equal("a", "b");
    }

    [Fact]
    public void ReadsAListIndentedUnderItsKey()
    {
        var path = Write("PostProcess:\n  character_dict:\n    - a\n    - 'b'\n  next: 1\n");

        PaddleInferenceConfig.ReadCharacterDictionary(path).Should().Equal("a", "b");
    }

    [Fact]
    public void Refuses_AFileWithoutTheList()
    {
        var path = Write("Global:\n  model_name: x\n");

        var act = () => PaddleInferenceConfig.ReadCharacterDictionary(path);

        act.Should().Throw<InvalidDataException>().WithMessage("*no 'character_dict' list*");
    }

    [Theory]
    [InlineData("\"\\q\"", "the escape")]
    [InlineData("'a' x", "text after a closing quote")]
    [InlineData("'unterminated", "unterminated")]
    [InlineData("&anchor", "starting with")]
    [InlineData("[a, b]", "starting with")]
    [InlineData("|", "starting with")]
    [InlineData("a: b", "plain entry")]
    [InlineData("\"\\u00\"", "hexadecimal")]
    [InlineData("''", "empty entry")]
    [InlineData("\"\"", "empty entry")]
    [InlineData("null", "non-text")]
    [InlineData("~", "non-text")]
    [InlineData("True", "non-text")]
    [InlineData("'a\u0085'", "NEL, LS or PS")]
    [InlineData("'\u2028'", "NEL, LS or PS")]
    [InlineData("'\u2029'", "NEL, LS or PS")]
    public void Refuses_SyntaxItDoesNotRead_InsteadOfGuessing(string entry, string expectedInMessage)
    {
        var path = Write(Config(entry));

        var act = () => PaddleInferenceConfig.ReadCharacterDictionary(path);

        act.Should().Throw<InvalidDataException>().WithMessage($"*{expectedInMessage}*").And.Message.Should().Contain("line 6");
    }

    [Fact]
    public void CharacterDictionary_ReadsAnInferenceYml_LikeADictFile()
    {
        var yml = Write(Config("a", "'b'", "c"));
        var txt = Path.Combine(_tempDir, "dict.txt");
        File.WriteAllLines(txt, ["a", "b", "c"]);

        var fromYml = new CharacterDictionary(yml);
        var fromTxt = new CharacterDictionary(txt);

        fromYml.Count.Should().Be(fromTxt.Count);
        for (var i = 0; i < fromTxt.Count; i++)
            fromYml.GetCharacter(i).Should().Be(fromTxt.GetCharacter(i));
        fromYml.Decode([1, 2, 3]).Should().Be("abc");
    }
}
