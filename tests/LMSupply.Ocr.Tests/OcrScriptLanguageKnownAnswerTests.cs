using AwesomeAssertions;
using Xunit;

namespace LMSupply.Ocr.Tests;

/// <summary>
/// Known-answer checks that a language code reaches a reader that can read the page. The registry tests say which recognizer a code
/// resolves to; these render a page in the language and ask what that reader sees, so a code that resolves to the wrong script, or to a
/// recognizer whose character set lacks the language's letters, shows up as unreadable text instead of as a passing lookup. They need
/// the models in the local cache, so they run at the release gate with the other local-only facts.
/// </summary>
[Trait("Category", "Integration")]
[Trait("Category", "LocalOnly")]
[Trait("Axis", "KnownAnswer")]
public class OcrScriptLanguageKnownAnswerTests
{
    private static readonly string[] SerbianLatin =
    [
        "Molimo vas da sačuvate račun; artikli se mogu zameniti u roku od sedam dana.",
        "Biblioteka je zatvorena ponedeljkom i tokom državnih praznika cele godine.",
        "Lozinka mora sadržati najmanje osam znakova uključujući slova i brojeve.",
    ];

    private static readonly string[] SerbianCyrillic =
    [
        "Молимо вас да сачувате рачун; артикли се могу заменити у року од седам дана.",
        "Библиотека је затворена понедељком и током државних празника целе године.",
        "Лозинка мора садржати најмање осам знакова укључујући слова и бројеве.",
    ];

    // Measured with the models in the cache, the character error rate (white space left out) is 0.000 for the Latin page and 0.005 for the
    // Cyrillic one (the full stop that ends a line is the only miss). The bound leaves room for another platform's font and arithmetic.
    private const double MaxCharacterErrorRate = 0.05;

    [Theory]
    [InlineData("sr-Latn")]
    [InlineData("sr_Latn")]
    [InlineData("sr-Latn-RS")]
    [InlineData("sr_RS@latin")]
    public async Task SerbianInLatinLetters_IsReadWithItsDiacritics(string code) =>
        await AssertRead(code, SerbianLatin);

    [Theory]
    [InlineData("sr")]
    [InlineData("sr-Cyrl")]
    [InlineData("sr_RS@cyrillic")]
    public async Task SerbianInCyrillicLetters_IsRead(string code) =>
        await AssertRead(code, SerbianCyrillic);

    private static async Task AssertRead(string code, string[] lines)
    {
        var png = KnownAnswerPage.Render(lines, width: 1300);

        await using var ocr = await LocalOcr.LoadForLanguageAsync(code, cancellationToken: TestContext.Current.CancellationToken);
        var result = await ocr.RecognizeAsync(png, TestContext.Current.CancellationToken);

        var heard = result.GetTextWithLayout(lineTolerancePixels: 20);
        var rate = KnownAnswerPage.CharacterErrorRate(string.Concat(lines), heard);

        rate.Should().BeLessThanOrEqualTo(MaxCharacterErrorRate, $"«{code}» was read as «{heard}» (error rate {rate:F3})");
    }
}
