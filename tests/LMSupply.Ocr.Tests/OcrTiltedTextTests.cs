using AwesomeAssertions;
using Xunit;

namespace LMSupply.Ocr.Tests;

/// <summary>
/// Known-answer checks for text that is not perfectly level. A test that only asks for some regions and some text cannot tell a
/// reader that works from one that garbles every long line tilted by a degree or two (the recognizer once cut each line out as
/// the axis-aligned box around it, which held its neighbours). These render English lines whose words are known, tilt them, add
/// pixel noise, and ask the default reader what it sees. They need the default models in the local cache, so they run at the
/// release gate with the other local-only facts.
/// </summary>
[Trait("Category", "Integration")]
[Trait("Category", "LocalOnly")]
[Trait("Axis", "KnownAnswer")]
public class OcrTiltedTextTests
{
    private static readonly string[] Lines =
    [
        "Please keep your receipt; items can be exchanged within 7 days of purchase.",
        "The library is closed on Mondays and on public holidays throughout the year.",
        "Password must contain at least 8 characters including letters and numbers.",
    ];

    // Measured with the default models, the character error rate (white space left out) is 0.010 for the level page and 0.010 to 0.016 for
    // each tilt. Before a tilted line was straightened, the page tilted by -2 degrees was read at 0.98 (as four letters) and the one tilted
    // by 4 degrees at 1.0 (nothing); the 1.6 degree page was already read at 0.026. The bound leaves room for another platform's font
    // and arithmetic.
    private const double MaxCharacterErrorRate = 0.05;

    [Theory]
    [InlineData(0.0)]
    [InlineData(1.6)]
    [InlineData(-2.0)]
    [InlineData(4.0)]
    public async Task LinesTiltedByADegreeOrTwo_AreReadAsLevelOnesAre(double degrees)
    {
        var png = KnownAnswerPage.Render(Lines, degrees);

        await using var ocr = await LocalOcr.LoadAsync(cancellationToken: TestContext.Current.CancellationToken);
        var result = await ocr.RecognizeAsync(png, TestContext.Current.CancellationToken);

        var heard = result.GetTextWithLayout(lineTolerancePixels: 20);
        var rate = KnownAnswerPage.CharacterErrorRate(string.Concat(Lines), heard);

        rate.Should().BeLessThanOrEqualTo(MaxCharacterErrorRate, $"the page tilted by {degrees}° was read as «{heard}»");
    }
}
