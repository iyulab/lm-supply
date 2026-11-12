using AwesomeAssertions;
using SkiaSharp;
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
        var png = Render(Lines, degrees);

        await using var ocr = await LocalOcr.LoadAsync(cancellationToken: TestContext.Current.CancellationToken);
        var result = await ocr.RecognizeAsync(png, TestContext.Current.CancellationToken);

        var heard = Squash(result.GetTextWithLayout(lineTolerancePixels: 20));
        var said = Squash(string.Concat(Lines));
        var rate = (double)Distance(said, heard) / said.Length;

        rate.Should().BeLessThanOrEqualTo(MaxCharacterErrorRate, $"the page tilted by {degrees}° was read as «{heard}»");
    }

    // White page, three lines of black text in a sans-serif face, rotated about the page center, then pixel noise.
    private static byte[] Render(string[] lines, double degrees)
    {
        const int width = 1100, height = 330;
        using var surface = SKSurface.Create(new SKImageInfo(width, height));
        var canvas = surface.Canvas;
        canvas.Clear(SKColors.White);
        using var typeface = SKTypeface.FromFamilyName("Arial") ?? SKTypeface.Default;
        using var font = new SKFont(typeface, 30);
        using var paint = new SKPaint { Color = new SKColor(20, 20, 20), IsAntialias = true };

        canvas.RotateDegrees((float)degrees, width / 2f, height / 2f);
        for (var i = 0; i < lines.Length; i++)
            canvas.DrawText(lines[i], 40, 70 + i * 90, SKTextAlign.Left, font, paint);

        using var bitmap = SKBitmap.FromImage(surface.Snapshot());
        var random = new Random(7);
        for (var n = 0; n < width * height / 25; n++)
        {
            var (x, y) = (random.Next(width), random.Next(height));
            var delta = random.Next(-45, 46);
            var pixel = bitmap.GetPixel(x, y);
            byte Shift(byte channel) => (byte)Math.Clamp(channel + delta, 0, 255);
            bitmap.SetPixel(x, y, new SKColor(Shift(pixel.Red), Shift(pixel.Green), Shift(pixel.Blue)));
        }

        using var image = SKImage.FromBitmap(bitmap);
        using var data = image.Encode(SKEncodedImageFormat.Png, 100);
        return data.ToArray();
    }

    private static string Squash(string text) => new(text.Where(c => !char.IsWhiteSpace(c)).ToArray());

    private static int Distance(string a, string b)
    {
        var row = Enumerable.Range(0, b.Length + 1).ToArray();
        for (var i = 1; i <= a.Length; i++)
        {
            var previous = row[0];
            row[0] = i;
            for (var j = 1; j <= b.Length; j++)
            {
                var above = row[j];
                row[j] = Math.Min(Math.Min(row[j] + 1, row[j - 1] + 1), previous + (a[i - 1] == b[j - 1] ? 0 : 1));
                previous = above;
            }
        }

        return row[b.Length];
    }
}
