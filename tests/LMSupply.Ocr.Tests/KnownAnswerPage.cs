using SkiaSharp;

namespace LMSupply.Ocr.Tests;

/// <summary>
/// Pages of text whose words are known, and the character error rate of what a reader made of them. A check that only asks for some
/// regions and some text cannot tell a reader that works from one that returns noise; these make the answer something to compare to.
/// </summary>
internal static class KnownAnswerPage
{
    /// <summary>
    /// A white page with one line of black text per entry in a sans-serif face, rotated about the page center by
    /// <paramref name="degrees"/>, then pixel noise. A line wider than <paramref name="width"/> is cut off at the page edge,
    /// so a page of long lines needs a wider one.
    /// </summary>
    public static byte[] Render(string[] lines, double degrees = 0.0, int width = 1100)
    {
        var height = 90 * lines.Length + 60;
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

    /// <summary>The edit distance between what the page says and what was read, over the length of what it says, white space left out.</summary>
    public static double CharacterErrorRate(string said, string heard)
    {
        var a = Squash(said);
        var b = Squash(heard);
        return (double)Distance(a, b) / a.Length;
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
