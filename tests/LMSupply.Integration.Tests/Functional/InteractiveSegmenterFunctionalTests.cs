using LMSupply.Segmenter;
using LMSupply.Segmenter.Interactive;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace LMSupply.Integration.Tests.Functional;

/// <summary>
/// The interactive (MobileSAM) segmenter segments what a point is on. The image is a black disc on a white, non-square canvas —
/// non-square so that a resize that ignores the aspect ratio (the encoder expects the longest side scaled to 1024) shows up as a
/// misplaced mask.
/// </summary>
[Trait("Category", "Integration")]
public sealed class InteractiveSegmenterFunctionalTests
{
    private const int Width = 800, Height = 500, CenterX = 520, CenterY = 250, Radius = 120;

    private static byte[] DiscImage()
    {
        using var image = new Image<Rgb24>(Width, Height, new Rgb24(255, 255, 255));
        for (var y = 0; y < Height; y++)
        for (var x = 0; x < Width; x++)
            if (InDisc(x, y))
                image[x, y] = new Rgb24(0, 0, 0);
        using var stream = new MemoryStream();
        image.SaveAsPng(stream);
        return stream.ToArray();
    }

    private static bool InDisc(int x, int y) => (x - CenterX) * (x - CenterX) + (y - CenterY) * (y - CenterY) <= Radius * Radius;

    [Fact]
    public async Task APointOnTheDisc_SegmentsTheDisc()
    {
        await using var segmenter = await LocalSegmenter.LoadInteractiveAsync(cancellationToken: TestContext.Current.CancellationToken);
        await using var session = await segmenter.CreateSessionAsync(DiscImage(), TestContext.Current.CancellationToken);

        var result = await session.SegmentAsync([new PointPrompt(CenterX, CenterY, PointLabel.Foreground)], cancellationToken: TestContext.Current.CancellationToken);

        var mask = result.BestMask;
        int both = 0, either = 0;
        for (var y = 0; y < Height; y++)
        for (var x = 0; x < Width; x++)
        {
            var m = mask.Mask[y, x];
            var d = InDisc(x, y);
            if (m && d) both++;
            if (m || d) either++;
        }

        var iou = (double)both / either;
        Assert.True(iou > 0.8, $"mask/disc IoU {iou:F3} (mask pixels {mask.PixelCount}, disc ≈ {Math.PI * Radius * Radius:F0})");
    }
}
