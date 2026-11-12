using AwesomeAssertions;
using LMSupply.Ocr.Recognition;
using LMSupply.Vision;

namespace LMSupply.Ocr.Tests.Recognition;

/// <summary>
/// A detected line is a rotated rectangle. The recognizer reads the straight strip along it, not the box around it, which on
/// a long tilted line holds parts of its neighbours.
/// </summary>
public class RotatedCropTests
{
    private const double CenterX = 200, CenterY = 200;
    private const int StripeWidth = 6;

    // A white image with black stripes perpendicular to a baseline at the given angle through the center: pixels along the
    // baseline alternate, pixels across it do not. A straightened strip shows the same stripes on every row.
    private static RgbImage StripedImage(double degrees)
    {
        var image = RgbImage.Filled(400, 400, 255, 255, 255);
        var (cos, sin) = (Math.Cos(degrees * Math.PI / 180), Math.Sin(degrees * Math.PI / 180));
        for (var y = 0; y < 400; y++)
        {
            for (var x = 0; x < 400; x++)
            {
                var along = (x - CenterX) * cos + (y - CenterY) * sin;
                var value = (byte)((int)Math.Floor(along / StripeWidth) % 2 == 0 ? 0 : 255);
                image[x, y] = (value, value, value);
            }
        }

        return image;
    }

    // The four corners of a width x height rectangle rotated by the angle about the center, in a scrambled order.
    private static IReadOnlyList<Point> Corners(double degrees, double width, double height)
    {
        var (cos, sin) = (Math.Cos(degrees * Math.PI / 180), Math.Sin(degrees * Math.PI / 180));
        Point At(double dx, double dy) => new((int)Math.Round(CenterX + dx * cos - dy * sin), (int)Math.Round(CenterY + dx * sin + dy * cos));
        var (w, h) = (width / 2, height / 2);
        return [At(w, h), At(-w, -h), At(-w, h), At(w, -h)];
    }

    private static double RowDisagreement(RgbImage strip, int row, int referenceRow)
    {
        var a = strip.GetRow(row);
        var b = strip.GetRow(referenceRow);
        var different = 0;
        for (var x = 0; x < strip.Width; x++)
        {
            if ((a[x * 3] >= 128) != (b[x * 3] >= 128))
                different++;
        }

        return (double)different / strip.Width;
    }

    [Theory]
    [InlineData(10)]
    [InlineData(-6.5)]
    [InlineData(1.6)]
    public void ATiltedRectangle_IsStraightened_SoEveryRowOfTheStripReadsTheSame(double degrees)
    {
        var strip = RotatedCrop.TryExtract(StripedImage(degrees), Corners(degrees, 160, 30));

        strip.Should().NotBeNull();
        strip!.Width.Should().BeInRange(159, 161);
        strip.Height.Should().BeInRange(29, 31);
        // Corners are whole pixels, so the strip can be sheared by up to about a pixel across its height: a few stripe edges
        // move by that much, hence a bound of a fifth, against more than 0.3 for the box around the same line.
        for (var row = 0; row < strip.Height; row++)
            RowDisagreement(strip, row, strip.Height / 2).Should().BeLessThan(0.2, $"row {row} of the strip should show the stripes where the middle row does");
    }

    [Fact]
    public void TheBoxAroundATiltedLine_IsNotStraight_WhichIsWhyItIsNotUsed()
    {
        // The same stripes cut as the plain box around the rectangle: the rows disagree across the box.
        var image = StripedImage(10);
        var corners = Corners(10, 160, 30);
        var (left, top) = (corners.Min(p => p.X), corners.Min(p => p.Y));
        var box = image.Crop(left, top, corners.Max(p => p.X) - left, corners.Max(p => p.Y) - top);

        RowDisagreement(box, 0, box.Height - 1).Should().BeGreaterThan(0.3);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(0.3)]
    [InlineData(-0.5)]
    public void ALevelRectangle_IsLeftToThePlainCrop(double degrees)
    {
        RotatedCrop.TryExtract(StripedImage(degrees), Corners(degrees, 160, 30)).Should().BeNull();
    }

    [Theory]
    [InlineData(60)]
    [InlineData(-80)]
    public void ARectangleTurnedTooFarToBeATilt_IsLeftToThePlainCrop(double degrees)
    {
        RotatedCrop.TryExtract(StripedImage(degrees), Corners(degrees, 160, 30)).Should().BeNull();
    }

    [Fact]
    public void ARegionWithoutFourCorners_IsLeftToThePlainCrop()
    {
        var image = StripedImage(10);

        RotatedCrop.TryExtract(image, null).Should().BeNull();
        RotatedCrop.TryExtract(image, []).Should().BeNull();
        RotatedCrop.TryExtract(image, [new Point(1, 1), new Point(50, 4), new Point(30, 40)]).Should().BeNull();
    }

    [Fact]
    public void ARectangleThatReachesOutsideTheImage_ReadsItsNearestEdgePixels_AndDoesNotThrow()
    {
        var corners = new List<Point> { new(-20, -10), new(120, 0), new(118, 30), new(-22, 20) };

        var strip = RotatedCrop.TryExtract(RgbImage.Filled(100, 100, 7, 8, 9), corners);

        strip.Should().NotBeNull();
        strip!.GetRow(0)[0].Should().Be(7);
    }
}
