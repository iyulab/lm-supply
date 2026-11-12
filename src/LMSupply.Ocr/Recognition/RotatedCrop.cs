using LMSupply.Vision;

namespace LMSupply.Ocr.Recognition;

/// <summary>
/// Cuts a slightly rotated line of text out of an image as the straight strip the recognizer reads. A detected line is a
/// rotated rectangle; the axis-aligned box around it grows by the line's length times the tangent of its angle, so on a
/// line of a hundred characters tilted by a degree and a half it holds a good part of the lines above and below.
/// </summary>
internal static class RotatedCrop
{
    // Below this tilt the axis-aligned crop is as good as a straightened one, and keeps the pixels exactly as they are.
    private const double MinTiltRadians = 0.75 * Math.PI / 180;

    // Beyond this the line is not slightly rotated but turned (vertical text); it is left to the plain crop.
    private const double MaxTiltRadians = 45 * Math.PI / 180;

    /// <summary>
    /// The strip along the rectangle <paramref name="polygon"/> (its four corners, in any order), straightened so that the
    /// text runs left to right, or null when the plain crop of the polygon's box is the right one: no four corners, a
    /// rectangle that is already level, or one turned too far to be a tilt.
    /// </summary>
    public static RgbImage? TryExtract(RgbImage image, IReadOnlyList<Point>? polygon)
    {
        if (polygon is not { Count: 4 })
            return null;

        var (topLeft, topRight, bottomRight, bottomLeft) = Order(polygon);

        var tilt = Math.Atan2(topRight.Y - topLeft.Y, topRight.X - topLeft.X);
        var absTilt = Math.Abs(tilt);
        if (absTilt < MinTiltRadians || absTilt > MaxTiltRadians)
            return null;

        var width = (int)Math.Round(Math.Max(Distance(topLeft, topRight), Distance(bottomLeft, bottomRight)));
        var height = (int)Math.Round(Math.Max(Distance(topLeft, bottomLeft), Distance(topRight, bottomRight)));
        if (width < 2 || height < 2)
            return null;

        // A strip much taller than it is wide is a vertical line, which the recognizer reads as it is cut, not turned.
        if (height >= 1.5 * width)
            return null;

        // The rectangle is a parallelogram of two edge vectors from the top-left corner; each output pixel is the source
        // point at its place in that parallelogram.
        var edgeX = (topRight.X - topLeft.X, topRight.Y - topLeft.Y);
        var edgeY = (bottomLeft.X - topLeft.X, bottomLeft.Y - topLeft.Y);

        var strip = new RgbImage(width, height);
        for (var y = 0; y < height; y++)
        {
            var row = strip.GetRow(y);
            var v = (y + 0.5) / height;
            for (var x = 0; x < width; x++)
            {
                var u = (x + 0.5) / width;
                var sx = topLeft.X + u * edgeX.Item1 + v * edgeY.Item1 - 0.5;
                var sy = topLeft.Y + u * edgeX.Item2 + v * edgeY.Item2 - 0.5;
                Sample(image, sx, sy, row.Slice(x * 3, 3));
            }
        }

        return strip;
    }

    // Top-left, top-right, bottom-right, bottom-left, as Paddle's own crop orders them: the two leftmost corners are the
    // left side (the higher is the top-left), the two others the right side.
    private static (PointD TopLeft, PointD TopRight, PointD BottomRight, PointD BottomLeft) Order(IReadOnlyList<Point> polygon)
    {
        var byX = polygon.OrderBy(p => p.X).ThenBy(p => p.Y).ToArray();
        var left = byX.Take(2).OrderBy(p => p.Y).ToArray();
        var right = byX.Skip(2).OrderBy(p => p.Y).ToArray();
        return (new PointD(left[0]), new PointD(right[0]), new PointD(right[1]), new PointD(left[1]));
    }

    private static double Distance(PointD a, PointD b) => Math.Sqrt((a.X - b.X) * (a.X - b.X) + (a.Y - b.Y) * (a.Y - b.Y));

    // Bilinear sample at a fractional position; a position outside the image reads its nearest edge pixel.
    private static void Sample(RgbImage image, double x, double y, Span<byte> destination)
    {
        var x0 = (int)Math.Floor(x);
        var y0 = (int)Math.Floor(y);
        var fx = x - x0;
        var fy = y - y0;

        for (var channel = 0; channel < 3; channel++)
        {
            var p00 = Pixel(image, x0, y0, channel);
            var p10 = Pixel(image, x0 + 1, y0, channel);
            var p01 = Pixel(image, x0, y0 + 1, channel);
            var p11 = Pixel(image, x0 + 1, y0 + 1, channel);
            var value = p00 * (1 - fx) * (1 - fy) + p10 * fx * (1 - fy) + p01 * (1 - fx) * fy + p11 * fx * fy;
            destination[channel] = (byte)Math.Clamp((int)Math.Round(value), 0, 255);
        }
    }

    private static int Pixel(RgbImage image, int x, int y, int channel)
    {
        var cx = Math.Clamp(x, 0, image.Width - 1);
        var cy = Math.Clamp(y, 0, image.Height - 1);
        return image.GetRow(cy)[cx * 3 + channel];
    }

    private readonly record struct PointD(double X, double Y)
    {
        public PointD(Point point) : this(point.X, point.Y)
        {
        }
    }
}
