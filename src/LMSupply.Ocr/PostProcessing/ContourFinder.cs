namespace LMSupply.Ocr.PostProcessing;

/// <summary>
/// Pure C# implementation of contour finding algorithm.
/// Uses Suzuki-Abe algorithm variant for border following.
/// </summary>
internal static class ContourFinder
{
    /// <summary>
    /// Finds all contours in a binary image.
    /// </summary>
    /// <param name="binaryMap">Binary image where true = foreground.</param>
    /// <returns>List of contours, each represented as a list of points.</returns>
    public static List<List<PointF>> FindContours(bool[,] binaryMap)
    {
        var height = binaryMap.GetLength(0);
        var width = binaryMap.GetLength(1);

        // Create labeled image for tracking
        var labels = new int[height, width];
        var contours = new List<List<PointF>>();
        var labelCounter = 1;

        // 8-connectivity directions (clockwise from right)
        int[] dx = [1, 1, 0, -1, -1, -1, 0, 1];
        int[] dy = [0, 1, 1, 1, 0, -1, -1, -1];

        // Scan the image
        for (var y = 0; y < height; y++)
        {
            for (var x = 0; x < width; x++)
            {
                // Check for outer border starting point
                if (binaryMap[y, x] && labels[y, x] == 0 && (x == 0 || !binaryMap[y, x - 1]))
                {
                    var contour = TraceContour(binaryMap, labels, x, y, labelCounter, dx, dy);
                    if (contour.Count >= 3)
                    {
                        contours.Add(SimplifyContour(contour));
                    }
                    labelCounter++;
                }
            }
        }

        // Sort by area (largest first)
        return contours.OrderByDescending(c => CalculateContourArea(c)).ToList();
    }

    private static List<PointF> TraceContour(bool[,] binaryMap, int[,] labels, int startX, int startY, int label, int[] dx, int[] dy)
    {
        var height = binaryMap.GetLength(0);
        var width = binaryMap.GetLength(1);

        var contour = new List<PointF>();
        var x = startX;
        var y = startY;
        var dir = 0; // Start direction (search from right)

        do
        {
            contour.Add(new PointF(x, y));
            labels[y, x] = label;

            // Search for next point in clockwise direction
            var found = false;
            var startDir = (dir + 5) % 8; // Start from the opposite direction + 1

            for (var i = 0; i < 8; i++)
            {
                var searchDir = (startDir + i) % 8;
                var nx = x + dx[searchDir];
                var ny = y + dy[searchDir];

                if (nx >= 0 && nx < width && ny >= 0 && ny < height && binaryMap[ny, nx])
                {
                    x = nx;
                    y = ny;
                    dir = searchDir;
                    found = true;
                    break;
                }
            }

            if (!found)
                break;

        } while (x != startX || y != startY);

        return contour;
    }

    /// <summary>
    /// Simplifies a contour using the Douglas-Peucker algorithm.
    /// </summary>
    private static List<PointF> SimplifyContour(List<PointF> contour, float epsilon = 2.0f)
    {
        if (contour.Count < 3)
            return contour;

        // Find the point with maximum distance from the line between first and last
        var dmax = 0f;
        var index = 0;
        var end = contour.Count - 1;

        for (var i = 1; i < end; i++)
        {
            var d = PerpendicularDistance(contour[i], contour[0], contour[end]);
            if (d > dmax)
            {
                index = i;
                dmax = d;
            }
        }

        // If max distance is greater than epsilon, recursively simplify
        if (dmax > epsilon)
        {
            var rec1 = SimplifyContour(contour.Take(index + 1).ToList(), epsilon);
            var rec2 = SimplifyContour(contour.Skip(index).ToList(), epsilon);

            // Build the result list
            var result = rec1.Take(rec1.Count - 1).ToList();
            result.AddRange(rec2);
            return result;
        }

        // Return just the endpoints
        return [contour[0], contour[end]];
    }

    private static float PerpendicularDistance(PointF point, PointF lineStart, PointF lineEnd)
    {
        var dx = lineEnd.X - lineStart.X;
        var dy = lineEnd.Y - lineStart.Y;

        // Normalize
        var mag = MathF.Sqrt(dx * dx + dy * dy);
        if (mag < 1e-6f)
            return MathF.Sqrt((point.X - lineStart.X) * (point.X - lineStart.X) +
                             (point.Y - lineStart.Y) * (point.Y - lineStart.Y));

        dx /= mag;
        dy /= mag;

        // Vector from line start to point
        var pvx = point.X - lineStart.X;
        var pvy = point.Y - lineStart.Y;

        // Get perpendicular distance
        var dist = MathF.Abs(pvx * dy - pvy * dx);
        return dist;
    }

    private static float CalculateContourArea(List<PointF> contour)
    {
        if (contour.Count < 3)
            return 0;

        float area = 0;
        var j = contour.Count - 1;

        for (var i = 0; i < contour.Count; i++)
        {
            area += (contour[j].X + contour[i].X) * (contour[j].Y - contour[i].Y);
            j = i;
        }

        return Math.Abs(area / 2);
    }
}
