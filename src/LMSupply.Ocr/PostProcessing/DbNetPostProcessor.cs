namespace LMSupply.Ocr.PostProcessing;

/// <summary>
/// Post-processor for DBNet detection model outputs.
/// Converts probability maps to text region bounding boxes.
/// </summary>
internal sealed class DbNetPostProcessor
{
    private readonly float _binarizationThreshold;
    private readonly float _boxThreshold;
    private readonly int _maxCandidates;
    private readonly float _unclipRatio;
    private readonly int _minBoxArea;

    public DbNetPostProcessor(OcrOptions options)
    {
        _binarizationThreshold = options.BinarizationThreshold;
        _boxThreshold = options.DetectionThreshold;
        _maxCandidates = options.MaxCandidates;
        _unclipRatio = options.UnclipRatio;
        _minBoxArea = options.MinBoxArea;
    }

    /// <summary>
    /// Processes the probability map output from DBNet to extract text regions.
    /// </summary>
    /// <param name="probabilityMap">2D probability map from model output (H x W).</param>
    /// <param name="originalWidth">Original image width.</param>
    /// <param name="originalHeight">Original image height.</param>
    /// <returns>List of detected text regions.</returns>
    public List<DetectedRegion> Process(float[,] probabilityMap, int originalWidth, int originalHeight)
    {
        var mapHeight = probabilityMap.GetLength(0);
        var mapWidth = probabilityMap.GetLength(1);

        // Binarize the probability map
        var binaryMap = Binarize(probabilityMap, _binarizationThreshold);

        // Find contours in the binary map
        var contours = ContourFinder.FindContours(binaryMap);

        // Process contours to get bounding boxes
        var regions = new List<DetectedRegion>();
        var scaleX = (float)originalWidth / mapWidth;
        var scaleY = (float)originalHeight / mapHeight;

        foreach (var contour in contours.Take(_maxCandidates))
        {
            // Get minimum area rectangle
            var box = GetMinAreaRect(contour);
            if (box.Count < 4) continue;

            // Calculate average score in the region
            var score = CalculateBoxScore(probabilityMap, box);
            if (score < _boxThreshold) continue;

            // Unclip (expand) the polygon
            var unclippedBox = UnclipPolygon(box, _unclipRatio);
            if (unclippedBox.Count < 4) continue;

            // Get minimum area rectangle again after unclipping
            var finalBox = GetMinAreaRect(unclippedBox);

            // Scale to original image coordinates
            var scaledPolygon = finalBox
                .Select(p => new Point(
                    (int)Math.Round(p.X * scaleX),
                    (int)Math.Round(p.Y * scaleY)))
                .ToList();

            // Create bounding box
            var boundingBox = BoundingBox.FromPolygon(scaledPolygon);

            // Filter by minimum area
            if (boundingBox.Area < _minBoxArea) continue;

            regions.Add(new DetectedRegion(boundingBox, score, scaledPolygon));
        }

        // Sort by position (top-to-bottom, left-to-right)
        return regions
            .OrderBy(r => r.BoundingBox.Y)
            .ThenBy(r => r.BoundingBox.X)
            .ToList();
    }

    private static bool[,] Binarize(float[,] probabilityMap, float threshold)
    {
        var height = probabilityMap.GetLength(0);
        var width = probabilityMap.GetLength(1);
        var binary = new bool[height, width];

        for (var y = 0; y < height; y++)
        {
            for (var x = 0; x < width; x++)
            {
                binary[y, x] = probabilityMap[y, x] > threshold;
            }
        }

        return binary;
    }

    private static float CalculateBoxScore(float[,] probabilityMap, List<PointF> box)
    {
        var height = probabilityMap.GetLength(0);
        var width = probabilityMap.GetLength(1);

        // Get bounding rect of the box
        var minX = (int)Math.Floor(box.Min(p => p.X));
        var maxX = (int)Math.Ceiling(box.Max(p => p.X));
        var minY = (int)Math.Floor(box.Min(p => p.Y));
        var maxY = (int)Math.Ceiling(box.Max(p => p.Y));

        minX = Math.Max(0, minX);
        maxX = Math.Min(width - 1, maxX);
        minY = Math.Max(0, minY);
        maxY = Math.Min(height - 1, maxY);

        // Calculate mean score inside the polygon
        float sum = 0;
        var count = 0;

        for (var y = minY; y <= maxY; y++)
        {
            for (var x = minX; x <= maxX; x++)
            {
                if (IsPointInPolygon(new PointF(x, y), box))
                {
                    sum += probabilityMap[y, x];
                    count++;
                }
            }
        }

        return count > 0 ? sum / count : 0;
    }

    private static bool IsPointInPolygon(PointF point, List<PointF> polygon)
    {
        var inside = false;
        var j = polygon.Count - 1;

        for (var i = 0; i < polygon.Count; i++)
        {
            if ((polygon[i].Y > point.Y) != (polygon[j].Y > point.Y) &&
                point.X < (polygon[j].X - polygon[i].X) * (point.Y - polygon[i].Y) / (polygon[j].Y - polygon[i].Y) + polygon[i].X)
            {
                inside = !inside;
            }
            j = i;
        }

        return inside;
    }

    private static List<PointF> GetMinAreaRect(IEnumerable<PointF> points)
    {
        var pointList = points.ToList();
        if (pointList.Count < 3)
            return pointList;

        // Simple approach: get convex hull then find minimum bounding rectangle
        var hull = ConvexHull(pointList);
        if (hull.Count < 3)
            return hull;

        // Find the minimum area bounding rectangle using rotating calipers
        return FindMinAreaRect(hull);
    }

    private static List<PointF> ConvexHull(List<PointF> points)
    {
        if (points.Count < 3)
            return [.. points];

        // Sort points by x, then by y
        var sorted = points.OrderBy(p => p.X).ThenBy(p => p.Y).ToList();

        var hull = new List<PointF>();

        // Build lower hull
        foreach (var p in sorted)
        {
            while (hull.Count >= 2 && Cross(hull[^2], hull[^1], p) <= 0)
                hull.RemoveAt(hull.Count - 1);
            hull.Add(p);
        }

        // Build upper hull
        var lowerCount = hull.Count;
        for (var i = sorted.Count - 2; i >= 0; i--)
        {
            while (hull.Count > lowerCount && Cross(hull[^2], hull[^1], sorted[i]) <= 0)
                hull.RemoveAt(hull.Count - 1);
            hull.Add(sorted[i]);
        }

        // Remove last point (same as first)
        hull.RemoveAt(hull.Count - 1);
        return hull;
    }

    private static float Cross(PointF o, PointF a, PointF b)
    {
        return (a.X - o.X) * (b.Y - o.Y) - (a.Y - o.Y) * (b.X - o.X);
    }

    private static List<PointF> FindMinAreaRect(List<PointF> hull)
    {
        if (hull.Count < 3)
            return hull;

        var minArea = float.MaxValue;
        var minRect = new List<PointF>();

        for (var i = 0; i < hull.Count; i++)
        {
            // Get edge vector
            var p1 = hull[i];
            var p2 = hull[(i + 1) % hull.Count];

            var edgeX = p2.X - p1.X;
            var edgeY = p2.Y - p1.Y;
            var edgeLen = MathF.Sqrt(edgeX * edgeX + edgeY * edgeY);

            if (edgeLen < 1e-6f) continue;

            // Normalize edge vector
            edgeX /= edgeLen;
            edgeY /= edgeLen;

            // Project all points onto this edge and perpendicular
            float minProj = float.MaxValue, maxProj = float.MinValue;
            float minPerp = float.MaxValue, maxPerp = float.MinValue;

            foreach (var p in hull)
            {
                var dx = p.X - p1.X;
                var dy = p.Y - p1.Y;
                var proj = dx * edgeX + dy * edgeY;
                var perp = -dx * edgeY + dy * edgeX;

                minProj = Math.Min(minProj, proj);
                maxProj = Math.Max(maxProj, proj);
                minPerp = Math.Min(minPerp, perp);
                maxPerp = Math.Max(maxPerp, perp);
            }

            var area = (maxProj - minProj) * (maxPerp - minPerp);
            if (area < minArea)
            {
                minArea = area;

                // Calculate rectangle corners
                minRect =
                [
                    new PointF(p1.X + minProj * edgeX - minPerp * edgeY, p1.Y + minProj * edgeY + minPerp * edgeX),
                    new PointF(p1.X + maxProj * edgeX - minPerp * edgeY, p1.Y + maxProj * edgeY + minPerp * edgeX),
                    new PointF(p1.X + maxProj * edgeX - maxPerp * edgeY, p1.Y + maxProj * edgeY + maxPerp * edgeX),
                    new PointF(p1.X + minProj * edgeX - maxPerp * edgeY, p1.Y + minProj * edgeY + maxPerp * edgeX)
                ];
            }
        }

        return minRect;
    }

    private static List<PointF> UnclipPolygon(List<PointF> polygon, float unclipRatio)
    {
        if (polygon.Count < 3)
            return polygon;

        // Calculate perimeter and area
        var area = CalculatePolygonArea(polygon);
        var perimeter = CalculatePolygonPerimeter(polygon);

        if (perimeter < 1e-6f)
            return polygon;

        // Calculate offset distance
        var distance = area * unclipRatio / perimeter;

        // Offset the polygon outward
        return OffsetPolygon(polygon, distance);
    }

    private static float CalculatePolygonArea(List<PointF> polygon)
    {
        float area = 0;
        var j = polygon.Count - 1;

        for (var i = 0; i < polygon.Count; i++)
        {
            area += (polygon[j].X + polygon[i].X) * (polygon[j].Y - polygon[i].Y);
            j = i;
        }

        return Math.Abs(area / 2);
    }

    private static float CalculatePolygonPerimeter(List<PointF> polygon)
    {
        float perimeter = 0;

        for (var i = 0; i < polygon.Count; i++)
        {
            var p1 = polygon[i];
            var p2 = polygon[(i + 1) % polygon.Count];
            perimeter += MathF.Sqrt((p2.X - p1.X) * (p2.X - p1.X) + (p2.Y - p1.Y) * (p2.Y - p1.Y));
        }

        return perimeter;
    }

    private static List<PointF> OffsetPolygon(List<PointF> polygon, float distance)
    {
        if (polygon.Count < 3)
            return polygon;

        var result = new List<PointF>();

        for (var i = 0; i < polygon.Count; i++)
        {
            var prev = polygon[(i - 1 + polygon.Count) % polygon.Count];
            var curr = polygon[i];
            var next = polygon[(i + 1) % polygon.Count];

            // Calculate edge vectors
            var dx1 = curr.X - prev.X;
            var dy1 = curr.Y - prev.Y;
            var dx2 = next.X - curr.X;
            var dy2 = next.Y - curr.Y;

            // Normalize
            var len1 = MathF.Sqrt(dx1 * dx1 + dy1 * dy1);
            var len2 = MathF.Sqrt(dx2 * dx2 + dy2 * dy2);

            if (len1 < 1e-6f || len2 < 1e-6f)
            {
                result.Add(curr);
                continue;
            }

            dx1 /= len1;
            dy1 /= len1;
            dx2 /= len2;
            dy2 /= len2;

            // Calculate normal vectors (pointing outward for CCW polygon)
            var nx1 = -dy1;
            var ny1 = dx1;
            var nx2 = -dy2;
            var ny2 = dx2;

            // Average normal for corner
            var nx = nx1 + nx2;
            var ny = ny1 + ny2;
            var nlen = MathF.Sqrt(nx * nx + ny * ny);

            if (nlen < 1e-6f)
            {
                result.Add(new PointF(curr.X + nx1 * distance, curr.Y + ny1 * distance));
            }
            else
            {
                // Scale by the miter factor
                var miter = 1.0f / (1.0f + nx1 * nx2 + ny1 * ny2);
                miter = Math.Min(miter, 4.0f); // Limit miter factor
                result.Add(new PointF(curr.X + nx / nlen * distance * miter, curr.Y + ny / nlen * distance * miter));
            }
        }

        return result;
    }
}

/// <summary>
/// A 2D point with float coordinates for internal processing.
/// </summary>
internal record struct PointF(float X, float Y);
