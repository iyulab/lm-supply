using AwesomeAssertions;
using LMSupply.Ocr.PostProcessing;

namespace LMSupply.Ocr.Tests.PostProcessing;

public class ContourFinderTests
{
    // --- FindContours: basic cases ---

    [Fact]
    public void FindContours_EmptyMap_ReturnsEmpty()
    {
        var map = new bool[5, 5]; // All false

        var contours = ContourFinder.FindContours(map);

        contours.Should().BeEmpty();
    }

    [Fact]
    public void FindContours_SinglePixel_ReturnsEmpty()
    {
        // Single pixel doesn't form a 3-point contour
        var map = new bool[5, 5];
        map[2, 2] = true;

        var contours = ContourFinder.FindContours(map);

        // A single isolated pixel has <3 points in trace, so it's filtered out
        contours.Should().BeEmpty();
    }

    [Fact]
    public void FindContours_LargerRectangle_FindsContour()
    {
        // 6x8 filled rectangle — large enough to survive Douglas-Peucker simplification
        var map = new bool[12, 14];
        for (var y = 2; y <= 7; y++)
            for (var x = 3; x <= 10; x++)
                map[y, x] = true;

        var contours = ContourFinder.FindContours(map);

        contours.Should().NotBeEmpty();
        contours[0].Count.Should().BeGreaterThanOrEqualTo(3);
    }

    [Fact]
    public void FindContours_TwoSeparateRegions_FindsTwoContours()
    {
        var map = new bool[10, 20];
        // Region 1: left side (3x3)
        for (var y = 2; y <= 4; y++)
            for (var x = 1; x <= 3; x++)
                map[y, x] = true;
        // Region 2: right side (3x3)
        for (var y = 2; y <= 4; y++)
            for (var x = 10; x <= 12; x++)
                map[y, x] = true;

        var contours = ContourFinder.FindContours(map);

        contours.Should().HaveCount(2);
    }

    [Fact]
    public void FindContours_SortedByAreaDescending()
    {
        var map = new bool[20, 20];
        // Small region (2x2)
        for (var y = 1; y <= 2; y++)
            for (var x = 1; x <= 2; x++)
                map[y, x] = true;
        // Large region (5x5)
        for (var y = 10; y <= 14; y++)
            for (var x = 10; x <= 14; x++)
                map[y, x] = true;

        var contours = ContourFinder.FindContours(map);

        contours.Should().HaveCountGreaterThanOrEqualTo(2);
        // First contour should be the larger one
        var area0 = CalculateArea(contours[0]);
        var area1 = CalculateArea(contours[1]);
        area0.Should().BeGreaterThanOrEqualTo(area1);
    }

    [Fact]
    public void FindContours_FullMap_ReturnsContour()
    {
        // Entire map is foreground
        var map = new bool[5, 5];
        for (var y = 0; y < 5; y++)
            for (var x = 0; x < 5; x++)
                map[y, x] = true;

        var contours = ContourFinder.FindContours(map);

        // Should detect at least the outer border
        contours.Should().NotBeEmpty();
    }

    [Fact]
    public void FindContours_HorizontalLine_ReturnsContour()
    {
        var map = new bool[5, 10];
        // Horizontal line of pixels
        for (var x = 1; x <= 8; x++)
            map[2, x] = true;

        var contours = ContourFinder.FindContours(map);

        // A single row of pixels should trace a contour with >= 3 points
        contours.Should().NotBeEmpty();
    }

    [Fact]
    public void FindContours_LShapedRegion_FindsOneContour()
    {
        var map = new bool[10, 10];
        // Vertical bar
        for (var y = 1; y <= 6; y++)
            map[y, 1] = true;
        // Horizontal bar (connected to bottom of vertical)
        for (var x = 1; x <= 5; x++)
            map[6, x] = true;

        var contours = ContourFinder.FindContours(map);

        // Connected region -> single contour
        contours.Should().HaveCount(1);
    }

    // --- ContourFinder output characteristics ---

    [Fact]
    public void FindContours_ContourPointsAreWithinMapBounds()
    {
        var map = new bool[8, 12];
        for (var y = 2; y <= 5; y++)
            for (var x = 3; x <= 9; x++)
                map[y, x] = true;

        var contours = ContourFinder.FindContours(map);

        contours.Should().NotBeEmpty();
        foreach (var contour in contours)
        {
            foreach (var point in contour)
            {
                point.X.Should().BeGreaterThanOrEqualTo(0);
                point.X.Should().BeLessThan(12);
                point.Y.Should().BeGreaterThanOrEqualTo(0);
                point.Y.Should().BeLessThan(8);
            }
        }
    }

    // Helper to calculate approximate contour area using shoelace formula
    private static float CalculateArea(List<PointF> contour)
    {
        if (contour.Count < 3) return 0;
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
