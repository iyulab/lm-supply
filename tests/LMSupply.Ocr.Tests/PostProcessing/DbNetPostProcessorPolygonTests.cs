using LMSupply.Ocr.PostProcessing;
using Xunit;

namespace LMSupply.Ocr.Tests.PostProcessing;

/// <summary>
/// <see cref="OcrOptions.UsePolygon"/> decides whether a region carries its polygon; the bounding box is there either way.
/// </summary>
public class DbNetPostProcessorPolygonTests
{
    // A 100x100 probability map with one solid text-like block.
    private static float[,] MapWithOneBlock()
    {
        var map = new float[100, 100];
        for (var y = 40; y < 60; y++)
            for (var x = 20; x < 80; x++)
                map[y, x] = 0.95f;
        return map;
    }

    [Fact]
    public void UsePolygonTrue_RegionsCarryTheirPolygon()
    {
        var regions = new DbNetPostProcessor(new OcrOptions { UsePolygon = true }).Process(MapWithOneBlock(), 100, 100);

        var region = Assert.Single(regions);
        Assert.NotNull(region.Polygon);
        Assert.True(region.Polygon!.Count >= 4);
        Assert.True(region.BoundingBox.Area > 0);
    }

    [Fact]
    public void UsePolygonFalse_RegionsCarryOnlyTheBoundingBox()
    {
        var withPolygon = Assert.Single(new DbNetPostProcessor(new OcrOptions { UsePolygon = true }).Process(MapWithOneBlock(), 100, 100));
        var region = Assert.Single(new DbNetPostProcessor(new OcrOptions { UsePolygon = false }).Process(MapWithOneBlock(), 100, 100));

        Assert.Null(region.Polygon);
        Assert.Equal(withPolygon.BoundingBox, region.BoundingBox);
        Assert.Equal(withPolygon.Confidence, region.Confidence);
    }
}
