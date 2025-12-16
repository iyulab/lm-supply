using FluentAssertions;
using LMSupply.Ocr;

namespace LMSupply.Ocr.Tests;

public class OcrResultTests
{
    [Fact]
    public void OcrResult_WithEmptyRegions_ShouldHaveEmptyFullText()
    {
        // Arrange
        var result = new OcrResult([], 100);

        // Assert
        result.FullText.Should().BeEmpty();
        result.Regions.Should().BeEmpty();
        result.ProcessingTimeMs.Should().Be(100);
    }

    [Fact]
    public void OcrResult_FullText_ShouldConcatenateRegionsSortedByPosition()
    {
        // Arrange
        var regions = new List<TextRegion>
        {
            new("World", 0.9f, new BoundingBox(100, 0, 50, 20)),
            new("Hello", 0.95f, new BoundingBox(0, 0, 50, 20)),
        };
        var result = new OcrResult(regions, 50);

        // Act
        var fullText = result.FullText;

        // Assert - Should be sorted by Y then X
        fullText.Should().Be($"Hello{Environment.NewLine}World");
    }

    [Fact]
    public void OcrResult_GetTextWithLayout_ShouldJoinSameLineRegions()
    {
        // Arrange
        var regions = new List<TextRegion>
        {
            new("Hello", 0.95f, new BoundingBox(0, 0, 50, 20)),
            new("World", 0.9f, new BoundingBox(60, 0, 50, 20)),
            new("Line2", 0.85f, new BoundingBox(0, 30, 50, 20)),
        };
        var result = new OcrResult(regions, 50);

        // Act
        var layoutText = result.GetTextWithLayout(lineTolerancePixels: 10);

        // Assert
        layoutText.Should().Be($"Hello World{Environment.NewLine}Line2");
    }

    [Fact]
    public void BoundingBox_Area_ShouldCalculateCorrectly()
    {
        // Arrange
        var box = new BoundingBox(0, 0, 100, 50);

        // Assert
        box.Area.Should().Be(5000);
    }

    [Fact]
    public void BoundingBox_Center_ShouldCalculateCorrectly()
    {
        // Arrange
        var box = new BoundingBox(10, 20, 100, 50);

        // Assert
        box.Center.X.Should().Be(60);
        box.Center.Y.Should().Be(45);
    }

    [Fact]
    public void BoundingBox_FromCorners_ShouldCreateCorrectBox()
    {
        // Act
        var box = BoundingBox.FromCorners(10, 20, 110, 70);

        // Assert
        box.X.Should().Be(10);
        box.Y.Should().Be(20);
        box.Width.Should().Be(100);
        box.Height.Should().Be(50);
    }

    [Fact]
    public void BoundingBox_FromPolygon_ShouldCreateEnclosingRect()
    {
        // Arrange
        var polygon = new List<Point>
        {
            new(10, 10),
            new(100, 20),
            new(90, 60),
            new(5, 50)
        };

        // Act
        var box = BoundingBox.FromPolygon(polygon);

        // Assert
        box.X.Should().Be(5);
        box.Y.Should().Be(10);
        box.Width.Should().Be(95);
        box.Height.Should().Be(50);
    }

    [Fact]
    public void BoundingBox_FromEmptyPolygon_ShouldReturnZeroBox()
    {
        // Arrange
        var polygon = new List<Point>();

        // Act
        var box = BoundingBox.FromPolygon(polygon);

        // Assert
        box.X.Should().Be(0);
        box.Y.Should().Be(0);
        box.Width.Should().Be(0);
        box.Height.Should().Be(0);
    }

    [Fact]
    public void TextRegion_ShouldStoreAllProperties()
    {
        // Arrange
        var box = new BoundingBox(10, 20, 100, 30);
        var polygon = new List<Point> { new(10, 20), new(110, 20), new(110, 50), new(10, 50) };

        // Act
        var region = new TextRegion("Test", 0.95f, box, polygon);

        // Assert
        region.Text.Should().Be("Test");
        region.Confidence.Should().Be(0.95f);
        region.BoundingBox.Should().Be(box);
        region.Polygon.Should().BeEquivalentTo(polygon);
    }

    [Fact]
    public void DetectedRegion_ShouldStoreAllProperties()
    {
        // Arrange
        var box = new BoundingBox(10, 20, 100, 30);
        var polygon = new List<Point> { new(10, 20), new(110, 20), new(110, 50), new(10, 50) };

        // Act
        var region = new DetectedRegion(box, 0.8f, polygon);

        // Assert
        region.BoundingBox.Should().Be(box);
        region.Confidence.Should().Be(0.8f);
        region.Polygon.Should().BeEquivalentTo(polygon);
    }

    [Fact]
    public void Point_ShouldStoreCoordinates()
    {
        // Act
        var point = new Point(100, 200);

        // Assert
        point.X.Should().Be(100);
        point.Y.Should().Be(200);
    }
}
