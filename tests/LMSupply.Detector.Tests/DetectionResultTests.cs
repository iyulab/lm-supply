using FluentAssertions;

namespace LMSupply.Detector.Tests;

public class DetectionResultTests
{
    [Fact]
    public void DetectionResult_ShouldStoreAllProperties()
    {
        var box = new BoundingBox(10, 20, 100, 200);
        var result = new DetectionResult(
            ClassId: 0,
            Label: "person",
            Confidence: 0.95f,
            Box: box);

        result.ClassId.Should().Be(0);
        result.Label.Should().Be("person");
        result.Confidence.Should().Be(0.95f);
        result.Box.Should().Be(box);
    }

    [Fact]
    public void DetectionResult_EqualResults_ShouldBeEqual()
    {
        var box = new BoundingBox(10, 20, 100, 200);
        var result1 = new DetectionResult(0, "person", 0.95f, box);
        var result2 = new DetectionResult(0, "person", 0.95f, box);

        result1.Should().Be(result2);
    }
}

public class BoundingBoxTests
{
    [Fact]
    public void BoundingBox_ShouldStoreCoordinates()
    {
        var box = new BoundingBox(10, 20, 100, 200);

        box.X1.Should().Be(10);
        box.Y1.Should().Be(20);
        box.X2.Should().Be(100);
        box.Y2.Should().Be(200);
    }

    [Fact]
    public void Width_ShouldReturnCorrectValue()
    {
        var box = new BoundingBox(10, 20, 110, 200);

        box.Width.Should().Be(100);
    }

    [Fact]
    public void Height_ShouldReturnCorrectValue()
    {
        var box = new BoundingBox(10, 20, 100, 220);

        box.Height.Should().Be(200);
    }

    [Fact]
    public void Area_ShouldReturnCorrectValue()
    {
        var box = new BoundingBox(0, 0, 100, 200);

        box.Area.Should().Be(20000);
    }

    [Fact]
    public void FromCenterSize_ShouldCreateCorrectBox()
    {
        var box = BoundingBox.FromCenterSize(50, 100, 100, 200);

        box.X1.Should().Be(0);
        box.Y1.Should().Be(0);
        box.X2.Should().Be(100);
        box.Y2.Should().Be(200);
    }

    [Fact]
    public void IoU_IdenticalBoxes_ShouldReturnOne()
    {
        var box1 = new BoundingBox(0, 0, 100, 100);
        var box2 = new BoundingBox(0, 0, 100, 100);

        box1.IoU(box2).Should().Be(1.0f);
    }

    [Fact]
    public void IoU_NoOverlap_ShouldReturnZero()
    {
        var box1 = new BoundingBox(0, 0, 100, 100);
        var box2 = new BoundingBox(200, 200, 300, 300);

        box1.IoU(box2).Should().Be(0.0f);
    }

    [Fact]
    public void IoU_PartialOverlap_ShouldReturnCorrectValue()
    {
        var box1 = new BoundingBox(0, 0, 100, 100);
        var box2 = new BoundingBox(50, 0, 150, 100);

        // Intersection: 50x100 = 5000
        // Union: 10000 + 10000 - 5000 = 15000
        // IoU: 5000 / 15000 = 0.333...
        box1.IoU(box2).Should().BeApproximately(1f / 3f, 0.001f);
    }

    [Fact]
    public void Clamp_WithinBounds_ShouldNotChange()
    {
        var box = new BoundingBox(10, 20, 100, 200);
        var clamped = box.Clamp(500, 500);

        clamped.Should().Be(box);
    }

    [Fact]
    public void Clamp_ExceedsBounds_ShouldClamp()
    {
        var box = new BoundingBox(-10, -20, 600, 700);
        var clamped = box.Clamp(500, 500);

        clamped.X1.Should().Be(0);
        clamped.Y1.Should().Be(0);
        clamped.X2.Should().Be(500);
        clamped.Y2.Should().Be(500);
    }

    [Fact]
    public void Scale_ShouldScaleCorrectly()
    {
        var box = new BoundingBox(10, 20, 100, 200);
        var scaled = box.Scale(2.0f, 0.5f);

        scaled.X1.Should().Be(20);
        scaled.Y1.Should().Be(10);
        scaled.X2.Should().Be(200);
        scaled.Y2.Should().Be(100);
    }
}
