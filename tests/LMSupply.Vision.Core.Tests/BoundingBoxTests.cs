using FluentAssertions;
using LMSupply.Vision;

namespace LMSupply.Vision.Core.Tests;

public class BoundingBoxTests
{
    [Fact]
    public void Create_ShouldSetProperties()
    {
        var box = new BoundingBox(10, 20, 100, 50);

        box.X.Should().Be(10);
        box.Y.Should().Be(20);
        box.Width.Should().Be(100);
        box.Height.Should().Be(50);
    }

    [Fact]
    public void ComputedProperties_ShouldBeCorrect()
    {
        var box = new BoundingBox(10, 20, 100, 50);

        box.Right.Should().Be(110);
        box.Bottom.Should().Be(70);
        box.CenterX.Should().Be(60);
        box.CenterY.Should().Be(45);
        box.Area.Should().Be(5000);
    }

    [Fact]
    public void FromCenter_ShouldCreateCorrectBox()
    {
        var box = BoundingBox.FromCenter(60, 45, 100, 50);

        box.X.Should().Be(10);
        box.Y.Should().Be(20);
        box.Width.Should().Be(100);
        box.Height.Should().Be(50);
    }

    [Fact]
    public void FromCorners_ShouldCreateCorrectBox()
    {
        var box = BoundingBox.FromCorners(10, 20, 110, 70);

        box.X.Should().Be(10);
        box.Y.Should().Be(20);
        box.Width.Should().Be(100);
        box.Height.Should().Be(50);
    }

    [Theory]
    [InlineData(0, 0, 10, 10, 5, 5, 10, 10, 0.142857f)] // Partial overlap
    [InlineData(0, 0, 10, 10, 0, 0, 10, 10, 1.0f)] // Complete overlap
    [InlineData(0, 0, 10, 10, 20, 20, 10, 10, 0.0f)] // No overlap
    public void IoU_ShouldReturnCorrectValue(
        float x1, float y1, float w1, float h1,
        float x2, float y2, float w2, float h2,
        float expectedIoU)
    {
        var box1 = new BoundingBox(x1, y1, w1, h1);
        var box2 = new BoundingBox(x2, y2, w2, h2);

        var iou = box1.IoU(box2);

        iou.Should().BeApproximately(expectedIoU, 0.001f);
    }

    [Fact]
    public void Scale_ShouldScaleCorrectly()
    {
        var box = new BoundingBox(10, 20, 100, 50);
        var scaled = box.Scale(2f, 0.5f);

        scaled.X.Should().Be(20);
        scaled.Y.Should().Be(10);
        scaled.Width.Should().Be(200);
        scaled.Height.Should().Be(25);
    }

    [Fact]
    public void Clip_ShouldClipToImageBounds()
    {
        var box = new BoundingBox(-10, -20, 200, 150);
        var clipped = box.Clip(100, 100);

        clipped.X.Should().Be(0);
        clipped.Y.Should().Be(0);
        clipped.Right.Should().BeLessThanOrEqualTo(100);
        clipped.Bottom.Should().BeLessThanOrEqualTo(100);
    }

    [Theory]
    [InlineData(50, 30, true)] // Inside
    [InlineData(10, 20, true)] // On corner
    [InlineData(0, 0, false)] // Outside
    [InlineData(150, 100, false)] // Outside
    public void Contains_ShouldReturnCorrectResult(float x, float y, bool expected)
    {
        var box = new BoundingBox(10, 20, 100, 50);
        box.Contains(x, y).Should().Be(expected);
    }

    [Theory]
    [InlineData(0, 0, 10, 10, 5, 5, 10, 10, true)] // Overlapping
    [InlineData(0, 0, 10, 10, 20, 20, 10, 10, false)] // Not overlapping
    [InlineData(0, 0, 10, 10, 10, 0, 10, 10, false)] // Adjacent (not overlapping)
    public void Intersects_ShouldReturnCorrectResult(
        float x1, float y1, float w1, float h1,
        float x2, float y2, float w2, float h2,
        bool expected)
    {
        var box1 = new BoundingBox(x1, y1, w1, h1);
        var box2 = new BoundingBox(x2, y2, w2, h2);

        box1.Intersects(box2).Should().Be(expected);
    }

    [Fact]
    public void ToString_ShouldReturnFormattedString()
    {
        var box = new BoundingBox(10.5f, 20.5f, 100.25f, 50.75f);
        var str = box.ToString();

        str.Should().Contain("10.50");
        str.Should().Contain("20.50");
        str.Should().Contain("BoundingBox");
    }
}
