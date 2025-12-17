using FluentAssertions;
using LMSupply.Vision;

namespace LMSupply.Vision.Core.Tests;

public class DetectionTests
{
    [Fact]
    public void Create_ShouldSetProperties()
    {
        var box = new BoundingBox(10, 20, 100, 50);
        var detection = new Detection(box, 5, 0.95f, "person");

        detection.BoundingBox.Should().Be(box);
        detection.ClassId.Should().Be(5);
        detection.Confidence.Should().Be(0.95f);
        detection.Label.Should().Be("person");
    }

    [Fact]
    public void ShortcutProperties_ShouldWorkCorrectly()
    {
        var detection = new Detection(new BoundingBox(10, 20, 100, 50), 0, 0.9f);

        detection.X.Should().Be(10);
        detection.Y.Should().Be(20);
        detection.Width.Should().Be(100);
        detection.Height.Should().Be(50);
        detection.Area.Should().Be(5000);
    }

    [Fact]
    public void Scale_ShouldScaleBoundingBox()
    {
        var detection = new Detection(new BoundingBox(10, 20, 100, 50), 0, 0.9f, "test");
        var scaled = detection.Scale(2f, 0.5f);

        scaled.X.Should().Be(20);
        scaled.Y.Should().Be(10);
        scaled.Width.Should().Be(200);
        scaled.Height.Should().Be(25);
        scaled.ClassId.Should().Be(0);
        scaled.Confidence.Should().Be(0.9f);
        scaled.Label.Should().Be("test");
    }

    [Fact]
    public void Clip_ShouldClipBoundingBox()
    {
        var detection = new Detection(new BoundingBox(-10, -20, 200, 150), 0, 0.9f);
        var clipped = detection.Clip(100, 100);

        clipped.X.Should().BeGreaterThanOrEqualTo(0);
        clipped.Y.Should().BeGreaterThanOrEqualTo(0);
        clipped.BoundingBox.Right.Should().BeLessThanOrEqualTo(100);
        clipped.BoundingBox.Bottom.Should().BeLessThanOrEqualTo(100);
    }

    [Fact]
    public void IoU_ShouldCalculateCorrectly()
    {
        var d1 = new Detection(new BoundingBox(0, 0, 10, 10), 0, 0.9f);
        var d2 = new Detection(new BoundingBox(0, 0, 10, 10), 0, 0.8f);

        d1.IoU(d2).Should().Be(1.0f);
    }

    [Fact]
    public void ToString_ShouldIncludeLabel()
    {
        var detection = new Detection(new BoundingBox(10, 20, 100, 50), 5, 0.95f, "person");
        var str = detection.ToString();

        str.Should().Contain("person");
        str.Should().Contain("95");
    }

    [Fact]
    public void ToString_WithoutLabel_ShouldShowClassId()
    {
        var detection = new Detection(new BoundingBox(10, 20, 100, 50), 5, 0.95f);
        var str = detection.ToString();

        str.Should().Contain("Class5");
    }
}

public class DetectionExtensionsTests
{
    private readonly List<Detection> _testDetections =
    [
        new(new BoundingBox(0, 0, 10, 10), 0, 0.9f, "person"),
        new(new BoundingBox(20, 20, 10, 10), 0, 0.7f, "person"),
        new(new BoundingBox(40, 40, 10, 10), 1, 0.8f, "car"),
        new(new BoundingBox(60, 60, 10, 10), 1, 0.5f, "car"),
        new(new BoundingBox(80, 80, 10, 10), 2, 0.95f, "dog")
    ];

    [Fact]
    public void FilterByConfidence_ShouldFilterCorrectly()
    {
        var filtered = _testDetections.FilterByConfidence(0.75f).ToList();

        filtered.Should().HaveCount(3);
        filtered.Should().OnlyContain(d => d.Confidence >= 0.75f);
    }

    [Fact]
    public void FilterByClass_ShouldFilterCorrectly()
    {
        var filtered = _testDetections.FilterByClass([0, 2]).ToList();

        filtered.Should().HaveCount(3);
        filtered.Should().OnlyContain(d => d.ClassId == 0 || d.ClassId == 2);
    }

    [Fact]
    public void FilterByLabel_ShouldFilterCorrectly()
    {
        var filtered = _testDetections.FilterByLabel("person", "dog").ToList();

        filtered.Should().HaveCount(3);
        filtered.Should().OnlyContain(d => d.Label == "person" || d.Label == "dog");
    }

    [Fact]
    public void FilterByLabel_ShouldBeCaseInsensitive()
    {
        var filtered = _testDetections.FilterByLabel("PERSON").ToList();

        filtered.Should().HaveCount(2);
    }

    [Fact]
    public void ApplyNms_ShouldRemoveOverlappingDetections()
    {
        var overlapping = new List<Detection>
        {
            new(new BoundingBox(0, 0, 10, 10), 0, 0.9f),
            new(new BoundingBox(1, 1, 10, 10), 0, 0.8f), // High IoU with first
            new(new BoundingBox(20, 20, 10, 10), 0, 0.7f) // No overlap
        };

        var result = overlapping.ApplyNms(0.5f);

        result.Should().HaveCount(2);
        result[0].Confidence.Should().Be(0.9f);
        result[1].Confidence.Should().Be(0.7f);
    }

    [Fact]
    public void ApplyClassAwareNms_ShouldApplyNmsPerClass()
    {
        var mixed = new List<Detection>
        {
            new(new BoundingBox(0, 0, 10, 10), 0, 0.9f),
            new(new BoundingBox(1, 1, 10, 10), 0, 0.8f), // Same class, high IoU
            new(new BoundingBox(1, 1, 10, 10), 1, 0.85f), // Different class, same position
            new(new BoundingBox(20, 20, 10, 10), 0, 0.7f)
        };

        var result = mixed.ApplyClassAwareNms(0.5f);

        // Should keep: class 0 (0.9f, 0.7f), class 1 (0.85f)
        result.Should().HaveCount(3);
    }

    [Fact]
    public void Scale_ShouldScaleAllDetections()
    {
        var scaled = _testDetections.Scale(2f, 2f).ToList();

        scaled.Should().HaveCount(5);
        scaled[0].X.Should().Be(0);
        scaled[1].X.Should().Be(40); // 20 * 2
    }

    [Fact]
    public void Clip_ShouldClipAllDetections()
    {
        var overflowing = new List<Detection>
        {
            new(new BoundingBox(-5, -5, 20, 20), 0, 0.9f),
            new(new BoundingBox(90, 90, 20, 20), 0, 0.8f)
        };

        var clipped = overflowing.Clip(100, 100).ToList();

        clipped[0].X.Should().BeGreaterThanOrEqualTo(0);
        clipped[1].BoundingBox.Right.Should().BeLessThanOrEqualTo(100);
    }

    [Fact]
    public void TopN_ShouldReturnTopByConfidence()
    {
        var topTwo = _testDetections.TopN(2);

        topTwo.Should().HaveCount(2);
        topTwo[0].Confidence.Should().Be(0.95f);
        topTwo[1].Confidence.Should().Be(0.9f);
    }

    [Fact]
    public void ChainedFilters_ShouldWorkCorrectly()
    {
        var result = _testDetections
            .FilterByConfidence(0.7f)
            .FilterByClass([0])
            .TopN(1);

        result.Should().HaveCount(1);
        result[0].Label.Should().Be("person");
        result[0].Confidence.Should().Be(0.9f);
    }
}
