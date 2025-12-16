using FluentAssertions;
using LMSupply.Vision;

namespace LMSupply.Vision.Core.Tests;

public class SegmentationMaskTests
{
    [Fact]
    public void Create_FromBoolArray_ShouldSetProperties()
    {
        var mask = new bool[10, 20];
        mask[5, 10] = true;
        mask[5, 11] = true;

        var segMask = new SegmentationMask(mask, 5);

        segMask.Width.Should().Be(20);
        segMask.Height.Should().Be(10);
        segMask.ClassId.Should().Be(5);
        segMask.PixelCount.Should().Be(2);
    }

    [Fact]
    public void Create_FromClassMap_ShouldExtractCorrectPixels()
    {
        var classMap = new int[5, 5];
        classMap[0, 0] = 1;
        classMap[1, 1] = 1;
        classMap[2, 2] = 2;
        classMap[3, 3] = 1;

        var segMask = new SegmentationMask(classMap, 1);

        segMask.PixelCount.Should().Be(3);
        segMask.Contains(0, 0).Should().BeTrue();
        segMask.Contains(1, 1).Should().BeTrue();
        segMask.Contains(2, 2).Should().BeFalse();
        segMask.Contains(3, 3).Should().BeTrue();
    }

    [Fact]
    public void BoundingBox_ShouldEncloseAllPixels()
    {
        var mask = new bool[10, 10];
        mask[2, 3] = true;
        mask[7, 8] = true;

        var segMask = new SegmentationMask(mask, 0);

        segMask.BoundingBox.X.Should().Be(3);
        segMask.BoundingBox.Y.Should().Be(2);
        segMask.BoundingBox.Right.Should().Be(9);
        segMask.BoundingBox.Bottom.Should().Be(8);
    }

    [Fact]
    public void EmptyMask_ShouldHaveZeroBoundingBox()
    {
        var mask = new bool[10, 10];
        var segMask = new SegmentationMask(mask, 0);

        segMask.PixelCount.Should().Be(0);
        segMask.BoundingBox.Area.Should().Be(0);
    }

    [Fact]
    public void Contains_ShouldReturnCorrectValues()
    {
        var mask = new bool[10, 10];
        mask[5, 5] = true;

        var segMask = new SegmentationMask(mask, 0);

        segMask.Contains(5, 5).Should().BeTrue();
        segMask.Contains(0, 0).Should().BeFalse();
        segMask.Contains(-1, -1).Should().BeFalse();
        segMask.Contains(100, 100).Should().BeFalse();
    }

    [Fact]
    public void GetConfidence_WithoutConfidenceMap_ShouldReturnNull()
    {
        var mask = new bool[10, 10];
        mask[5, 5] = true;

        var segMask = new SegmentationMask(mask, 0);

        segMask.GetConfidence(5, 5).Should().BeNull();
    }

    [Fact]
    public void GetConfidence_WithConfidenceMap_ShouldReturnValue()
    {
        var mask = new bool[10, 10];
        mask[5, 5] = true;

        var confidence = new float[10, 10];
        confidence[5, 5] = 0.95f;

        var segMask = new SegmentationMask(mask, 0) { Confidence = confidence };

        segMask.GetConfidence(5, 5).Should().Be(0.95f);
    }

    [Fact]
    public void GetCoveragePercent_ShouldCalculateCorrectly()
    {
        var mask = new bool[10, 10]; // 100 pixels total
        for (int i = 0; i < 5; i++)
            for (int j = 0; j < 5; j++)
                mask[i, j] = true; // 25 pixels set

        var segMask = new SegmentationMask(mask, 0);

        segMask.GetCoveragePercent().Should().Be(25f);
    }

    [Fact]
    public void Resize_ShouldResizeCorrectly()
    {
        var mask = new bool[10, 10];
        for (int i = 0; i < 10; i++)
            for (int j = 0; j < 10; j++)
                mask[i, j] = true;

        var segMask = new SegmentationMask(mask, 5) { Label = "test" };
        var resized = segMask.Resize(20, 20);

        resized.Width.Should().Be(20);
        resized.Height.Should().Be(20);
        resized.ClassId.Should().Be(5);
        resized.Label.Should().Be("test");
        resized.PixelCount.Should().Be(400); // All pixels should still be set
    }

    [Fact]
    public void Resize_ShouldPreservePattern()
    {
        var mask = new bool[4, 4];
        mask[0, 0] = true;
        mask[0, 1] = true;
        mask[1, 0] = true;
        mask[1, 1] = true;

        var segMask = new SegmentationMask(mask, 0);
        var resized = segMask.Resize(8, 8);

        // Top-left quadrant should be filled
        resized.Contains(0, 0).Should().BeTrue();
        resized.Contains(3, 3).Should().BeTrue();
        // Bottom-right should be empty
        resized.Contains(7, 7).Should().BeFalse();
    }

    [Fact]
    public void IoU_IdenticalMasks_ShouldReturnOne()
    {
        var mask = new bool[10, 10];
        mask[5, 5] = true;

        var segMask1 = new SegmentationMask(mask, 0);
        var segMask2 = new SegmentationMask(mask, 0);

        segMask1.IoU(segMask2).Should().Be(1.0f);
    }

    [Fact]
    public void IoU_NoOverlap_ShouldReturnZero()
    {
        var mask1 = new bool[10, 10];
        mask1[0, 0] = true;

        var mask2 = new bool[10, 10];
        mask2[9, 9] = true;

        var segMask1 = new SegmentationMask(mask1, 0);
        var segMask2 = new SegmentationMask(mask2, 0);

        segMask1.IoU(segMask2).Should().Be(0.0f);
    }

    [Fact]
    public void IoU_PartialOverlap_ShouldCalculateCorrectly()
    {
        var mask1 = new bool[10, 10];
        var mask2 = new bool[10, 10];

        // mask1: pixels (0,0), (0,1), (1,0), (1,1)
        // mask2: pixels (1,1), (1,2), (2,1), (2,2)
        // Overlap: (1,1)
        // Union: 7 pixels
        // IoU: 1/7

        mask1[0, 0] = mask1[0, 1] = mask1[1, 0] = mask1[1, 1] = true;
        mask2[1, 1] = mask2[1, 2] = mask2[2, 1] = mask2[2, 2] = true;

        var segMask1 = new SegmentationMask(mask1, 0);
        var segMask2 = new SegmentationMask(mask2, 0);

        segMask1.IoU(segMask2).Should().BeApproximately(1f / 7f, 0.001f);
    }

    [Fact]
    public void IoU_DifferentDimensions_ShouldThrow()
    {
        var mask1 = new bool[10, 10];
        var mask2 = new bool[20, 20];

        var segMask1 = new SegmentationMask(mask1, 0);
        var segMask2 = new SegmentationMask(mask2, 0);

        var act = () => segMask1.IoU(segMask2);
        act.Should().Throw<ArgumentException>();
    }

    [Fact]
    public void ToByteArray_ShouldConvertCorrectly()
    {
        var mask = new bool[2, 3];
        mask[0, 0] = true;
        mask[1, 2] = true;

        var segMask = new SegmentationMask(mask, 0);
        var bytes = segMask.ToByteArray();

        bytes.Should().HaveCount(6);
        bytes[0].Should().Be(255); // (0,0)
        bytes[1].Should().Be(0);   // (0,1)
        bytes[5].Should().Be(255); // (1,2)
    }

    [Fact]
    public void FromByteArray_ShouldCreateCorrectMask()
    {
        var bytes = new byte[] { 255, 0, 0, 0, 0, 255 };
        var segMask = SegmentationMask.FromByteArray(bytes, 3, 2, 5);

        segMask.Width.Should().Be(3);
        segMask.Height.Should().Be(2);
        segMask.ClassId.Should().Be(5);
        segMask.Contains(0, 0).Should().BeTrue();
        segMask.Contains(2, 1).Should().BeTrue();
        segMask.Contains(1, 0).Should().BeFalse();
    }

    [Fact]
    public void FromByteArray_WithCustomThreshold_ShouldWork()
    {
        var bytes = new byte[] { 100, 200, 50 };
        var segMask = SegmentationMask.FromByteArray(bytes, 3, 1, 0, threshold: 100);

        segMask.Contains(0, 0).Should().BeTrue();  // 100 >= 100
        segMask.Contains(1, 0).Should().BeTrue();  // 200 >= 100
        segMask.Contains(2, 0).Should().BeFalse(); // 50 < 100
    }

    [Fact]
    public void FromByteArray_InvalidLength_ShouldThrow()
    {
        var bytes = new byte[] { 255, 0, 0 };
        var act = () => SegmentationMask.FromByteArray(bytes, 5, 5, 0);

        act.Should().Throw<ArgumentException>();
    }

    [Fact]
    public void Label_ShouldBeSettable()
    {
        var mask = new bool[5, 5];
        var segMask = new SegmentationMask(mask, 1) { Label = "wall" };

        segMask.Label.Should().Be("wall");
    }
}
