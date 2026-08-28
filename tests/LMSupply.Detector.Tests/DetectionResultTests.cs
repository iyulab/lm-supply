using AwesomeAssertions;

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

    [Fact]
    public void DetectionResult_WithoutKeypoints_HasKeypointsShouldBeFalse()
    {
        var result = new DetectionResult(0, "person", 0.95f, new BoundingBox(0, 0, 100, 100));

        result.HasKeypoints.Should().BeFalse();
        result.Keypoints.Should().BeNull();
    }

    [Fact]
    public void DetectionResult_WithKeypoints_HasKeypointsShouldBeTrue()
    {
        var keypoints = new[]
        {
            new Keypoint(50, 30, 0.9f),
            new Keypoint(45, 35, 0.8f),
            new Keypoint(55, 35, 0.85f)
        };
        var result = new DetectionResult(0, "person", 0.95f, new BoundingBox(0, 0, 100, 200), keypoints);

        result.HasKeypoints.Should().BeTrue();
        result.Keypoints.Should().HaveCount(3);
        result.Keypoints![0].X.Should().Be(50);
        result.Keypoints![0].Y.Should().Be(30);
        result.Keypoints![0].Confidence.Should().Be(0.9f);
    }

    [Fact]
    public void DetectionResult_BackwardCompatible_FourArgConstructor()
    {
        // Existing code creating DetectionResult with 4 args should still compile and work
        var box = new BoundingBox(10, 20, 100, 200);
        var result = new DetectionResult(0, "car", 0.8f, box);

        result.ClassId.Should().Be(0);
        result.Label.Should().Be("car");
        result.Keypoints.Should().BeNull();
        result.HasKeypoints.Should().BeFalse();
    }

    [Fact]
    public void DetectionResult_ToString_WithKeypoints_IncludesKeypointCount()
    {
        var keypoints = new Keypoint[17];
        var result = new DetectionResult(0, "person", 0.95f, new BoundingBox(0, 0, 100, 200), keypoints);

        result.ToString().Should().Contain("17 keypoints");
    }
}

public class KeypointTests
{
    [Fact]
    public void Keypoint_ShouldStoreAllProperties()
    {
        var kp = new Keypoint(100.5f, 200.3f, 0.95f);

        kp.X.Should().Be(100.5f);
        kp.Y.Should().Be(200.3f);
        kp.Confidence.Should().Be(0.95f);
    }

    [Fact]
    public void IsVisible_AboveThreshold_ShouldReturnTrue()
    {
        var kp = new Keypoint(0, 0, 0.8f);

        kp.IsVisible().Should().BeTrue();
        kp.IsVisible(0.7f).Should().BeTrue();
    }

    [Fact]
    public void IsVisible_BelowThreshold_ShouldReturnFalse()
    {
        var kp = new Keypoint(0, 0, 0.3f);

        kp.IsVisible().Should().BeFalse();
        kp.IsVisible(0.5f).Should().BeFalse();
    }
}

public class PoseSkeletonTests
{
    [Fact]
    public void Count_ShouldBe17()
    {
        PoseSkeleton.Count.Should().Be(17);
    }

    [Fact]
    public void Names_ShouldHave17Entries()
    {
        PoseSkeleton.Names.Should().HaveCount(17);
    }

    [Fact]
    public void Bones_ShouldHaveCorrectConnections()
    {
        PoseSkeleton.Bones.Should().NotBeEmpty();
        PoseSkeleton.Bones.Should().Contain((PoseSkeleton.LeftShoulder, PoseSkeleton.RightShoulder));
    }

    [Fact]
    public void KeypointIndices_ShouldBeCorrect()
    {
        PoseSkeleton.Nose.Should().Be(0);
        PoseSkeleton.LeftShoulder.Should().Be(5);
        PoseSkeleton.RightAnkle.Should().Be(16);
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
