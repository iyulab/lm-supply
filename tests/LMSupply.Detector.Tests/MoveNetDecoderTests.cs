using System.Text.Json;
using AwesomeAssertions;
using LMSupply.Detector.Core;
using LMSupply.Detector.Models;
using Xunit;

namespace LMSupply.Detector.Tests;

/// <summary>
/// The MoveNet decode: row before column, COCO order kept, coordinates mapped back through the padding the
/// reference preprocessing adds.
///
/// <para>
/// Each of those mistakes produces plausible keypoints rather than an error - x and y swapped still lands
/// inside the image, and a padded input decoded as if stretched only pulls every point towards the padded
/// edge. The hand-built tensors pin each rule on its own; the fixtures are recordings of the real models on
/// real photographs, with the pixel coordinates computed outside the library, and the recorded keypoints were
/// drawn onto the photograph and looked at.
/// </para>
/// </summary>
public class MoveNetDecoderTests
{
    private const string Label = "person";

    private static readonly JsonSerializerOptions JsonOptions = new() { PropertyNameCaseInsensitive = true };

    private static float[] SinglePose(Func<int, (float Y, float X, float Score)> keypoint)
    {
        var output = new float[PoseSkeleton.Count * 3];
        for (var k = 0; k < PoseSkeleton.Count; k++)
        {
            var (y, x, score) = keypoint(k);
            output[k * 3] = y;
            output[k * 3 + 1] = x;
            output[k * 3 + 2] = score;
        }

        return output;
    }

    [Fact]
    public void SinglePose_ReadsRowBeforeColumn_AndKeepsCocoOrder()
    {
        // Keypoint k sits at y = k / 100, x = 0.5 + k / 100 on a square, unpadded input.
        var output = SinglePose(k => (k / 100f, 0.5f + k / 100f, 0.9f));
        var frame = DetectorInputFrame.Padded(100, 100, 1000, 1000);

        var person = MoveNetDecoder.DecodeSinglePose(output, frame, 0.25f, Label).Should().ContainSingle().Subject;

        person.Keypoints.Should().HaveCount(PoseSkeleton.Count);
        person.Keypoints![PoseSkeleton.Nose].X.Should().BeApproximately(500f, 1e-3f);
        person.Keypoints[PoseSkeleton.Nose].Y.Should().BeApproximately(0f, 1e-3f);
        person.Keypoints[PoseSkeleton.RightAnkle].X.Should().BeApproximately(660f, 1e-3f);
        person.Keypoints[PoseSkeleton.RightAnkle].Y.Should().BeApproximately(160f, 1e-3f);
    }

    [Fact]
    public void SinglePose_KeepsEachKeypointsOwnScore_AndScoresThePersonByTheirMean()
    {
        var output = SinglePose(k => (0.5f, 0.5f, k == PoseSkeleton.LeftWrist ? 0.1f : 0.8f));
        var frame = DetectorInputFrame.Padded(192, 192, 640, 640);

        var person = MoveNetDecoder.DecodeSinglePose(output, frame, 0.25f, Label).Should().ContainSingle().Subject;

        person.Keypoints![PoseSkeleton.LeftWrist].Confidence.Should().Be(0.1f);
        person.Keypoints[PoseSkeleton.Nose].Confidence.Should().Be(0.8f);
        person.Confidence.Should().BeApproximately((16 * 0.8f + 0.1f) / 17, 1e-5f);
        person.Label.Should().Be(Label);
        person.ClassId.Should().Be(0);
        person.HasKeypoints.Should().BeTrue();
    }

    [Fact]
    public void SinglePose_ReportsNobody_WhenTheMeanKeypointScoreIsBelowTheThreshold()
    {
        // The model always places 17 points; on a frame without a person their scores are low.
        var output = SinglePose(k => (0.3f, 0.3f, 0.1f));

        MoveNetDecoder.DecodeSinglePose(output, DetectorInputFrame.Padded(192, 192, 640, 480), 0.25f, Label)
            .Should().BeEmpty();
    }

    [Fact]
    public void SinglePose_MapsThroughThePadding_NotAsIfStretched()
    {
        // A 400x200 image on a 100x100 input: resized to 100x50, padded 25 above and below.
        var frame = DetectorInputFrame.Padded(100, 100, 400, 200);
        frame.ContentWidth.Should().Be(100);
        frame.ContentHeight.Should().Be(50);
        frame.PadY.Should().Be(25);

        // y = 0.25 is the top edge of the image content, y = 0.75 its bottom edge.
        var output = SinglePose(k => (k % 2 == 0 ? 0.25f : 0.75f, k / 16f, 0.9f));

        var person = MoveNetDecoder.DecodeSinglePose(output, frame, 0f, Label).Single();

        person.Keypoints![0].Y.Should().BeApproximately(0f, 1e-3f);
        person.Keypoints[1].Y.Should().BeApproximately(200f, 1e-3f);
        person.Keypoints[16].X.Should().BeApproximately(400f, 1e-3f);
    }

    [Fact]
    public void SinglePose_BoxIsTheExtentOfTheKeypoints_ClampedToTheImage()
    {
        var frame = DetectorInputFrame.Padded(100, 100, 100, 100);
        var output = SinglePose(k => k switch
        {
            0 => (0.1f, 0.2f, 0.9f),
            1 => (0.9f, 0.7f, 0.9f),
            2 => (0.5f, 1.2f, 0.9f), // placed past the right edge
            _ => (0.5f, 0.5f, 0.9f)
        });

        var person = MoveNetDecoder.DecodeSinglePose(output, frame, 0f, Label).Single();

        person.Box.X1.Should().BeApproximately(20f, 1e-3f);
        person.Box.Y1.Should().BeApproximately(10f, 1e-3f);
        person.Box.X2.Should().Be(100f, "the box is clamped to the image");
        person.Box.Y2.Should().BeApproximately(90f, 1e-3f);
        person.Keypoints![2].X.Should().BeApproximately(120f, 1e-3f, "a keypoint is an estimate and is not moved onto the border");
    }

    [Fact]
    public void SinglePose_RefusesAnOutputOfTheWrongLength()
    {
        var act = () => MoveNetDecoder.DecodeSinglePose(new float[56], DetectorInputFrame.Padded(192, 192, 10, 10), 0f, Label);

        act.Should().Throw<ArgumentException>().WithMessage("*51 values*");
    }

    [Fact]
    public void MultiPose_ReadsTheBoxAsYMinXMinYMaxXMax_AndKeepsItsScore()
    {
        var output = new float[2 * MoveNetDecoder.MultiPoseInstanceLength];
        // Instance 0: keypoint k at (y = 0.1, x = 0.2 + k / 100), box y 0.1..0.6, x 0.2..0.4, score 0.7.
        for (var k = 0; k < PoseSkeleton.Count; k++)
        {
            output[k * 3] = 0.1f;
            output[k * 3 + 1] = 0.2f + k / 100f;
            output[k * 3 + 2] = 0.5f;
        }

        output[51] = 0.1f;
        output[52] = 0.2f;
        output[53] = 0.6f;
        output[54] = 0.4f;
        output[55] = 0.7f;
        // Instance 1 scores 0.1 and is dropped at the threshold.
        output[MoveNetDecoder.MultiPoseInstanceLength + 55] = 0.1f;

        var frame = DetectorInputFrame.Padded(256, 256, 1000, 1000);
        var person = MoveNetDecoder.DecodeMultiPose(output, frame, 0.25f, Label).Should().ContainSingle().Subject;

        person.Confidence.Should().Be(0.7f);
        person.Box.X1.Should().BeApproximately(200f, 1e-2f);
        person.Box.Y1.Should().BeApproximately(100f, 1e-2f);
        person.Box.X2.Should().BeApproximately(400f, 1e-2f);
        person.Box.Y2.Should().BeApproximately(600f, 1e-2f);
        person.Keypoints.Should().HaveCount(PoseSkeleton.Count);
        person.Keypoints![PoseSkeleton.RightAnkle].X.Should().BeApproximately(360f, 1e-2f);
        person.Keypoints[PoseSkeleton.RightAnkle].Y.Should().BeApproximately(100f, 1e-2f);
    }

    [Fact]
    public void MultiPose_RefusesAnOutputThatIsNotWholeInstances()
    {
        var act = () => MoveNetDecoder.DecodeMultiPose(new float[51], DetectorInputFrame.Padded(256, 256, 10, 10), 0f, Label);

        act.Should().Throw<ArgumentException>().WithMessage("*56-value instances*");
    }

    // ── Recorded runs of the real models ─────────────────────────────

    private sealed record SinglePoseFixture(int ImageWidth, int ImageHeight, int InputSize, float[] Output, float[][] Expected);

    private sealed record MultiPoseFixture(int ImageWidth, int ImageHeight, int InputSize, float[] Output, ExpectedInstance[] Expected);

    private sealed record ExpectedInstance(float Score, float[] Box);

    private static T Load<T>(string name)
    {
        using var stream = File.OpenRead(Path.Combine(AppContext.BaseDirectory, "Fixtures", name));
        return JsonSerializer.Deserialize<T>(stream, JsonOptions)!;
    }

    [Fact]
    public void RecordedSinglePoseRun_DecodesToTheKeypointsComputedOutsideTheLibrary()
    {
        var fixture = Load<SinglePoseFixture>("movenet-singlepose-person.json");
        var frame = DetectorInputFrame.Padded(fixture.InputSize, fixture.InputSize, fixture.ImageWidth, fixture.ImageHeight);

        var person = MoveNetDecoder.DecodeSinglePose(fixture.Output, frame, 0.25f, Label).Should().ContainSingle().Subject;

        for (var k = 0; k < PoseSkeleton.Count; k++)
        {
            person.Keypoints![k].X.Should().BeApproximately(fixture.Expected[k][0], 0.05f, PoseSkeleton.Names[k]);
            person.Keypoints[k].Y.Should().BeApproximately(fixture.Expected[k][1], 0.05f, PoseSkeleton.Names[k]);
            person.Keypoints[k].Confidence.Should().BeApproximately(fixture.Expected[k][2], 1e-3f, PoseSkeleton.Names[k]);
        }

        // A standing person: head above shoulders above hips above ankles.
        var kp = person.Keypoints!;
        kp[PoseSkeleton.Nose].Y.Should().BeLessThan(kp[PoseSkeleton.LeftShoulder].Y);
        kp[PoseSkeleton.LeftShoulder].Y.Should().BeLessThan(kp[PoseSkeleton.LeftHip].Y);
        kp[PoseSkeleton.LeftHip].Y.Should().BeLessThan(kp[PoseSkeleton.LeftAnkle].Y);
    }

    [Fact]
    public void RecordedMultiPoseRun_FindsThePlayers_AndDropsTheInstanceBelowTheThreshold()
    {
        var fixture = Load<MultiPoseFixture>("movenet-multipose-football.json");
        var frame = DetectorInputFrame.Padded(fixture.InputSize, fixture.InputSize, fixture.ImageWidth, fixture.ImageHeight);

        // Four instances score at least 0.2; the fourth is 0.18 and falls below the library default of 0.25.
        MoveNetDecoder.DecodeMultiPose(fixture.Output, frame, 0.1f, Label).Should().HaveCount(4);
        var people = MoveNetDecoder.DecodeMultiPose(fixture.Output, frame, 0.25f, Label);

        people.Should().HaveCount(fixture.Expected.Length);
        foreach (var (person, expected) in people.Zip(fixture.Expected))
        {
            person.Confidence.Should().BeApproximately(expected.Score, 1e-3f);
            person.Box.X1.Should().BeApproximately(expected.Box[0], 0.05f);
            person.Box.Y1.Should().BeApproximately(expected.Box[1], 0.05f);
            person.Box.X2.Should().BeApproximately(expected.Box[2], 0.05f);
            person.Box.Y2.Should().BeApproximately(expected.Box[3], 0.05f);
            person.Keypoints.Should().HaveCount(PoseSkeleton.Count);
        }
    }

    [Fact]
    public void BothMoveNetLayouts_CarryTheCocoKeypoints_AndNeedNoSuppression()
    {
        foreach (var layout in new[] { DetectorOutputLayout.MoveNetSinglePose, DetectorOutputLayout.MoveNetMultiPose })
        {
            layout.KeypointCount().Should().Be(PoseSkeleton.Count);
            layout.RequiresNms().Should().BeFalse("MoveNet emits final instances");
        }
    }
}
