using System.Text.Json;
using AwesomeAssertions;
using LMSupply.Detector.Core;
using Xunit;

namespace LMSupply.Detector.Tests;

/// <summary>
/// The YuNet decode, pinned against a real run of the real model.
///
/// <para>
/// A face detector is not verifiable by reading the code: the twelve outputs could be paired with the wrong
/// strides, the anchor index could be read column-major, the extent could be taken as a length rather than
/// its logarithm, and every one of those produces boxes rather than an error. So the fixture is a recording
/// of an actual inference — every anchor that scored above 0.3 when
/// <c>face_detection_yunet_2023mar.onnx</c> was run on a street photograph — together with the boxes that
/// run produced. The highest-scoring box was cropped out of the source image and looked at; it sits on a
/// face.
/// </para>
/// <para>
/// Only the anchors matter, not the photograph: a few dozen floats travel with the repository, the image
/// does not.
/// </para>
/// </summary>
public class YuNetDecoderTests
{
    private sealed record Fixture(
        int ImageWidth,
        int ImageHeight,
        int InputSize,
        float ConfidenceThreshold,
        float IouThreshold,
        IReadOnlyList<Anchor> Anchors,
        IReadOnlyList<ExpectedBox> ExpectedBeforeNms,
        IReadOnlyList<ExpectedBox> ExpectedAfterNms);

    private sealed record Anchor(int Stride, int Index, float Cls, float Obj, float[] Bbox, float[] Kps);

    private sealed record ExpectedBox(float Score, float X1, float Y1, float X2, float Y2);

    private static readonly JsonSerializerOptions JsonOptions = new() { PropertyNameCaseInsensitive = true };

    private static readonly Fixture Recorded = Load();

    private static Fixture Load()
    {
        var path = Path.Combine(AppContext.BaseDirectory, "Fixtures", "yunet-crowd-anchors.json");
        using var stream = File.OpenRead(path);
        return JsonSerializer.Deserialize<Fixture>(stream, JsonOptions)!;
    }

    /// <summary>
    /// Rebuilds the full output tensors from the recorded sparse anchors. Every anchor that is not in the
    /// fixture scored below the recording threshold, and zeros decode to a score of zero.
    /// </summary>
    private static Dictionary<int, YuNetStrideOutput> Rebuild(int inputSize)
    {
        var branches = new Dictionary<int, YuNetStrideOutput>();

        foreach (var stride in YuNetDecoder.Strides)
        {
            var grid = inputSize / stride;
            var count = grid * grid;
            branches[stride] = new YuNetStrideOutput(
                new float[count],
                new float[count],
                new float[count * 4],
                new float[count * YuNetDecoder.LandmarkCount * 2]);
        }

        foreach (var anchor in Recorded.Anchors)
        {
            var branch = branches[anchor.Stride];
            ((float[])branch.Cls)[anchor.Index] = anchor.Cls;
            ((float[])branch.Obj)[anchor.Index] = anchor.Obj;
            Array.Copy(anchor.Bbox, 0, (float[])branch.Bbox, anchor.Index * 4, 4);
            Array.Copy(anchor.Kps, 0, (float[])branch.Kps, anchor.Index * YuNetDecoder.LandmarkCount * 2,
                YuNetDecoder.LandmarkCount * 2);
        }

        return branches;
    }

    private static List<DetectionResult> DecodeRecorded() => YuNetDecoder.Decode(
        Rebuild(Recorded.InputSize),
        Recorded.InputSize,
        Recorded.ImageWidth,
        Recorded.ImageHeight,
        Recorded.ConfidenceThreshold,
        "face");

    [Fact]
    public void DecodingTheRecordedRun_ReproducesEveryBoxTheModelProduced()
    {
        var decoded = DecodeRecorded().OrderByDescending(d => d.Confidence).ToList();

        decoded.Should().HaveCount(Recorded.ExpectedBeforeNms.Count);

        foreach (var (actual, expected) in decoded.Zip(Recorded.ExpectedBeforeNms))
        {
            actual.Confidence.Should().BeApproximately(expected.Score, 1e-4f);
            actual.Box.X1.Should().BeApproximately(expected.X1, 0.05f);
            actual.Box.Y1.Should().BeApproximately(expected.Y1, 0.05f);
            actual.Box.X2.Should().BeApproximately(expected.X2, 0.05f);
            actual.Box.Y2.Should().BeApproximately(expected.Y2, 0.05f);
        }
    }

    [Fact]
    public void TheBoxThatWasLookedAt_LandsWhereItWasLookedAt()
    {
        // The single assertion in this file that is anchored to something outside the code: this box was
        // cropped out of the source photograph and contains a face.
        var top = DecodeRecorded().MaxBy(d => d.Confidence);

        top.Label.Should().Be("face");
        top.ClassId.Should().Be(0);
        top.Confidence.Should().BeApproximately(0.7438f, 1e-3f);
        top.Box.X1.Should().BeApproximately(581.4f, 0.5f);
        top.Box.Y1.Should().BeApproximately(876.5f, 0.5f);
        top.Box.Width.Should().BeApproximately(22.8f, 0.5f);
        top.Box.Height.Should().BeApproximately(27.2f, 0.5f);
    }

    [Fact]
    public void SuppressionCollapsesTheDuplicateAnchorsEachFaceProduces()
    {
        var decoded = DecodeRecorded();

        // Several anchors fire on one face; that is the head's shape, not a defect.
        decoded.Should().HaveCountGreaterThan(Recorded.ExpectedAfterNms.Count);

        var suppressed = DetectionNms.Apply(decoded, Recorded.IouThreshold);

        suppressed.Should().HaveCount(Recorded.ExpectedAfterNms.Count);
        suppressed.MaxBy(d => d.Confidence).Box.X1
            .Should().BeApproximately(Recorded.ExpectedAfterNms[0].X1, 0.05f);
    }

    [Fact]
    public void EveryDetectionCarriesFiveLandmarksInsideTheImage()
    {
        foreach (var detection in DecodeRecorded())
        {
            detection.HasKeypoints.Should().BeTrue();
            detection.Keypoints!.Should().HaveCount(5, "YuNet emits both eyes, the nose tip and both mouth corners");

            foreach (var point in detection.Keypoints)
            {
                point.X.Should().BeInRange(0, Recorded.ImageWidth);
                point.Y.Should().BeInRange(0, Recorded.ImageHeight);
            }
        }
    }

    [Theory]
    [InlineData(8)]
    [InlineData(16)]
    [InlineData(32)]
    public void AnchorIndexIsReadRowMajor(int stride)
    {
        // Column-major would place this box at the transposed position, and nothing else would complain.
        const int inputSize = 640;
        var grid = inputSize / stride;
        var column = 3;
        var row = 5;
        var index = row * grid + column;

        var branches = Empty(inputSize);
        ((float[])branches[stride].Cls)[index] = 1f;
        ((float[])branches[stride].Obj)[index] = 1f;
        // zero offsets, zero log-extent: the box is one stride wide, centred on the cell corner
        ((float[])branches[stride].Bbox)[index * 4 + 2] = 0f;
        ((float[])branches[stride].Bbox)[index * 4 + 3] = 0f;

        var decoded = YuNetDecoder.Decode(branches, inputSize, inputSize, inputSize, 0.5f, "face");

        decoded.Should().ContainSingle();
        decoded[0].Box.CenterX.Should().BeApproximately(column * stride, 1e-3f);
        decoded[0].Box.CenterY.Should().BeApproximately(row * stride, 1e-3f);
        decoded[0].Box.Width.Should().BeApproximately(stride, 1e-3f);
    }

    [Fact]
    public void ScoreIsTheGeometricMeanOfClassAndObjectness()
    {
        var branches = Empty(640);
        ((float[])branches[32].Cls)[0] = 0.64f;
        ((float[])branches[32].Obj)[0] = 0.25f;

        var decoded = YuNetDecoder.Decode(branches, 640, 640, 640, 0.1f, "face");

        decoded.Should().ContainSingle();
        decoded[0].Confidence.Should().BeApproximately(0.4f, 1e-4f, "sqrt(0.64 * 0.25)");
    }

    [Fact]
    public void AMissingStrideIsRefusedRatherThanDecodedAsFewerFaces()
    {
        var branches = Empty(640);
        branches.Remove(16);

        var act = () => YuNetDecoder.Decode(branches, 640, 640, 640, 0.5f, "face");

        act.Should().Throw<ArgumentException>().WithMessage("*stride 16*");
    }

    [Fact]
    public void OutputsThatDisagreeOnAnchorCountAreRefused()
    {
        var branches = Empty(640);
        branches[8] = branches[8] with { Bbox = new float[7] };

        var act = () => YuNetDecoder.Decode(branches, 640, 640, 640, 0.5f, "face");

        act.Should().Throw<ArgumentException>().WithMessage("*disagree on anchor count*");
    }

    private static Dictionary<int, YuNetStrideOutput> Empty(int inputSize)
    {
        var branches = new Dictionary<int, YuNetStrideOutput>();

        foreach (var stride in YuNetDecoder.Strides)
        {
            var count = (inputSize / stride) * (inputSize / stride);
            branches[stride] = new YuNetStrideOutput(
                new float[count],
                new float[count],
                new float[count * 4],
                new float[count * YuNetDecoder.LandmarkCount * 2]);
        }

        return branches;
    }
}
