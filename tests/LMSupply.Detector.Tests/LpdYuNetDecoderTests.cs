using System.Text.Json;
using AwesomeAssertions;
using LMSupply.Detector.Core;
using Xunit;

namespace LMSupply.Detector.Tests;

/// <summary>
/// The licence-plate decode, pinned against a real run of the real model.
///
/// <para>
/// The priors are the part worth guarding. They are not stored in the model — they are regenerated from the
/// input size by halving it five times with a particular rounding, and if that reconstruction is off by one
/// anywhere, every box lands somewhere plausible and wrong. The count is the cheapest check that it is
/// right: the model's outputs are 4385 rows long, so a generator producing any other number is provably
/// disagreeing with the weights.
/// </para>
/// <para>
/// The rest is a recording: every prior that scored above 0.3 when the model ran on a photograph of a car,
/// and the box that run produced. That box was cropped out of the source and contains the number plate.
/// </para>
/// </summary>
public class LpdYuNetDecoderTests
{
    private sealed record Fixture(
        int ImageWidth,
        int ImageHeight,
        int InputWidth,
        int InputHeight,
        float ConfidenceThreshold,
        float IouThreshold,
        IReadOnlyList<Anchor> Anchors,
        IReadOnlyList<ExpectedBox> ExpectedBeforeNms);

    private sealed record Anchor(int Index, float Cls, float Iou, float[] Loc);

    private sealed record ExpectedBox(float Score, float X1, float Y1, float X2, float Y2);

    private static readonly JsonSerializerOptions JsonOptions = new() { PropertyNameCaseInsensitive = true };

    private static readonly Fixture Recorded = Load();

    private static Fixture Load()
    {
        var path = Path.Combine(AppContext.BaseDirectory, "Fixtures", "lpd-car-anchors.json");
        using var stream = File.OpenRead(path);
        return JsonSerializer.Deserialize<Fixture>(stream, JsonOptions)!;
    }

    private static LpdPrior[] Priors() =>
        LpdYuNetDecoder.GeneratePriors(Recorded.InputWidth, Recorded.InputHeight);

    private static LpdYuNetOutput Rebuild()
    {
        var priors = Priors();
        var loc = new float[priors.Length * LpdYuNetDecoder.LocStride];
        var conf = new float[priors.Length * 2];
        var iou = new float[priors.Length];

        foreach (var anchor in Recorded.Anchors)
        {
            Array.Copy(anchor.Loc, 0, loc, anchor.Index * LpdYuNetDecoder.LocStride, LpdYuNetDecoder.LocStride);
            conf[anchor.Index * 2 + 1] = anchor.Cls;
            iou[anchor.Index] = anchor.Iou;
        }

        return new LpdYuNetOutput(loc, conf, iou);
    }

    private static List<DetectionResult> DecodeRecorded(float? threshold = null) => LpdYuNetDecoder.Decode(
        Rebuild(),
        Priors(),
        Recorded.ImageWidth,
        Recorded.ImageHeight,
        threshold ?? Recorded.ConfidenceThreshold,
        "plate");

    [Fact]
    public void ThePriorsMatchTheLengthOfTheModelsOwnOutputs()
    {
        // 30x40x3 + 15x20x2 + 7x10x2 + 3x5x3. Any other number means the reconstruction disagrees with the
        // weights, and every offset would then be read against the wrong reference point.
        LpdYuNetDecoder.GeneratePriors(320, 240).Should().HaveCount(4385);
    }

    [Fact]
    public void PriorsAreNormalisedAndOrderedFinestFirst()
    {
        var priors = Priors();

        priors.Should().OnlyContain(p =>
            p.CenterX > 0 && p.CenterX < 1 && p.CenterY > 0 && p.CenterY < 1 && p.ScaleX > 0 && p.ScaleY > 0);

        // First cell of the finest map, first of its three sizes: centre half a step in, extent 10 pixels.
        priors[0].CenterX.Should().BeApproximately(0.5f * 8 / 320, 1e-6f);
        priors[0].CenterY.Should().BeApproximately(0.5f * 8 / 240, 1e-6f);
        priors[0].ScaleX.Should().BeApproximately(10f / 320, 1e-6f);

        // The coarsest map's priors come last and are much larger.
        priors[^1].ScaleX.Should().BeApproximately(256f / 320, 1e-6f);
    }

    [Fact]
    public void DecodingTheRecordedRun_ReproducesTheBoxTheModelProduced()
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
        // Cropped out of the source photograph: this rectangle contains the car's number plate.
        var top = DecodeRecorded().MaxBy(d => d.Confidence);

        top.Label.Should().Be("plate");
        top.Confidence.Should().BeApproximately(0.9136f, 1e-3f);
        top.Box.X1.Should().BeApproximately(449.0f, 0.5f);
        top.Box.Y1.Should().BeApproximately(476.5f, 0.5f);
        top.Box.X2.Should().BeApproximately(513.9f, 0.5f);
        top.Box.Y2.Should().BeApproximately(496.9f, 0.5f);
    }

    [Fact]
    public void TheFourCornersAreKeptAndGoClockwiseFromTopLeft()
    {
        // A plate seen from an angle is a quadrilateral, and the upright hull of a slanted plate covers a
        // good deal that is not plate — so the corners survive rather than being collapsed into the box.
        var top = DecodeRecorded().MaxBy(d => d.Confidence);

        top.Keypoints.Should().NotBeNull().And.HaveCount(4);

        var (topLeft, topRight, bottomRight, bottomLeft) =
            (top.Keypoints![0], top.Keypoints[1], top.Keypoints[2], top.Keypoints[3]);

        topLeft.X.Should().BeLessThan(topRight.X);
        bottomLeft.X.Should().BeLessThan(bottomRight.X);
        topLeft.Y.Should().BeLessThan(bottomLeft.Y);
        topRight.Y.Should().BeLessThan(bottomRight.Y);
    }

    [Fact]
    public void TheBoxIsTheUprightHullOfTheCorners()
    {
        foreach (var detection in DecodeRecorded(0.3f))
        {
            var xs = detection.Keypoints!.Select(k => k.X).ToList();
            var ys = detection.Keypoints!.Select(k => k.Y).ToList();

            detection.Box.X1.Should().BeApproximately(xs.Min(), 0.01f);
            detection.Box.Y1.Should().BeApproximately(ys.Min(), 0.01f);
            detection.Box.X2.Should().BeApproximately(xs.Max(), 0.01f);
            detection.Box.Y2.Should().BeApproximately(ys.Max(), 0.01f);
        }
    }

    [Fact]
    public void ScoreIsTheGeometricMeanOfClassAndOverlapHeads()
    {
        var priors = Priors();
        var conf = new float[priors.Length * 2];
        var iou = new float[priors.Length];
        conf[1] = 0.81f;
        iou[0] = 0.49f;

        var decoded = LpdYuNetDecoder.Decode(
            new LpdYuNetOutput(new float[priors.Length * LpdYuNetDecoder.LocStride], conf, iou),
            priors, 320, 240, 0.1f, "plate");

        decoded.Should().ContainSingle();
        decoded[0].Confidence.Should().BeApproximately(0.63f, 1e-4f, "sqrt(0.81 * 0.49)");
    }

    [Fact]
    public void OutputsThatDoNotMatchThePriorCountAreRefused()
    {
        var priors = Priors();

        var act = () => LpdYuNetDecoder.Decode(
            new LpdYuNetOutput(new float[10], new float[priors.Length * 2], new float[priors.Length]),
            priors, 320, 240, 0.5f, "plate");

        act.Should().Throw<ArgumentException>().WithMessage("*do not match*");
    }
}
