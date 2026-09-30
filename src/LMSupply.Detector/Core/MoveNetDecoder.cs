namespace LMSupply.Detector.Core;

/// <summary>
/// Decodes MoveNet's single output into detections. Pure arithmetic over the returned tensor, so it is
/// verifiable without loading the model.
/// </summary>
/// <remarks>
/// <para>
/// Both variants emit the 17 COCO keypoints in COCO order as <c>(y, x, score)</c> - row before column -
/// normalised to the model input. The order is kept, so <see cref="PoseSkeleton"/> indices address the
/// returned keypoints directly, and each keypoint keeps its own score.
/// </para>
/// <para>
/// Coordinates are mapped back to the original image through <see cref="DetectorInputFrame"/>, which accounts
/// for the padding MoveNet's reference preprocessing adds. Keypoints are not clamped to the image: a joint the
/// model places just outside the frame is an estimate, and moving it onto the border would change the angles
/// a consumer computes from it. Boxes are clamped, as every other layout's are.
/// </para>
/// </remarks>
internal static class MoveNetDecoder
{
    /// <summary>Values per keypoint: y, x, score.</summary>
    private const int ValuesPerKeypoint = 3;

    /// <summary>Values per MultiPose instance: 17 keypoints of three values, then ymin, xmin, ymax, xmax, score.</summary>
    public const int MultiPoseInstanceLength = PoseSkeleton.Count * ValuesPerKeypoint + 5;

    /// <summary>
    /// Decodes a SinglePose output (<c>[1, 1, 17, 3]</c>, flattened) into at most one person.
    /// </summary>
    /// <remarks>
    /// SinglePose has no person score and no box. The detection's confidence is the mean of the 17 keypoint
    /// scores - the usual reading of this model's output - so <paramref name="confidenceThreshold"/> separates
    /// a frame with a person in it from one without, where the model still places 17 points somewhere with low
    /// scores. The box is the extent of the 17 keypoints, clamped to the image: the skeleton's extent, not a
    /// detector box, and tighter than the person (the top of the head and the hands lie outside it).
    /// </remarks>
    /// <param name="output">The flattened output; exactly 51 values.</param>
    /// <param name="frame">Where the original image sat inside the model input.</param>
    /// <param name="confidenceThreshold">Minimum mean keypoint score to report a person.</param>
    /// <param name="label">The label to attach, taken from the model's own vocabulary.</param>
    public static List<DetectionResult> DecodeSinglePose(
        IReadOnlyList<float> output,
        DetectorInputFrame frame,
        float confidenceThreshold,
        string label)
    {
        ArgumentNullException.ThrowIfNull(output);
        ArgumentException.ThrowIfNullOrEmpty(label);

        const int expected = PoseSkeleton.Count * ValuesPerKeypoint;
        if (output.Count != expected)
        {
            throw new ArgumentException(
                $"MoveNet SinglePose output must hold {expected} values (17 keypoints of y, x, score); got {output.Count}.",
                nameof(output));
        }

        var keypoints = ReadKeypoints(output, 0, frame);
        var confidence = keypoints.Average(k => k.Confidence);
        if (confidence < confidenceThreshold)
            return [];

        var box = new BoundingBox(
                keypoints.Min(k => k.X), keypoints.Min(k => k.Y),
                keypoints.Max(k => k.X), keypoints.Max(k => k.Y))
            .Clamp(frame.OriginalWidth, frame.OriginalHeight);

        return [new DetectionResult(ClassId: 0, Label: label, Confidence: confidence, Box: box, Keypoints: keypoints)];
    }

    /// <summary>
    /// Decodes a MultiPose output (<c>[1, instances, 56]</c>, flattened) into one detection per instance whose
    /// box score reaches <paramref name="confidenceThreshold"/>. The box and its score come from the model.
    /// </summary>
    /// <param name="output">The flattened output; <c>instances * 56</c> values.</param>
    /// <param name="frame">Where the original image sat inside the model input.</param>
    /// <param name="confidenceThreshold">Minimum instance score to keep.</param>
    /// <param name="label">The label to attach, taken from the model's own vocabulary.</param>
    public static List<DetectionResult> DecodeMultiPose(
        IReadOnlyList<float> output,
        DetectorInputFrame frame,
        float confidenceThreshold,
        string label)
    {
        ArgumentNullException.ThrowIfNull(output);
        ArgumentException.ThrowIfNullOrEmpty(label);

        if (output.Count == 0 || output.Count % MultiPoseInstanceLength != 0)
        {
            throw new ArgumentException(
                $"MoveNet MultiPose output must hold a whole number of {MultiPoseInstanceLength}-value instances " +
                $"(17 keypoints of y, x, score, then ymin, xmin, ymax, xmax, score); got {output.Count} values.",
                nameof(output));
        }

        var results = new List<DetectionResult>();
        for (var offset = 0; offset < output.Count; offset += MultiPoseInstanceLength)
        {
            var b = offset + PoseSkeleton.Count * ValuesPerKeypoint;
            var score = output[b + 4];
            if (score < confidenceThreshold)
                continue;

            var box = new BoundingBox(
                    frame.ToOriginalX(output[b + 1]), frame.ToOriginalY(output[b]),
                    frame.ToOriginalX(output[b + 3]), frame.ToOriginalY(output[b + 2]))
                .Clamp(frame.OriginalWidth, frame.OriginalHeight);

            results.Add(new DetectionResult(
                ClassId: 0,
                Label: label,
                Confidence: score,
                Box: box,
                Keypoints: ReadKeypoints(output, offset, frame)));
        }

        return results;
    }

    private static Keypoint[] ReadKeypoints(IReadOnlyList<float> output, int offset, DetectorInputFrame frame)
    {
        var keypoints = new Keypoint[PoseSkeleton.Count];
        for (var k = 0; k < PoseSkeleton.Count; k++)
        {
            var i = offset + k * ValuesPerKeypoint;
            keypoints[k] = new Keypoint(
                X: frame.ToOriginalX(output[i + 1]),
                Y: frame.ToOriginalY(output[i]),
                Confidence: output[i + 2]);
        }

        return keypoints;
    }
}
