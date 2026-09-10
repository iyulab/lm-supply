namespace LMSupply.Detector.Core;

/// <summary>
/// The raw outputs of one YuNet stride branch, flattened in the order ONNX Runtime returns them.
/// </summary>
/// <param name="Cls">Class score per anchor. Length <c>anchors</c>.</param>
/// <param name="Obj">Objectness per anchor. Length <c>anchors</c>.</param>
/// <param name="Bbox">Box offsets per anchor, four per anchor. Length <c>anchors * 4</c>.</param>
/// <param name="Kps">Landmark offsets per anchor, ten per anchor. Length <c>anchors * 10</c>.</param>
internal readonly record struct YuNetStrideOutput(
    IReadOnlyList<float> Cls,
    IReadOnlyList<float> Obj,
    IReadOnlyList<float> Bbox,
    IReadOnlyList<float> Kps);

/// <summary>
/// Decodes YuNet's twelve outputs into detections. Pure arithmetic over the returned tensors, so it is
/// verifiable without loading the model.
/// </summary>
/// <remarks>
/// The model is anchor-free and emits one branch per stride. For stride <c>s</c> the feature map is
/// <c>(inputSize / s)</c> square and anchor <c>i</c> sits at column <c>i % width</c>, row <c>i / width</c>.
/// The box is that cell's position plus a learned offset, with the extent stored as a logarithm; the score
/// is the geometric mean of the class score and the objectness, which is how the reference implementation
/// combines the two heads.
/// </remarks>
internal static class YuNetDecoder
{
    /// <summary>The strides the exported model branches on, finest first.</summary>
    public static IReadOnlyList<int> Strides { get; } = [8, 16, 32];

    /// <summary>Landmarks per face: right eye, left eye, nose tip, right mouth corner, left mouth corner.</summary>
    public const int LandmarkCount = 5;

    /// <summary>
    /// Decodes every stride branch into detections above <paramref name="confidenceThreshold"/>, in the
    /// original image's pixel coordinates. Non-maximum suppression is deliberately not applied here: YuNet
    /// emits several anchors per face and the caller suppresses them.
    /// </summary>
    /// <param name="strideOutputs">Outputs keyed by stride. Every entry of <see cref="Strides"/> must be present.</param>
    /// <param name="inputSize">The square input size the model was run at.</param>
    /// <param name="originalWidth">Width of the image before it was resized to the input size.</param>
    /// <param name="originalHeight">Height of the image before it was resized to the input size.</param>
    /// <param name="confidenceThreshold">Minimum combined score to keep.</param>
    /// <param name="label">The label to attach, taken from the model's own vocabulary.</param>
    public static List<DetectionResult> Decode(
        IReadOnlyDictionary<int, YuNetStrideOutput> strideOutputs,
        int inputSize,
        int originalWidth,
        int originalHeight,
        float confidenceThreshold,
        string label)
    {
        ArgumentNullException.ThrowIfNull(strideOutputs);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(inputSize);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(originalWidth);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(originalHeight);
        ArgumentException.ThrowIfNullOrEmpty(label);

        var results = new List<DetectionResult>();
        var scaleX = originalWidth / (float)inputSize;
        var scaleY = originalHeight / (float)inputSize;

        foreach (var stride in Strides)
        {
            if (!strideOutputs.TryGetValue(stride, out var branch))
            {
                throw new ArgumentException(
                    $"YuNet output for stride {stride} is missing; the model must emit one cls/obj/bbox/kps group per stride.",
                    nameof(strideOutputs));
            }

            DecodeStride(branch, stride, inputSize, scaleX, scaleY,
                originalWidth, originalHeight, confidenceThreshold, label, results);
        }

        return results;
    }

    private static void DecodeStride(
        YuNetStrideOutput branch,
        int stride,
        int inputSize,
        float scaleX,
        float scaleY,
        int originalWidth,
        int originalHeight,
        float confidenceThreshold,
        string label,
        List<DetectionResult> results)
    {
        var gridWidth = inputSize / stride;
        var anchors = branch.Cls.Count;

        if (branch.Obj.Count != anchors ||
            branch.Bbox.Count != anchors * 4 ||
            branch.Kps.Count != anchors * LandmarkCount * 2)
        {
            throw new ArgumentException(
                $"YuNet stride {stride} outputs disagree on anchor count: cls={anchors}, " +
                $"obj={branch.Obj.Count}, bbox={branch.Bbox.Count}, kps={branch.Kps.Count}.",
                nameof(branch));
        }

        for (var i = 0; i < anchors; i++)
        {
            var score = MathF.Sqrt(Clamp01(branch.Cls[i]) * Clamp01(branch.Obj[i]));
            if (score < confidenceThreshold)
                continue;

            var column = i % gridWidth;
            var row = i / gridWidth;
            var b = i * 4;

            var centerX = (column + branch.Bbox[b]) * stride;
            var centerY = (row + branch.Bbox[b + 1]) * stride;
            var width = MathF.Exp(branch.Bbox[b + 2]) * stride;
            var height = MathF.Exp(branch.Bbox[b + 3]) * stride;

            var box = new BoundingBox(
                    (centerX - width / 2) * scaleX,
                    (centerY - height / 2) * scaleY,
                    (centerX + width / 2) * scaleX,
                    (centerY + height / 2) * scaleY)
                .Clamp(originalWidth, originalHeight);

            // YuNet publishes no per-landmark confidence, so the detection's own score is carried across
            // rather than inventing a number that would read as independent evidence.
            var k = i * LandmarkCount * 2;
            var landmarks = new Keypoint[LandmarkCount];
            for (var n = 0; n < LandmarkCount; n++)
            {
                landmarks[n] = new Keypoint(
                    (column + branch.Kps[k + n * 2]) * stride * scaleX,
                    (row + branch.Kps[k + n * 2 + 1]) * stride * scaleY,
                    score);
            }

            results.Add(new DetectionResult(
                ClassId: 0,
                Label: label,
                Confidence: score,
                Box: box,
                Keypoints: landmarks));
        }
    }

    private static float Clamp01(float value) => Math.Clamp(value, 0f, 1f);
}
