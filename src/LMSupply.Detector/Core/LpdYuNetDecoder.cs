namespace LMSupply.Detector.Core;

/// <summary>
/// One prior box: a normalised centre and a normalised extent, against which the model's offsets are read.
/// </summary>
internal readonly record struct LpdPrior(float CenterX, float CenterY, float ScaleX, float ScaleY);

/// <summary>
/// The raw outputs of the licence-plate YuNet, flattened as ONNX Runtime returns them.
/// </summary>
/// <param name="Loc">Offsets per prior, fourteen per prior.</param>
/// <param name="Conf">Class scores per prior, two per prior (background, plate).</param>
/// <param name="Iou">Predicted overlap quality per prior, one per prior.</param>
internal readonly record struct LpdYuNetOutput(
    IReadOnlyList<float> Loc,
    IReadOnlyList<float> Conf,
    IReadOnlyList<float> Iou);

/// <summary>
/// Decodes the licence-plate YuNet. Despite the shared name this is a different architecture from the face
/// model: prior boxes rather than an anchor-free grid, one flat set of outputs rather than a branch per
/// stride, and a quadrilateral rather than an upright box.
/// </summary>
internal static class LpdYuNetDecoder
{
    /// <summary>Offsets per prior in the <c>loc</c> output.</summary>
    public const int LocStride = 14;

    /// <summary>Corners per plate: top-left, top-right, bottom-right, bottom-left.</summary>
    public const int CornerCount = 4;

    /// <summary>
    /// The pairs of <c>loc</c> columns holding the quadrilateral, in corner order. The remaining columns
    /// hold an upright box and a centre point that the reference decode does not use.
    /// </summary>
    private static ReadOnlySpan<int> CornerColumns => [4, 6, 10, 12];

    /// <summary>The offset scale the model was trained with.</summary>
    private const float LocationVariance = 0.1f;

    private static readonly int[] Steps = [8, 16, 32, 64];

    private static readonly int[][] MinSizes =
    [
        [10, 16, 24],
        [32, 48],
        [64, 96],
        [128, 192, 256]
    ];

    /// <summary>
    /// Generates the prior boxes for an input size. These are a pure function of the input dimensions - the
    /// model does not carry them - so they must be reproduced exactly or every box lands somewhere else.
    /// </summary>
    /// <remarks>
    /// The feature maps come from halving the input five times with the same integer rounding the reference
    /// uses, and the last four of those are the ones with detection heads. For the exported 320x240 input
    /// that is 30x40, 15x20, 7x10 and 3x5, which with three, two, two and three sizes per cell gives 4385
    /// priors - the length the model's outputs actually have.
    /// </remarks>
    public static LpdPrior[] GeneratePriors(int inputWidth, int inputHeight)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(inputWidth);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(inputHeight);

        var secondHeight = (inputHeight + 1) / 2 / 2;
        var secondWidth = (inputWidth + 1) / 2 / 2;

        var maps = new (int Rows, int Columns)[4];
        var rows = secondHeight;
        var columns = secondWidth;

        for (var level = 0; level < maps.Length; level++)
        {
            rows /= 2;
            columns /= 2;
            maps[level] = (rows, columns);
        }

        var priors = new List<LpdPrior>();

        for (var level = 0; level < maps.Length; level++)
        {
            var (mapRows, mapColumns) = maps[level];
            var step = Steps[level];

            for (var row = 0; row < mapRows; row++)
            {
                for (var column = 0; column < mapColumns; column++)
                {
                    foreach (var size in MinSizes[level])
                    {
                        priors.Add(new LpdPrior(
                            (column + 0.5f) * step / inputWidth,
                            (row + 0.5f) * step / inputHeight,
                            (float)size / inputWidth,
                            (float)size / inputHeight));
                    }
                }
            }
        }

        return [.. priors];
    }

    /// <summary>
    /// Decodes every prior scoring above <paramref name="confidenceThreshold"/> into a detection whose box is
    /// the upright hull of the plate's four corners and whose keypoints are those corners, in the original
    /// image's pixel coordinates. Suppression is left to the caller.
    /// </summary>
    /// <remarks>
    /// The corners are kept rather than discarded because a plate photographed from an angle is a
    /// quadrilateral, and the upright hull of a slanted plate covers a good deal that is not plate. A caller
    /// blurring a region wants the hull; one rectifying the plate for reading wants the corners.
    /// </remarks>
    public static List<DetectionResult> Decode(
        LpdYuNetOutput output,
        IReadOnlyList<LpdPrior> priors,
        int originalWidth,
        int originalHeight,
        float confidenceThreshold,
        string label)
    {
        ArgumentNullException.ThrowIfNull(priors);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(originalWidth);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(originalHeight);
        ArgumentException.ThrowIfNullOrEmpty(label);
        ArgumentNullException.ThrowIfNull(output.Loc);
        ArgumentNullException.ThrowIfNull(output.Conf);
        ArgumentNullException.ThrowIfNull(output.Iou);

        if (output.Loc.Count != priors.Count * LocStride ||
            output.Conf.Count != priors.Count * 2 ||
            output.Iou.Count != priors.Count)
        {
            throw new ArgumentException(
                $"Licence-plate YuNet outputs do not match {priors.Count} priors: loc={output.Loc.Count} " +
                $"(expected {priors.Count * LocStride}), conf={output.Conf.Count} " +
                $"(expected {priors.Count * 2}), iou={output.Iou.Count}.",
                nameof(output));
        }

        var results = new List<DetectionResult>();

        for (var i = 0; i < priors.Count; i++)
        {
            // conf holds [background, plate] per prior; the objectness head is a separate overlap estimate,
            // and the reference combines the two as a geometric mean.
            var score = MathF.Sqrt(Math.Clamp(output.Conf[i * 2 + 1], 0f, 1f) * Math.Clamp(output.Iou[i], 0f, 1f));
            if (score < confidenceThreshold)
                continue;

            var prior = priors[i];
            var corners = new Keypoint[CornerCount];
            var minX = float.MaxValue;
            var minY = float.MaxValue;
            var maxX = float.MinValue;
            var maxY = float.MinValue;

            for (var c = 0; c < CornerCount; c++)
            {
                var column = CornerColumns[c];
                var x = (prior.CenterX + output.Loc[i * LocStride + column] * LocationVariance * prior.ScaleX) * originalWidth;
                var y = (prior.CenterY + output.Loc[i * LocStride + column + 1] * LocationVariance * prior.ScaleY) * originalHeight;

                corners[c] = new Keypoint(x, y, score);
                minX = MathF.Min(minX, x);
                minY = MathF.Min(minY, y);
                maxX = MathF.Max(maxX, x);
                maxY = MathF.Max(maxY, y);
            }

            results.Add(new DetectionResult(
                ClassId: 0,
                Label: label,
                Confidence: score,
                Box: new BoundingBox(minX, minY, maxX, maxY).Clamp(originalWidth, originalHeight),
                Keypoints: corners));
        }

        return results;
    }
}
