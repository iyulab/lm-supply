namespace LMSupply.Detector.Core;

/// <summary>
/// Greedy non-maximum suppression over detections, applied per class.
/// </summary>
/// <remarks>
/// Kept as a pure static so it can be exercised directly. As a private instance method reading the
/// detector's options it was reachable only by loading a model and running inference, which meant the
/// suppression itself was never covered by a test.
/// </remarks>
internal static class DetectionNms
{
    /// <summary>
    /// Keeps the highest-scoring detection of each overlapping group, comparing only within a class.
    /// </summary>
    /// <param name="detections">Candidate detections. Not modified.</param>
    /// <param name="iouThreshold">Boxes overlapping the kept box by more than this are discarded.</param>
    public static List<DetectionResult> Apply(IReadOnlyList<DetectionResult> detections, float iouThreshold)
    {
        ArgumentNullException.ThrowIfNull(detections);

        var results = new List<DetectionResult>();

        foreach (var group in detections.GroupBy(d => d.ClassId))
        {
            var sorted = group.OrderByDescending(d => d.Confidence).ToList();

            while (sorted.Count > 0)
            {
                var best = sorted[0];
                results.Add(best);
                sorted.RemoveAt(0);
                sorted = sorted.Where(d => best.Box.IoU(d.Box) <= iouThreshold).ToList();
            }
        }

        return results;
    }
}
