using LMSupply.Detector.Core;

namespace LMSupply.Detector.Models;

/// <summary>
/// The facts that follow from a <see cref="DetectorOutputLayout"/>, stated once.
/// </summary>
/// <remarks>
/// Both <see cref="DetectorModelInfo"/> and the running detector need to know how many keypoints a layout
/// carries and whether it needs suppression. Deriving each independently is how two answers to one question
/// drift apart, so the mapping lives here and both defer to it.
/// </remarks>
public static class DetectorOutputLayoutExtensions
{
    /// <summary>
    /// How many keypoints this layout emits per detection; zero when it emits none.
    /// </summary>
    public static int KeypointCount(this DetectorOutputLayout layout) => layout switch
    {
        DetectorOutputLayout.YoloPose => PoseSkeleton.Count,
        DetectorOutputLayout.YuNet => YuNetDecoder.LandmarkCount,
        _ => 0
    };

    /// <summary>
    /// Whether this layout emits overlapping candidates that must be suppressed.
    /// </summary>
    public static bool RequiresNms(this DetectorOutputLayout layout) =>
        layout is not DetectorOutputLayout.RtDetr;
}
