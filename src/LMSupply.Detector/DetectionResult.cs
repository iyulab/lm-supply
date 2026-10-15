using LMSupply.Vision;
namespace LMSupply.Detector;

/// <summary>
/// Represents a single detected object in an image.
/// </summary>
/// <param name="ClassId">The class ID from COCO or custom dataset (0-indexed).</param>
/// <param name="Label">Human-readable class label (e.g., "person", "car").</param>
/// <param name="Confidence">Detection confidence score (0.0 to 1.0).</param>
/// <param name="Box">Bounding box coordinates.</param>
/// <param name="Keypoints">Optional keypoint data for pose estimation models. Null for standard object detection.</param>
public readonly record struct DetectionResult(
    int ClassId,
    string Label,
    float Confidence,
    BoundingBox Box,
    IReadOnlyList<Keypoint>? Keypoints = null)
{
    /// <summary>
    /// Gets whether this detection includes keypoint (pose) data.
    /// </summary>
    public bool HasKeypoints => Keypoints is { Count: > 0 };

    /// <summary>
    /// Returns a string representation of the detection.
    /// </summary>
    public override string ToString() =>
        HasKeypoints
            ? $"{Label} ({Confidence:P1}) at [{Box.X:F0}, {Box.Y:F0}, {Box.Right:F0}, {Box.Bottom:F0}] ({Keypoints!.Count} keypoints)"
            : $"{Label} ({Confidence:P1}) at [{Box.X:F0}, {Box.Y:F0}, {Box.Right:F0}, {Box.Bottom:F0}]";
}

/// <summary>
/// Represents a single keypoint in a pose estimation result.
/// </summary>
/// <param name="X">X coordinate in original image pixels.</param>
/// <param name="Y">Y coordinate in original image pixels.</param>
/// <param name="Confidence">Keypoint visibility/confidence score (0.0 to 1.0).</param>
public readonly record struct Keypoint(float X, float Y, float Confidence)
{
    /// <summary>
    /// Gets whether this keypoint is considered visible (confidence above threshold).
    /// </summary>
    public bool IsVisible(float threshold = 0.5f) => Confidence >= threshold;

    /// <summary>
    /// Returns a string representation of the keypoint.
    /// </summary>
    public override string ToString() => $"({X:F1}, {Y:F1}, {Confidence:F2})";
}

/// <summary>
/// COCO 17-keypoint skeleton indices for pose estimation.
/// </summary>
public static class PoseSkeleton
{
    public const int Nose = 0;
    public const int LeftEye = 1;
    public const int RightEye = 2;
    public const int LeftEar = 3;
    public const int RightEar = 4;
    public const int LeftShoulder = 5;
    public const int RightShoulder = 6;
    public const int LeftElbow = 7;
    public const int RightElbow = 8;
    public const int LeftWrist = 9;
    public const int RightWrist = 10;
    public const int LeftHip = 11;
    public const int RightHip = 12;
    public const int LeftKnee = 13;
    public const int RightKnee = 14;
    public const int LeftAnkle = 15;
    public const int RightAnkle = 16;

    /// <summary>
    /// Total number of COCO keypoints.
    /// </summary>
    public const int Count = 17;

    /// <summary>
    /// COCO skeleton bone connections (pairs of keypoint indices).
    /// </summary>
    public static IReadOnlyList<(int From, int To)> Bones { get; } =
    [
        (Nose, LeftEye), (Nose, RightEye),
        (LeftEye, LeftEar), (RightEye, RightEar),
        (LeftShoulder, RightShoulder),
        (LeftShoulder, LeftElbow), (RightShoulder, RightElbow),
        (LeftElbow, LeftWrist), (RightElbow, RightWrist),
        (LeftShoulder, LeftHip), (RightShoulder, RightHip),
        (LeftHip, RightHip),
        (LeftHip, LeftKnee), (RightHip, RightKnee),
        (LeftKnee, LeftAnkle), (RightKnee, RightAnkle)
    ];

    /// <summary>
    /// COCO keypoint names.
    /// </summary>
    public static IReadOnlyList<string> Names { get; } =
    [
        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle"
    ];
}
