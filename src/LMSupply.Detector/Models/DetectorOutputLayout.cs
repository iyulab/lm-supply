namespace LMSupply.Detector.Models;

/// <summary>
/// The shape of a detector model's raw ONNX output, which decides how it is decoded.
/// </summary>
/// <remarks>
/// <para>
/// Post-processing used to be selected from two independent booleans - "does it need NMS" and "how many
/// keypoints does it have". That pair can describe a plain detector and a pose model, and nothing else: a
/// head with three stride-specific branches has no representation in it, and the nearest combination
/// (needs NMS, has keypoints) routes to the pose decoder, which reads a single tensor of a different shape
/// and returns garbage rather than an error.
/// </para>
/// <para>
/// Naming the layout instead makes the wrong dispatch unrepresentable, and lets
/// <see cref="DetectorModelInfo.RequiresNms"/> and <see cref="DetectorModelInfo.NumKeypoints"/> be derived
/// from it so the description cannot contradict the decoder.
/// </para>
/// <para>
/// Values are numbered explicitly: this enum is persisted in the user's <c>aliases.json</c>, so a member
/// added or removed in the middle must not shift the meaning of a number already written to disk.
/// </para>
/// </remarks>
public enum DetectorOutputLayout
{
    /// <summary>
    /// RT-DETR family. NMS-free - the model emits final detections, either as named
    /// <c>labels</c>/<c>boxes</c>/<c>scores</c> outputs or as <c>logits</c> plus <c>boxes</c>.
    /// </summary>
    RtDetr = 1,

    /// <summary>
    /// YOLO-style detection: one tensor of <c>[cx, cy, w, h, class scores...]</c> per box, requiring NMS.
    /// </summary>
    YoloDetect = 2,

    /// <summary>
    /// YOLO-style pose: one tensor of <c>[cx, cy, w, h, confidence, (x, y, confidence) x 17]</c>, requiring NMS.
    /// </summary>
    YoloPose = 3,

    /// <summary>
    /// YuNet face detection: twelve outputs, one <c>cls</c>/<c>obj</c>/<c>bbox</c>/<c>kps</c> group per
    /// stride (8, 16, 32), anchor-free, five landmarks, requiring NMS.
    /// </summary>
    YuNet = 4,

    /// <summary>
    /// Licence-plate YuNet: three outputs (<c>loc</c>, <c>conf</c>, <c>iou</c>) against generated prior
    /// boxes, emitting a quadrilateral rather than an upright box, requiring NMS. Despite the shared name
    /// this is not <see cref="YuNet"/> with different weights - the head is a different design.
    /// </summary>
    YuNetPlate = 5
}
