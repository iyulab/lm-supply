namespace LMSupply.Detector.Models;

/// <summary>
/// How a detector model expects its input tensor to be built from image pixels.
/// </summary>
/// <remarks>
/// <para>
/// Preprocessing was previously fixed for every model - RGB, divided by 255, then shifted by the ImageNet
/// mean and standard deviation. A model that wants anything else cannot simply be registered, and the
/// failure is silent: feeding YuNet RGB where it expects BGR raises nothing and produces an empty result,
/// which is indistinguishable from a photograph with no faces in it.
/// </para>
/// <para>
/// Values are numbered explicitly for the same reason as <see cref="DetectorOutputLayout"/>.
/// </para>
/// </remarks>
public enum DetectorInputFormat
{
    /// <summary>
    /// RGB channel order, pixel values scaled to <c>0..1</c>, no mean/standard-deviation shift.
    /// This is what the RT-DETR reference preprocessing does.
    /// </summary>
    ScaledRgb = 1,

    /// <summary>
    /// BGR channel order, raw <c>0..255</c> pixel values, no scaling and no shift. This is what OpenCV's
    /// default <c>blobFromImage</c> produces, and what YuNet was exported against.
    /// </summary>
    RawBgr = 2
}
