namespace LMSupply.Vision;

/// <summary>
/// Defines image preprocessing parameters for a specific model.
/// </summary>
/// <param name="Width">Target image width in pixels.</param>
/// <param name="Height">Target image height in pixels.</param>
/// <param name="Mean">Per-channel mean values for normalization (RGB order).</param>
/// <param name="Std">Per-channel standard deviation values for normalization (RGB order).</param>
/// <param name="ResizeMode">How to resize the image to target dimensions.</param>
/// <param name="ChannelFirst">If true, output tensor is NCHW format; otherwise NHWC.</param>
public record PreprocessProfile(
    int Width,
    int Height,
    float[] Mean,
    float[] Std,
    ResizeMode ResizeMode = ResizeMode.Stretch,
    bool ChannelFirst = true)
{
    /// <summary>
    /// Standard ImageNet normalization profile (224x224).
    /// Used by ViT, ResNet, and most vision models.
    /// </summary>
    public static PreprocessProfile ImageNet { get; } = new(
        Width: 224,
        Height: 224,
        Mean: [0.485f, 0.456f, 0.406f],
        Std: [0.229f, 0.224f, 0.225f],
        ResizeMode: ResizeMode.CenterCrop);

    /// <summary>
    /// CLIP model normalization profile (224x224).
    /// </summary>
    public static PreprocessProfile Clip { get; } = new(
        Width: 224,
        Height: 224,
        Mean: [0.48145466f, 0.4578275f, 0.40821073f],
        Std: [0.26862954f, 0.26130258f, 0.27577711f],
        ResizeMode: ResizeMode.CenterCrop);

    /// <summary>
    /// BLIP model normalization profile (384x384).
    /// </summary>
    public static PreprocessProfile Blip { get; } = new(
        Width: 384,
        Height: 384,
        Mean: [0.485f, 0.456f, 0.406f],
        Std: [0.229f, 0.224f, 0.225f],
        ResizeMode: ResizeMode.Stretch);

    /// <summary>
    /// ViT-GPT2 captioning model profile (224x224).
    /// Uses standard ImageNet normalization.
    /// </summary>
    public static PreprocessProfile ViTGpt2 { get; } = new(
        Width: 224,
        Height: 224,
        Mean: [0.485f, 0.456f, 0.406f],
        Std: [0.229f, 0.224f, 0.225f],
        ResizeMode: ResizeMode.CenterCrop);

    /// <summary>
    /// Florence-2 model profile (768x768).
    /// Uses ImageNet normalization with larger resolution.
    /// </summary>
    public static PreprocessProfile Florence2 { get; } = new(
        Width: 768,
        Height: 768,
        Mean: [0.485f, 0.456f, 0.406f],
        Std: [0.229f, 0.224f, 0.225f],
        ResizeMode: ResizeMode.Stretch);

    /// <summary>
    /// SmolVLM model profile (384x384).
    /// </summary>
    public static PreprocessProfile SmolVLM { get; } = new(
        Width: 384,
        Height: 384,
        Mean: [0.5f, 0.5f, 0.5f],
        Std: [0.5f, 0.5f, 0.5f],
        ResizeMode: ResizeMode.Stretch);

    /// <summary>
    /// RT-DETR object detection model profile (640x640).
    /// Uses ImageNet normalization.
    /// </summary>
    public static PreprocessProfile RtDetr { get; } = new(
        Width: 640,
        Height: 640,
        Mean: [0.485f, 0.456f, 0.406f],
        Std: [0.229f, 0.224f, 0.225f],
        ResizeMode: ResizeMode.Stretch);

    /// <summary>
    /// EfficientDet object detection model profile (512x512).
    /// Uses ImageNet normalization.
    /// </summary>
    public static PreprocessProfile EfficientDet { get; } = new(
        Width: 512,
        Height: 512,
        Mean: [0.485f, 0.456f, 0.406f],
        Std: [0.229f, 0.224f, 0.225f],
        ResizeMode: ResizeMode.Stretch);

    /// <summary>
    /// SegFormer semantic segmentation model profile (512x512).
    /// Uses ImageNet normalization.
    /// </summary>
    public static PreprocessProfile SegFormer { get; } = new(
        Width: 512,
        Height: 512,
        Mean: [0.485f, 0.456f, 0.406f],
        Std: [0.229f, 0.224f, 0.225f],
        ResizeMode: ResizeMode.Stretch);

    /// <summary>
    /// SegFormer ADE20K profile with standard 512x512 input.
    /// </summary>
    public static PreprocessProfile SegFormerAde20k { get; } = new(
        Width: 512,
        Height: 512,
        Mean: [0.485f, 0.456f, 0.406f],
        Std: [0.229f, 0.224f, 0.225f],
        ResizeMode: ResizeMode.Stretch);
}

/// <summary>
/// Specifies how to resize an image to target dimensions.
/// </summary>
public enum ResizeMode
{
    /// <summary>
    /// Stretch the image to fit exactly, may distort aspect ratio.
    /// </summary>
    Stretch,

    /// <summary>
    /// Resize to fit within bounds while maintaining aspect ratio, then pad.
    /// </summary>
    Fit,

    /// <summary>
    /// Resize to fill bounds while maintaining aspect ratio, then center crop.
    /// </summary>
    CenterCrop,

    /// <summary>
    /// Resize the shorter edge to target size, then center crop the longer edge.
    /// </summary>
    ShortEdgeCrop
}
