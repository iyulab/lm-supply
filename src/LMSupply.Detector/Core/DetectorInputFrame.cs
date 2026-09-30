namespace LMSupply.Detector.Core;

/// <summary>
/// Where the original image sits inside the model input: the size it was resized to and how far it was offset
/// by padding. Coordinates a model emits relative to its input are mapped back to the original image through
/// this, and nothing else.
/// </summary>
/// <remarks>
/// A stretched input is the degenerate case - the content fills the input, with no padding and a different
/// scale per axis. A padded input keeps the aspect ratio and centres the image. Decoding a padded input as if it
/// had been stretched moves every point towards the padded edge, silently; carrying the geometry from the
/// preprocessing step to the decoder is what keeps the two in agreement.
/// </remarks>
/// <param name="InputWidth">Width of the model input, in pixels.</param>
/// <param name="InputHeight">Height of the model input, in pixels.</param>
/// <param name="ContentWidth">Width the original image was resized to inside the input.</param>
/// <param name="ContentHeight">Height the original image was resized to inside the input.</param>
/// <param name="PadX">Input pixels of padding to the left of the image.</param>
/// <param name="PadY">Input pixels of padding above the image.</param>
/// <param name="OriginalWidth">Width of the original image.</param>
/// <param name="OriginalHeight">Height of the original image.</param>
internal readonly record struct DetectorInputFrame(
    int InputWidth,
    int InputHeight,
    int ContentWidth,
    int ContentHeight,
    int PadX,
    int PadY,
    int OriginalWidth,
    int OriginalHeight)
{
    /// <summary>
    /// The frame of an image stretched to fill the input.
    /// </summary>
    public static DetectorInputFrame Stretched(int inputWidth, int inputHeight, int originalWidth, int originalHeight)
    {
        Validate(inputWidth, inputHeight, originalWidth, originalHeight);
        return new(inputWidth, inputHeight, inputWidth, inputHeight, 0, 0, originalWidth, originalHeight);
    }

    /// <summary>
    /// The frame of an image resized with its aspect ratio kept and centred on an input-sized canvas. The resized
    /// size is rounded to whole pixels and, when the padding is odd, the extra pixel goes to the right or bottom.
    /// </summary>
    public static DetectorInputFrame Padded(int inputWidth, int inputHeight, int originalWidth, int originalHeight)
    {
        Validate(inputWidth, inputHeight, originalWidth, originalHeight);

        var scale = Math.Min(inputWidth / (double)originalWidth, inputHeight / (double)originalHeight);
        var contentWidth = Math.Clamp((int)Math.Round(originalWidth * scale), 1, inputWidth);
        var contentHeight = Math.Clamp((int)Math.Round(originalHeight * scale), 1, inputHeight);

        return new(inputWidth, inputHeight, contentWidth, contentHeight,
            (inputWidth - contentWidth) / 2, (inputHeight - contentHeight) / 2,
            originalWidth, originalHeight);
    }

    /// <summary>
    /// Maps an x coordinate normalised to the input (<c>0..1</c> across its width) to original pixels. The scale
    /// is taken from the rounded content size, so the mapping matches the pixels actually written.
    /// </summary>
    public float ToOriginalX(float normalisedX) =>
        (normalisedX * InputWidth - PadX) * OriginalWidth / ContentWidth;

    /// <summary>
    /// Maps a y coordinate normalised to the input (<c>0..1</c> down its height) to original pixels.
    /// </summary>
    public float ToOriginalY(float normalisedY) =>
        (normalisedY * InputHeight - PadY) * OriginalHeight / ContentHeight;

    private static void Validate(int inputWidth, int inputHeight, int originalWidth, int originalHeight)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(inputWidth);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(inputHeight);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(originalWidth);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(originalHeight);
    }
}
