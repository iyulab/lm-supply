namespace LMSupply.Segmenter.Models;

/// <summary>
/// What a segmentation model's outputs are, which decides how they become a class map.
/// </summary>
public enum SegmenterOutputKind
{
    /// <summary>One output of per-class scores, <c>[1, classes, height, width]</c>; each pixel takes its highest-scoring class.</summary>
    ClassLogits,

    /// <summary>One output with a foreground probability per pixel, <c>[1, 1, height, width]</c> (or <c>[1, height, width, 1]</c>); a pixel above one half is class 1, the rest class 0.</summary>
    ForegroundAlpha,

    /// <summary>
    /// Two outputs of a query-based model (MaskFormer): per-query class scores <c>[1, queries, classes + 1]</c> (the last class is "no object")
    /// and per-query mask scores <c>[1, queries, height, width]</c>; each pixel takes the class with the most mask-weighted query probability.
    /// </summary>
    QueryMasks,
}

/// <summary>
/// How an image is turned into the model's input values.
/// </summary>
public enum SegmenterInputNormalization
{
    /// <summary>Pixels scaled to 0-1, then the ImageNet mean subtracted and divided by its standard deviation.</summary>
    ImageNet,

    /// <summary>Pixels scaled to 0-1 and nothing else.</summary>
    UnitRange,
}
