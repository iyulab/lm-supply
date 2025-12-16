namespace LMSupply.Segmenter;

/// <summary>
/// ADE20K dataset class labels (150 classes).
/// </summary>
public static class Ade20kLabels
{
    /// <summary>
    /// ADE20K 150-class labels.
    /// </summary>
    public static IReadOnlyList<string> Labels { get; } =
    [
        "wall", "building", "sky", "floor", "tree",
        "ceiling", "road", "bed", "windowpane", "grass",
        "cabinet", "sidewalk", "person", "earth", "door",
        "table", "mountain", "plant", "curtain", "chair",
        "car", "water", "painting", "sofa", "shelf",
        "house", "sea", "mirror", "rug", "field",
        "armchair", "seat", "fence", "desk", "rock",
        "wardrobe", "lamp", "bathtub", "railing", "cushion",
        "base", "box", "column", "signboard", "chest of drawers",
        "counter", "sand", "sink", "skyscraper", "fireplace",
        "refrigerator", "grandstand", "path", "stairs", "runway",
        "case", "pool table", "pillow", "screen door", "stairway",
        "river", "bridge", "bookcase", "blind", "coffee table",
        "toilet", "flower", "book", "hill", "bench",
        "countertop", "stove", "palm", "kitchen island", "computer",
        "swivel chair", "boat", "bar", "arcade machine", "hovel",
        "bus", "towel", "light", "truck", "tower",
        "chandelier", "awning", "streetlight", "booth", "television",
        "airplane", "dirt track", "apparel", "pole", "land",
        "bannister", "escalator", "ottoman", "bottle", "buffet",
        "poster", "stage", "van", "ship", "fountain",
        "conveyer belt", "canopy", "washer", "plaything", "swimming pool",
        "stool", "barrel", "basket", "waterfall", "tent",
        "bag", "minibike", "cradle", "oven", "ball",
        "food", "step", "tank", "trade name", "microwave",
        "pot", "animal", "bicycle", "lake", "dishwasher",
        "screen", "blanket", "sculpture", "hood", "sconce",
        "vase", "traffic light", "tray", "ashcan", "fan",
        "pier", "crt screen", "plate", "monitor", "bulletin board",
        "shower", "radiator", "glass", "clock", "flag"
    ];

    /// <summary>
    /// Gets the label for a class ID.
    /// </summary>
    /// <param name="classId">Class ID (0-149).</param>
    /// <returns>Class label or "unknown" if out of range.</returns>
    public static string GetLabel(int classId) =>
        classId >= 0 && classId < Labels.Count ? Labels[classId] : "unknown";
}
