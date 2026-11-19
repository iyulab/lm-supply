using System.Text.Json.Serialization;

namespace LMSupply.Console.Host.Models.OpenAI;

/// <summary>
/// Image caption response using choices[] pattern
/// </summary>
public sealed record CaptionResponse
{
    public required string Id { get; init; }
    [JsonPropertyName("object")]
    public string ObjectType { get; init; } = "image.caption";
    public long Created { get; init; } = DateTimeOffset.UtcNow.ToUnixTimeSeconds();
    public required string Model { get; init; }
    public required IReadOnlyList<CaptionChoice> Choices { get; init; }
    public ImageTokenUsage? Usage { get; init; }
}

public sealed record CaptionChoice
{
    public int Index { get; init; }
    public required string Caption { get; init; }
    public float? Confidence { get; init; }
    [JsonPropertyName("finish_reason")]
    public string FinishReason { get; init; } = "stop";
}

public sealed record ImageTokenUsage
{
    [JsonPropertyName("image_tokens")]
    public int ImageTokens { get; init; } = 196;
}

/// <summary>
/// Visual QA response
/// </summary>
public sealed record VqaResponse
{
    public required string Id { get; init; }
    public required string Model { get; init; }
    public required string Question { get; init; }
    public required string Answer { get; init; }
    public float? Confidence { get; init; }
}

/// <summary>
/// OCR response with block/line/word hierarchy
/// </summary>
public sealed record OcrResponse
{
    public required string Id { get; init; }
    public required string Model { get; init; }
    [JsonPropertyName("full_text")]
    public required string FullText { get; init; }
    public required IReadOnlyList<Infrastructure.Vision.OcrPage> Pages { get; init; }
}

/// <summary>
/// Bounding box coordinates
/// </summary>
public sealed record BoundingBox
{
    public float X { get; init; }
    public float Y { get; init; }
    public float Width { get; init; }
    public float Height { get; init; }
}

/// <summary>
/// Object detection response
/// </summary>
public sealed record DetectionResponse
{
    public required string Id { get; init; }
    public required string Model { get; init; }
    public required IReadOnlyList<DetectedObject> Objects { get; init; }
}

/// <summary>
/// Detected object
/// </summary>
public sealed record DetectedObject
{
    public required string Label { get; init; }
    public float Confidence { get; init; }
    [JsonPropertyName("bounding_box")]
    public required BoundingBox BoundingBox { get; init; }
}

/// <summary>
/// Segmentation response
/// </summary>
public sealed record SegmentationResponse
{
    public required string Id { get; init; }
    public required string Model { get; init; }
    public required IReadOnlyList<Segment> Segments { get; init; }
}

/// <summary>
/// Image segment
/// </summary>
public sealed record Segment
{
    public int Id { get; init; }
    public string? Label { get; init; }
    public float? Score { get; init; }
    [JsonPropertyName("bounding_box")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public BoundingBox? BoundingBox { get; init; }
    [JsonPropertyName("mask")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public SegmentMask? Mask { get; init; }
}

/// <summary>
/// RLE or raw segment mask
/// </summary>
public sealed record SegmentMask
{
    public required string Format { get; init; }
    public required int[] Size { get; init; }
    public required string Counts { get; init; }
}
