namespace LMSupply.Console.Host.Models.Requests;

/// <summary>
/// 번역 요청
/// </summary>
public sealed record TranslateRequest
{
    /// <summary>
    /// 모델 ID (예: "default", "ko-en", "ja-en")
    /// </summary>
    public string ModelId { get; init; } = "default";

    /// <summary>
    /// 번역할 텍스트 (단일)
    /// </summary>
    public string? Text { get; init; }

    /// <summary>
    /// 번역할 텍스트 목록 (배치)
    /// </summary>
    public IReadOnlyList<string>? Texts { get; init; }
}
