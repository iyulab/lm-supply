namespace LMSupply.Transcriber.Models;

/// <summary>
/// <see cref="TranscriberModelInfo.Architecture"/> 값. 문자열 계약은 그대로 두되(사용자 별칭 파일이 이 값을 쓴다),
/// 코드 안에서는 매직 문자열 대신 이 상수로 디스패치한다.
/// </summary>
internal static class TranscriberArchitectures
{
    /// <summary>OpenAI Whisper 계열 — encoder/decoder ONNX + byte-level BPE, 30 s 창.</summary>
    public const string Whisper = "Whisper";

    /// <summary>NVIDIA Parakeet TDT 계열 — Conformer 인코더 + Token-and-Duration Transducer, NeMo 128-mel 전처리 ONNX, SentencePiece.</summary>
    public const string ParakeetTdt = "ParakeetTdt";

    public static bool IsParakeetTdt(string? architecture)
        => string.Equals(architecture, ParakeetTdt, StringComparison.OrdinalIgnoreCase);
}
