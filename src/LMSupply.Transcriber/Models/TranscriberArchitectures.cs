namespace LMSupply.Transcriber.Models;

/// <summary>
/// Values of <see cref="TranscriberModelInfo.Architecture"/>. The string contract is unchanged (user alias files use these values),
/// but code dispatches on these constants instead of magic strings.
/// </summary>
internal static class TranscriberArchitectures
{
    /// <summary>OpenAI Whisper family — encoder/decoder ONNX + byte-level BPE, 30 s windows.</summary>
    public const string Whisper = "Whisper";

    /// <summary>NVIDIA Parakeet TDT family — Conformer encoder + Token-and-Duration Transducer, NeMo 128-mel preprocessor ONNX, SentencePiece.</summary>
    public const string ParakeetTdt = "ParakeetTdt";

    public static bool IsParakeetTdt(string? architecture)
        => string.Equals(architecture, ParakeetTdt, StringComparison.OrdinalIgnoreCase);
}
