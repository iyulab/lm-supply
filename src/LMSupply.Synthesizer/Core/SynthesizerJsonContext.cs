using System.Text.Json.Serialization;

namespace LMSupply.Synthesizer.Core;

/// <summary>
/// Source-generated metadata for the voice config, so trimmed/AOT hosts can load a voice. A Piper config names its keys in
/// snake_case (<c>phoneme_id_map</c>, <c>sample_rate</c>, <c>noise_w</c>).
/// </summary>
[JsonSourceGenerationOptions(PropertyNamingPolicy = JsonKnownNamingPolicy.SnakeCaseLower)]
[JsonSerializable(typeof(PiperVoiceConfig))]
internal sealed partial class SynthesizerJsonContext : JsonSerializerContext;
