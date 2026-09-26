using System.Text.Json.Serialization;

namespace LMSupply.Synthesizer.Core;

/// <summary>Source-generated metadata for the VITS model config, so trimmed/AOT hosts can load a voice.</summary>
[JsonSerializable(typeof(VitsConfig))]
internal sealed partial class SynthesizerJsonContext : JsonSerializerContext;
