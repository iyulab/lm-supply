using LMSupply.Synthesizer.Core;
using LMSupply.Synthesizer.Models;

namespace LMSupply.Synthesizer.Tests;

/// <summary>
/// Voice configs for the tests: the default voice's real one (see Fixtures/README.md), and small hand-written ones for
/// what no catalog voice declares. All of them go through the library's loader, so a key the source-generated JSON
/// metadata misses fails here (the suite runs with reflection-based JSON off).
/// </summary>
internal static class VoiceFixture
{
    public static string LjSpeechJson { get; } =
        File.ReadAllText(Path.Combine(AppContext.BaseDirectory, "Fixtures", "en_US-ljspeech-medium.onnx.json"));

    public static VoiceConfig LjSpeech() => VoiceConfig.Parse(LjSpeechJson, SynthesizerModels.EnUsLjSpeech);

    /// <summary>A config with the given top-level members and a minimal id map: pad 0, start 1, end 2, a 3, b 4.</summary>
    public static VoiceConfig Minimal(string members = "", string extraPhonemes = "", SynthesizerModelInfo? catalog = null)
    {
        var json = "{" + (members.Length > 0 ? members + "," : "")
            + "\"phoneme_id_map\": {\"_\": [0], \"^\": [1], \"$\": [2], \"a\": [3], \"b\": [4]"
            + (extraPhonemes.Length > 0 ? ", " + extraPhonemes : "") + "}}";
        return VoiceConfig.Parse(json, catalog ?? SynthesizerModels.EnUsLjSpeech);
    }
}
