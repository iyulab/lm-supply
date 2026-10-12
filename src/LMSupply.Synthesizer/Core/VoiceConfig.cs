using System.Diagnostics;
using System.Text.Json;
using LMSupply.Exceptions;
using LMSupply.Synthesizer.Models;

namespace LMSupply.Synthesizer.Core;

/// <summary>
/// What a Piper voice declares about itself in the <c>.onnx.json</c> next to its model: the phonemes it takes and their
/// ids, the inference defaults it was tuned with, its sample rate and its speaker count. The voice is run by this, not by
/// the catalog entry that named it.
/// </summary>
internal sealed class VoiceConfig
{
    // Piper's two phoneme types; a config without the key is "espeak".
    internal const string EspeakType = "espeak";
    internal const string TextType = "text";

    private VoiceConfig(PhonemeScheme scheme, PhonemeIdEncoder encoder, int sampleRate, int numSpeakers,
        float noiseScale, float lengthScale, float noiseWidth)
    {
        Scheme = scheme;
        Encoder = encoder;
        SampleRate = sampleRate;
        NumSpeakers = numSpeakers;
        NoiseScale = noiseScale;
        LengthScale = lengthScale;
        NoiseWidth = noiseWidth;
    }

    public PhonemeScheme Scheme { get; }
    public PhonemeIdEncoder Encoder { get; }
    public int SampleRate { get; }
    public int NumSpeakers { get; }
    public float NoiseScale { get; }
    public float LengthScale { get; }
    public float NoiseWidth { get; }

    /// <summary>
    /// Parses a voice config. <paramref name="catalog"/> supplies only what a config may leave out (sample rate,
    /// language); where both state a value, the config's is used.
    /// </summary>
    public static VoiceConfig Parse(string json, SynthesizerModelInfo catalog)
    {
        PiperVoiceConfig? config;
        try
        {
            config = JsonSerializer.Deserialize(json, SynthesizerJsonContext.Default.PiperVoiceConfig);
        }
        catch (JsonException ex)
        {
            throw new ModelLoadException($"The voice config '{catalog.ConfigFile}' is not valid JSON: {ex.Message}", catalog.Id, ex);
        }

        if (config?.PhonemeIdMap is not { Count: > 0 } idMap)
        {
            throw new ModelLoadException(
                $"The voice config '{catalog.ConfigFile}' has no phoneme_id_map, so there is no way to encode phonemes for this voice.",
                catalog.Id);
        }

        var scheme = new PhonemeScheme
        {
            Type = string.IsNullOrWhiteSpace(config.PhonemeType) ? EspeakType : config.PhonemeType,
            Voice = string.IsNullOrWhiteSpace(config.Espeak?.Voice) ? null : config.Espeak.Voice,
            Language = string.IsNullOrWhiteSpace(config.Language?.Code) ? catalog.Language : config.Language.Code.Replace('_', '-')
        };

        // Piper's defaults for a config without an "inference" section, or without one of its values.
        return new VoiceConfig(
            scheme,
            PhonemeIdEncoder.FromIdMap(idMap, catalog),
            sampleRate: config.Audio?.SampleRate ?? catalog.SampleRate,
            numSpeakers: config.NumSpeakers ?? 1,
            noiseScale: config.Inference?.NoiseScale ?? 0.667f,
            lengthScale: config.Inference?.LengthScale ?? 1.0f,
            noiseWidth: config.Inference?.NoiseW ?? 0.8f);
    }

    /// <summary>
    /// The model's <c>scales</c> input, <c>[noise_scale, length_scale, noise_w]</c>: the caller's value where one is set,
    /// the voice's own default otherwise. <see cref="SynthesizeOptions.Speed"/> is a rate and <c>length_scale</c> a
    /// duration, so the voice's length scale is divided by the speed.
    /// </summary>
    public float[] Scales(SynthesizeOptions? options)
    {
        var speed = options?.Speed ?? 1.0f;
        if (!(speed > 0) || float.IsInfinity(speed))
            throw new ArgumentOutOfRangeException(nameof(options), speed, "SynthesizeOptions.Speed must be a positive, finite number.");

        return [options?.NoiseScale ?? NoiseScale, LengthScale / speed, options?.NoiseWidth ?? NoiseWidth];
    }

    /// <summary>
    /// The model's <c>sid</c> input: <see langword="null"/> for a single-speaker voice (the graph has no such input and
    /// the option is ignored, as Piper does), the caller's speaker otherwise.
    /// </summary>
    public long? SpeakerId(SynthesizeOptions? options)
    {
        if (NumSpeakers <= 1)
            return null;

        var speakerId = options?.SpeakerId ?? 0;
        if (speakerId < 0 || speakerId >= NumSpeakers)
        {
            throw new ArgumentOutOfRangeException(nameof(options), speakerId,
                $"SynthesizeOptions.SpeakerId must be between 0 and {NumSpeakers - 1} for this voice.");
        }

        return speakerId;
    }

    /// <summary>
    /// The catalog entry as the loaded voice actually is. A catalog entry is curated by hand and can be wrong about what
    /// the voice file declares; a difference is traced, and the voice's value is the one reported and used.
    /// </summary>
    public SynthesizerModelInfo Reconcile(SynthesizerModelInfo catalog)
    {
        if (catalog.SampleRate == SampleRate && catalog.NumSpeakers == NumSpeakers)
            return catalog;

        Trace.TraceWarning(
            $"[VoiceConfig] {catalog.AliasName}: the catalog says {catalog.SampleRate} Hz and {catalog.NumSpeakers} speaker(s), " +
            $"the voice's config says {SampleRate} Hz and {NumSpeakers}. Using the config.");
        return catalog with { SampleRate = SampleRate, NumSpeakers = NumSpeakers };
    }
}

/// <summary>The subset of a Piper voice config (<c>*.onnx.json</c>) that decides how the voice is run.</summary>
internal sealed class PiperVoiceConfig
{
    public PiperAudioConfig? Audio { get; set; }
    public PiperEspeakConfig? Espeak { get; set; }
    public PiperLanguageConfig? Language { get; set; }
    public PiperInferenceConfig? Inference { get; set; }
    public string? PhonemeType { get; set; }
    public Dictionary<string, long[]>? PhonemeIdMap { get; set; }
    public int? NumSpeakers { get; set; }
}

internal sealed class PiperAudioConfig
{
    public int? SampleRate { get; set; }
}

internal sealed class PiperEspeakConfig
{
    public string? Voice { get; set; }
}

internal sealed class PiperLanguageConfig
{
    public string? Code { get; set; }
}

internal sealed class PiperInferenceConfig
{
    public float? NoiseScale { get; set; }
    public float? LengthScale { get; set; }
    public float? NoiseW { get; set; }
}
