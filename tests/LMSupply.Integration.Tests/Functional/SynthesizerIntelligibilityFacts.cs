using LMSupply.Synthesizer;
using LMSupply.Transcriber;

namespace LMSupply.Integration.Tests.Functional;

/// <summary>
/// The synthesizer speaks: what it says, Whisper hears. This is the positive control the other synthesizer facts lack — «audio
/// came out» holds for noise too.
/// </summary>
/// <remarks>
/// <b>Red today, by design.</b> The synthesizer has no text-to-phoneme step (letters are mapped to fixed ids, not to the
/// voices' phoneme ids), so the output is voice-like noise: for «The weather is beautiful today.» Whisper heard «tube of warrior
/// practitioner». These facts turn green when a G2P step lands; that is its completion criterion.
/// </remarks>
[Trait("Category", "Integration")]
[Trait("Category", "Intelligibility")]
public sealed class SynthesizerIntelligibilityFacts
{
    [Theory]
    [InlineData("default", "The weather is beautiful today.", "en")]
    [InlineData("korean", "오늘 날씨가 정말 좋습니다.", "ko")]
    public async Task WhatTheVoiceSays_WhisperHears(string alias, string text, string language)
    {
        await using var synthesizer = await LocalSynthesizer.LoadAsync(alias, cancellationToken: TestContext.Current.CancellationToken);
        var audio = await synthesizer.SynthesizeAsync(text, cancellationToken: TestContext.Current.CancellationToken);
        await using var transcriber = await LocalTranscriber.LoadAsync("default", cancellationToken: TestContext.Current.CancellationToken);

        var heard = await transcriber.TranscribeAsync(audio.ToWavBytes(), new TranscribeOptions { Language = language }, TestContext.Current.CancellationToken);

        var said = Words(text);
        var overlap = said.Intersect(Words(heard.Text)).Count();
        Assert.True(overlap * 2 >= said.Count,
            $"said «{text}», Whisper heard «{heard.Text}» ({audio.DurationSeconds:F1} s) — {overlap} of {said.Count} words");
    }

    private static HashSet<string> Words(string text) =>
        [.. text.ToLowerInvariant().Split([' ', '.', ',', '!', '?'], StringSplitOptions.RemoveEmptyEntries)];
}
