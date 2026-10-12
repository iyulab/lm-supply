using LMSupply.Integration.Tests.Helpers;
using LMSupply.Synthesizer;
using LMSupply.Transcriber;

namespace LMSupply.Integration.Tests.Functional;

/// <summary>
/// Functional tests for the Synthesizer (Text→Speech) domain.
/// Uses Piper TTS models. Tests L/I/Q/E axes.
/// Includes cross-domain TTS→STT roundtrip test.
/// Requires GPU + network access. Run locally only.
/// </summary>
/// <remarks>
/// The package has no phonemizer for these voices, so the facts pass phonemes (<see cref="SpokenPhonemes"/>), or text
/// through a phonemizer set in the options; text on its own is refused (<see cref="E_TextWithoutAPhonemizer_IsRefused"/>).
/// </remarks>
[Trait("Category", "Functional")]
[Trait("Category", "LocalOnly")]
[Trait("Domain", "Synthesizer")]
public class SynthesizerFunctionalTests
{
    private static readonly string[] Hello = [SpokenPhonemes.Hello];

    // ── L axis: Model Loading ───────────────────────────────────────

    [Fact]
    [Trait("Axis", "Loading")]
    public async Task L_FastAlias_LoadsSuccessfully()
    {
        await using var model = await LocalSynthesizer.LoadAsync("fast", cancellationToken: TestContext.Current.CancellationToken);

        model.ModelId.Should().NotBeNullOrEmpty();
        model.SampleRate.Should().BeGreaterThan(0);
    }

    [Fact]
    [Trait("Axis", "Loading")]
    public async Task L_FastAlias_ReportsWhatItsConfigDeclares()
    {
        await using var model = await LocalSynthesizer.LoadAsync("fast", cancellationToken: TestContext.Current.CancellationToken);

        // This voice's config names no phoneme type (so it is "espeak") and declares 16 kHz audio.
        model.SampleRate.Should().Be(16000);
        model.PhonemeScheme.Should().Be(new PhonemeScheme { Type = "espeak", Voice = "en-us", Language = "en-US" });
    }

    [Fact]
    [Trait("Axis", "Loading")]
    public async Task L_GetModelInfo_ReturnsValidInfo()
    {
        await using var model = await LocalSynthesizer.LoadAsync("fast", cancellationToken: TestContext.Current.CancellationToken);

        var info = model.GetModelInfo();
        info.Should().NotBeNull();
    }

    // ── I axis: Basic Inference ─────────────────────────────────────

    [Fact]
    [Trait("Axis", "Inference")]
    public async Task I_BasicSynthesis_ReturnsAudioSamples()
    {
        await using var model = await LocalSynthesizer.LoadAsync("fast", cancellationToken: TestContext.Current.CancellationToken);

        var result = await model.SynthesizePhonemesAsync(Hello, cancellationToken: TestContext.Current.CancellationToken);

        result.Should().NotBeNull();
        result.AudioSamples.Should().NotBeEmpty();
        result.SampleRate.Should().BeGreaterThan(0);
        result.UnknownPhonemes.Should().BeEmpty();
    }

    [Fact]
    [Trait("Axis", "Inference")]
    public async Task I_ToWavBytes_ProducesValidWav()
    {
        await using var model = await LocalSynthesizer.LoadAsync("fast", cancellationToken: TestContext.Current.CancellationToken);

        var result = await model.SynthesizePhonemesAsync(Hello, cancellationToken: TestContext.Current.CancellationToken);
        var wavBytes = result.ToWavBytes();

        // Validate WAV header
        wavBytes.Length.Should().BeGreaterThan(44, "WAV should have header + data");
        wavBytes[0].Should().Be((byte)'R');
        wavBytes[1].Should().Be((byte)'I');
        wavBytes[2].Should().Be((byte)'F');
        wavBytes[3].Should().Be((byte)'F');
    }

    // ── Q axis: Quality ─────────────────────────────────────────────

    [Fact]
    [Trait("Axis", "Quality")]
    public async Task Q_AudioDuration_IsPositive()
    {
        await using var model = await LocalSynthesizer.LoadAsync("fast", cancellationToken: TestContext.Current.CancellationToken);

        var result = await model.SynthesizePhonemesAsync(Hello, cancellationToken: TestContext.Current.CancellationToken);

        result.DurationSeconds.Should().BeGreaterThan(0, "synthesized audio should have positive duration");
    }

    [Fact]
    [Trait("Axis", "Quality")]
    public async Task Q_AudioSamples_AreInValidRange()
    {
        await using var model = await LocalSynthesizer.LoadAsync("fast", cancellationToken: TestContext.Current.CancellationToken);

        var result = await model.SynthesizePhonemesAsync(Hello, cancellationToken: TestContext.Current.CancellationToken);

        result.AudioSamples.Should().AllSatisfy(sample =>
            sample.Should().BeInRange(-1.1f, 1.1f, "samples should be approximately in [-1, 1]"));
    }

    [Fact]
    [Trait("Axis", "Quality")]
    public async Task Q_MorePhonemes_ProduceLongerAudio()
    {
        await using var model = await LocalSynthesizer.LoadAsync("fast", cancellationToken: TestContext.Current.CancellationToken);

        var shortResult = await model.SynthesizePhonemesAsync(["hˈaɪ."], cancellationToken: TestContext.Current.CancellationToken);
        var longResult = await model.SynthesizePhonemesAsync([SpokenPhonemes.Fox], cancellationToken: TestContext.Current.CancellationToken);

        longResult.AudioSamples.Length.Should().BeGreaterThan(shortResult.AudioSamples.Length,
            "a longer sentence should produce more audio samples");
    }

    // ── E axis: Edge Cases ──────────────────────────────────────────

    [Fact]
    [Trait("Axis", "EdgeCase")]
    public async Task E_TextWithoutAPhonemizer_IsRefused()
    {
        await using var model = await LocalSynthesizer.LoadAsync("fast", cancellationToken: TestContext.Current.CancellationToken);

        var act = () => model.SynthesizeAsync("2024 January 1st, $100.00", cancellationToken: TestContext.Current.CancellationToken);

        await act.Should().ThrowAsync<NotSupportedException>().WithMessage("*'espeak'*");
    }

    [Fact]
    [Trait("Axis", "EdgeCase")]
    public async Task E_PhonemesTheVoiceDoesNotKnow_AreSkippedAndReported()
    {
        await using var model = await LocalSynthesizer.LoadAsync("fast", cancellationToken: TestContext.Current.CancellationToken);

        // '%' and '@' are not phonemes of this voice.
        var result = await model.SynthesizePhonemesAsync(["hˈaɪ% @."], cancellationToken: TestContext.Current.CancellationToken);

        result.AudioSamples.Should().NotBeEmpty();
        result.UnknownPhonemes.Should().Be("%@");
    }

    [Fact]
    [Trait("Axis", "EdgeCase")]
    public async Task E_ManySentences_DoNotOOM()
    {
        await using var model = await LocalSynthesizer.LoadAsync("fast", cancellationToken: TestContext.Current.CancellationToken);

        var sentences = Enumerable.Repeat(SpokenPhonemes.Hello, 50).ToList();
        var act = () => model.SynthesizePhonemesAsync(sentences, cancellationToken: TestContext.Current.CancellationToken);
        await act.Should().NotThrowAsync("synthesizing many sentences should not OOM");
    }

    // ── Cross-Domain: TTS → STT Roundtrip ───────────────────────────

    [Fact]
    [Trait("Axis", "CrossDomain")]
    [Trait("Scenario", "TTS-STT-Roundtrip")]
    public async Task CrossDomain_TtsToStt_ProducesRecognizableText()
    {
        // 1. Synthesize speech
        await using var synthesizer = await LocalSynthesizer.LoadAsync("fast", cancellationToken: TestContext.Current.CancellationToken);
        var synthesis = await synthesizer.SynthesizePhonemesAsync(Hello, cancellationToken: TestContext.Current.CancellationToken);

        synthesis.AudioSamples.Should().NotBeEmpty("synthesis should produce audio");

        // 2. Convert to WAV
        var wavBytes = synthesis.ToWavBytes();

        // 3. Transcribe the synthesized audio
        await using var transcriber = await LocalTranscriber.LoadAsync("fast", cancellationToken: TestContext.Current.CancellationToken);
        var transcription = await transcriber.TranscribeAsync(wavBytes, cancellationToken: TestContext.Current.CancellationToken);

        // 4. Verify the transcription is not empty; whether it matches is SynthesizerIntelligibilityFacts' question
        transcription.Text.Should().NotBeNullOrEmpty(
            "transcriber should recognize synthesized speech");
    }

    // ── Gap Tests: Default Alias, Voice, Models, Properties ────────

    [Fact]
    [Trait("Axis", "Loading")]
    public async Task L_DefaultAlias_LoadsSuccessfully()
    {
        await using var model = await LocalSynthesizer.LoadAsync("default", cancellationToken: TestContext.Current.CancellationToken);

        model.ModelId.Should().NotBeNullOrEmpty();
    }

    [Fact]
    [Trait("Axis", "Loading")]
    public void L_GetAvailableModels_ReturnsAliases()
    {
        var models = LocalSynthesizer.GetAvailableModels().ToList();

        models.Should().NotBeEmpty();
        models.Should().Contain("fast");
    }

    [Fact]
    [Trait("Axis", "Loading")]
    public void L_GetAllModels_ReturnsModelInfo()
    {
        var models = LocalSynthesizer.GetAllModels().ToList();

        models.Should().NotBeEmpty();
        models.Should().AllSatisfy(m => m.Should().NotBeNull());
    }

    [Fact]
    [Trait("Axis", "Loading")]
    public async Task L_Voice_IsPopulated()
    {
        await using var model = await LocalSynthesizer.LoadAsync("fast", cancellationToken: TestContext.Current.CancellationToken);

        model.Voice.Should().NotBeNullOrEmpty("model should have a voice identifier");
    }

    [Fact]
    [Trait("Axis", "Loading")]
    public async Task L_ModelProperties_ArePopulated()
    {
        await using var model = await LocalSynthesizer.LoadAsync("fast", cancellationToken: TestContext.Current.CancellationToken);

        model.ActiveProviders.Should().NotBeNull();
        model.RequestedProvider.Should().BeDefined();
    }

    [Fact]
    [Trait("Axis", "Inference")]
    public async Task I_SynthesizeWithSpeakerId_Works()
    {
        await using var model = await LocalSynthesizer.LoadAsync("fast", cancellationToken: TestContext.Current.CancellationToken);

        var options = new SynthesizeOptions { SpeakerId = 0 };
        var result = await model.SynthesizePhonemesAsync(Hello, options, TestContext.Current.CancellationToken);

        result.Should().NotBeNull();
        result.AudioSamples.Should().NotBeEmpty();
    }

    [Fact]
    [Trait("Axis", "Inference")]
    public async Task I_SynthesizeToStream_Works()
    {
        var options = new SynthesizerOptions { Phonemizer = new SpokenPhonemes.Phonemizer() };
        await using var model = await LocalSynthesizer.LoadAsync("fast", options, cancellationToken: TestContext.Current.CancellationToken);

        using var stream = new MemoryStream();
        await model.SynthesizeToStreamAsync(SpokenPhonemes.HelloText, stream, cancellationToken: TestContext.Current.CancellationToken);

        stream.Length.Should().BeGreaterThan(0, "stream should contain audio data");
    }

    [Fact]
    [Trait("Axis", "Inference")]
    public async Task I_Streaming_YieldsOneChunkPerSentence()
    {
        await using var model = await LocalSynthesizer.LoadAsync("fast", cancellationToken: TestContext.Current.CancellationToken);

        var chunks = new List<AudioChunk>();
        await foreach (var chunk in model.SynthesizePhonemesStreamingAsync(
            [SpokenPhonemes.Hello, SpokenPhonemes.Weather], cancellationToken: TestContext.Current.CancellationToken))
        {
            chunks.Add(chunk);
        }

        chunks.Select(c => c.Index).Should().Equal(0, 1);
        chunks.Select(c => c.IsFinal).Should().Equal(false, true);
        chunks.Should().AllSatisfy(c => c.Samples.Should().NotBeEmpty());
    }

    [Fact]
    [Trait("Axis", "EdgeCase")]
    public async Task E_NoPhonemes_ProduceNoAudio()
    {
        await using var model = await LocalSynthesizer.LoadAsync("fast", cancellationToken: TestContext.Current.CancellationToken);

        var result = await model.SynthesizePhonemesAsync([], cancellationToken: TestContext.Current.CancellationToken);

        result.AudioSamples.Should().BeEmpty();
    }

    [Fact]
    [Trait("Axis", "EdgeCase")]
    public async Task E_PunctuationOnly_DoesNotCrash()
    {
        await using var model = await LocalSynthesizer.LoadAsync("fast", cancellationToken: TestContext.Current.CancellationToken);

        var act = () => model.SynthesizePhonemesAsync(["..., ???, !!!"], cancellationToken: TestContext.Current.CancellationToken);
        await act.Should().NotThrowAsync("punctuation-only phonemes should not crash");
    }

    // ── Static Tests: Model Registry Validation (no model loading) ─

    [Fact]
    [Trait("Axis", "Loading")]
    public void L_GetAvailableModels_ContainsAllStandardAliases()
    {
        var models = LocalSynthesizer.GetAvailableModels().ToList();

        models.Should().Contain("default");
        models.Should().Contain("fast");
        models.Should().Contain("quality");
        models.Should().Contain("auto");
    }

    [Fact]
    [Trait("Axis", "Loading")]
    public void L_GetAllModels_HasExpectedCount()
    {
        var models = LocalSynthesizer.GetAllModels().ToList();

        models.Count.Should().BeGreaterThanOrEqualTo(5,
            "registry should have at least 5 voice models");
    }

    [Fact]
    [Trait("Axis", "Loading")]
    public void L_GetAllModels_AllHaveRequiredFields()
    {
        var models = LocalSynthesizer.GetAllModels().ToList();

        models.Should().AllSatisfy(m =>
        {
            m.Id.Should().NotBeNullOrEmpty("model Id is required");
            m.AliasName.Should().NotBeNullOrEmpty("AliasName is required");
            m.DisplayName.Should().NotBeNullOrEmpty("DisplayName is required");
            m.SampleRate.Should().BeGreaterThan(0, "SampleRate should be > 0");
            m.Language.Should().NotBeNullOrEmpty("Language is required");
        });
    }

    [Fact]
    [Trait("Axis", "Loading")]
    public void L_GetAllModels_ContainsMultilingualVoices()
    {
        var models = LocalSynthesizer.GetAllModels().ToList();
        var languages = models.Select(m => m.Language).Distinct().ToList();

        languages.Count.Should().BeGreaterThanOrEqualTo(2,
            "registry should have voices for at least 2 languages");
    }

    [Fact]
    [Trait("Axis", "Quality")]
    public void Q_SynthesizeOptions_Defaults_AreCorrect()
    {
        var opts = new SynthesizeOptions();

        opts.Speed.Should().BeApproximately(1.0f, 0.01f,
            "default speed should be 1.0");
        opts.SpeakerId.Should().Be(0,
            "default speaker ID should be 0");
    }

    // ── Model-Loading Tests: Quality, Korean, Speed ──────────────

    [Fact]
    [Trait("Axis", "Loading")]
    public async Task L_QualityAlias_LoadsSuccessfully()
    {
        await using var model = await LocalSynthesizer.LoadAsync("quality", cancellationToken: TestContext.Current.CancellationToken);

        model.ModelId.Should().NotBeNullOrEmpty();
        model.SampleRate.Should().BeGreaterThan(0);
    }

    [Fact]
    [Trait("Axis", "Loading")]
    public async Task L_AutoAlias_LoadsSuccessfully()
    {
        await using var model = await LocalSynthesizer.LoadAsync("auto", cancellationToken: TestContext.Current.CancellationToken);

        model.ModelId.Should().NotBeNullOrEmpty();
    }

    [Fact]
    [Trait("Axis", "Inference")]
    public async Task I_SpeedOption_AffectsDuration()
    {
        await using var model = await LocalSynthesizer.LoadAsync("fast", cancellationToken: TestContext.Current.CancellationToken);

        var normalResult = await model.SynthesizePhonemesAsync([SpokenPhonemes.Fox], new SynthesizeOptions { Speed = 1.0f }, TestContext.Current.CancellationToken);
        var fastResult = await model.SynthesizePhonemesAsync([SpokenPhonemes.Fox], new SynthesizeOptions { Speed = 2.0f }, TestContext.Current.CancellationToken);

        // Faster speed should produce fewer samples (shorter audio)
        fastResult.AudioSamples.Length.Should().BeLessThan(normalResult.AudioSamples.Length,
            "speed=2.0 should produce shorter audio than speed=1.0");
    }

    [Fact]
    [Trait("Axis", "Inference")]
    public async Task I_ToPcm16Bytes_ProducesValidData()
    {
        await using var model = await LocalSynthesizer.LoadAsync("fast", cancellationToken: TestContext.Current.CancellationToken);

        var result = await model.SynthesizePhonemesAsync(Hello, cancellationToken: TestContext.Current.CancellationToken);
        var pcmBytes = result.ToPcm16Bytes();

        pcmBytes.Should().NotBeEmpty();
        // PCM16 is 2 bytes per sample
        pcmBytes.Length.Should().Be(result.AudioSamples.Length * 2,
            "PCM16 should be 2 bytes per sample");
    }

    // ── Gap Tests: SynthesizeOptions defaults, voice variation ───

    [Fact]
    [Trait("Axis", "Quality")]
    public void Q_SynthesizeOptions_AllDefaults_AreDocumented()
    {
        var opts = new SynthesizeOptions();

        opts.Speed.Should().BeApproximately(1.0f, 0.01f);
        opts.SpeakerId.Should().Be(0);
        opts.NoiseScale.Should().BeNull("unset means the voice's own default");
        opts.NoiseWidth.Should().BeNull("unset means the voice's own default");
        opts.OutputFormat.Should().Be(AudioFormat.Wav);
    }

    [Fact]
    [Trait("Axis", "Quality")]
    public void Q_SynthesizerOptions_Defaults_AreReasonable()
    {
        var opts = new SynthesizerOptions();

        opts.ModelId.Should().Be("default");
        opts.Provider.Should().Be(ExecutionProvider.Auto);
        opts.CacheDirectory.Should().BeNull();
        opts.ThreadCount.Should().BeNull();
    }

    [Fact]
    [Trait("Axis", "Quality")]
    public void Q_SynthesizerOptions_Clone_IsIndependentCopy()
    {
        var original = new SynthesizerOptions
        {
            ModelId = "quality",
            Provider = ExecutionProvider.Cpu
        };

        var clone = original.Clone();

        clone.ModelId.Should().Be("quality");
        clone.Provider.Should().Be(ExecutionProvider.Cpu);

        clone.ModelId = "fast";
        original.ModelId.Should().Be("quality", "original unaffected by clone mutation");
    }
}
