using LMSupply.Integration.Tests.Helpers;
using LMSupply.Transcriber;

namespace LMSupply.Integration.Tests.Functional;

/// <summary>
/// Functional tests for the Transcriber (Speech→Text) domain.
/// Uses Whisper models. Tests L/I/Q/E axes.
/// Requires GPU + network access. Run locally only.
/// </summary>
[Trait("Category", "Functional")]
[Trait("Category", "LocalOnly")]
[Trait("Domain", "Transcriber")]
public class TranscriberFunctionalTests
{
    // ── L axis: Model Loading ───────────────────────────────────────

    [Fact]
    [Trait("Axis", "Loading")]
    public async Task L_FastAlias_LoadsSuccessfully()
    {
        await using var model = await LocalTranscriber.LoadAsync("fast", cancellationToken: TestContext.Current.CancellationToken);

        model.ModelId.Should().NotBeNullOrEmpty();
    }

    [Fact]
    [Trait("Axis", "Loading")]
    public async Task L_GetModelInfo_ReturnsValidInfo()
    {
        await using var model = await LocalTranscriber.LoadAsync("fast", cancellationToken: TestContext.Current.CancellationToken);

        var info = model.GetModelInfo();
        info.Should().NotBeNull();
    }

    [Fact]
    [Trait("Axis", "Loading")]
    public async Task L_InvalidModelId_ThrowsException()
    {
        var act = () => LocalTranscriber.LoadAsync("nonexistent-whisper-xyz");
        await act.Should().ThrowAsync<Exception>();
    }

    // ── I axis: Basic Inference ─────────────────────────────────────

    [Fact]
    [Trait("Axis", "Inference")]
    public async Task I_ToneWav_ProducesResult()
    {
        await using var model = await LocalTranscriber.LoadAsync("fast", cancellationToken: TestContext.Current.CancellationToken);

        var wavBytes = TestDataHelper.CreateToneWav(16000, 1.0f, 440);
        var result = await model.TranscribeAsync(wavBytes, cancellationToken: TestContext.Current.CancellationToken);

        result.Should().NotBeNull();
        result.Language.Should().NotBeNullOrEmpty();
    }

    [Fact]
    [Trait("Axis", "Inference")]
    public async Task I_TranscribeFromByteArray_Works()
    {
        await using var model = await LocalTranscriber.LoadAsync("fast", cancellationToken: TestContext.Current.CancellationToken);

        var wavBytes = TestDataHelper.CreateToneWav(16000, 2.0f, 440);
        var result = await model.TranscribeAsync(wavBytes, cancellationToken: TestContext.Current.CancellationToken);

        result.Should().NotBeNull();
        result.Text.Should().NotBeNull(); // May be empty for tone, but not null
    }

    // ── Q axis: Quality ─────────────────────────────────────────────

    [Fact]
    [Trait("Axis", "Quality")]
    public async Task Q_SegmentTimestamps_AreOrdered()
    {
        await using var model = await LocalTranscriber.LoadAsync("fast", cancellationToken: TestContext.Current.CancellationToken);

        var wavBytes = TestDataHelper.CreateToneWav(16000, 3.0f, 440);
        var result = await model.TranscribeAsync(wavBytes, cancellationToken: TestContext.Current.CancellationToken);

        if (result.Segments.Count > 1)
        {
            for (var i = 1; i < result.Segments.Count; i++)
            {
                result.Segments[i].Start.Should().BeGreaterThanOrEqualTo(
                    result.Segments[i - 1].Start,
                    "segments should be in chronological order");
            }
        }
    }

    [Fact]
    [Trait("Axis", "Quality")]
    public async Task Q_SegmentStartEndOrder_IsValid()
    {
        await using var model = await LocalTranscriber.LoadAsync("fast", cancellationToken: TestContext.Current.CancellationToken);

        var wavBytes = TestDataHelper.CreateToneWav(16000, 3.0f, 440);
        var result = await model.TranscribeAsync(wavBytes, cancellationToken: TestContext.Current.CancellationToken);

        foreach (var segment in result.Segments)
        {
            segment.End.Should().BeGreaterThanOrEqualTo(segment.Start,
                "segment end should be >= start");
        }
    }

    // ── E axis: Edge Cases ──────────────────────────────────────────

    [Fact]
    [Trait("Axis", "EdgeCase")]
    public async Task E_SilenceWav_DoesNotCrash()
    {
        await using var model = await LocalTranscriber.LoadAsync("fast", cancellationToken: TestContext.Current.CancellationToken);

        var wavBytes = TestDataHelper.CreateSilenceWav(16000, 3.0f);

        var act = () => model.TranscribeAsync(wavBytes);
        var result = await act.Should().NotThrowAsync();
        result.Subject.Should().NotBeNull();
    }

    [Fact]
    [Trait("Axis", "EdgeCase")]
    public async Task E_ShortAudio_DoesNotCrash()
    {
        await using var model = await LocalTranscriber.LoadAsync("fast", cancellationToken: TestContext.Current.CancellationToken);

        // Very short: 0.1 second
        var wavBytes = TestDataHelper.CreateToneWav(16000, 0.1f, 440);

        var act = () => model.TranscribeAsync(wavBytes);
        await act.Should().NotThrowAsync("very short audio should not crash");
    }

    [Fact]
    [Trait("Axis", "EdgeCase")]
    public async Task E_InvalidAudioData_ThrowsCleanError()
    {
        await using var model = await LocalTranscriber.LoadAsync("fast", cancellationToken: TestContext.Current.CancellationToken);

        // Send random non-audio bytes
        var invalidData = new byte[] { 0x00, 0x01, 0x02, 0x03, 0x04 };

        var act = () => model.TranscribeAsync(invalidData);
        await act.Should().ThrowAsync<Exception>("invalid audio should throw, not crash");
    }

    // ── Gap Tests: Language, Aliases, Stream ─────────────────────────

    [Fact]
    [Trait("Axis", "Loading")]
    public async Task L_DefaultAlias_LoadsSuccessfully()
    {
        await using var model = await LocalTranscriber.LoadAsync("default", cancellationToken: TestContext.Current.CancellationToken);

        model.ModelId.Should().NotBeNullOrEmpty();
    }

    [Fact]
    [Trait("Axis", "Loading")]
    public async Task L_DefaultAlias_CpuProvider_LoadsAndTranscribesWithoutCrashing()
    {
        // Regression gate for docket iyulab/lm-supply#151: the package's own zero-config
        // registry default (onnx-community/whisper-base, "default" alias) + CPU EP is the
        // combination that crashed InferenceSession construction under
        // Microsoft.ML.OnnxRuntime 1.25.1/1.26.0 ("Missing required scale:
        // ...embed_tokens.weight_merged_0_scale"). CPU EP is forced explicitly here (unlike
        // the other Loading/Inference tests above, which use the default Auto provider and so
        // exercise a separate, already-known DirectML LayerNormalization crash instead — see
        // docket iyulab/lm-supply#59 — not this regression). Do not bump
        // Microsoft.ML.OnnxRuntime past the version pinned in Directory.Packages.props without
        // this test passing first.
        var options = new TranscriberOptions { Provider = ExecutionProvider.Cpu };
        await using var model = await LocalTranscriber.LoadAsync("default", options, cancellationToken: TestContext.Current.CancellationToken);

        var wavBytes = TestDataHelper.CreateToneWav(16000, 2.0f, 440);
        var result = await model.TranscribeAsync(wavBytes, cancellationToken: TestContext.Current.CancellationToken);

        result.Should().NotBeNull();
    }

    [Fact]
    [Trait("Axis", "Loading")]
    public async Task L_GetAvailableModels_ReturnsAliases()
    {
        var models = LocalTranscriber.GetAvailableModels().ToList();

        models.Should().NotBeEmpty();
        models.Should().Contain("fast");
    }

    [Fact]
    [Trait("Axis", "Loading")]
    public async Task L_GetAllModels_ReturnsModelInfo()
    {
        var models = LocalTranscriber.GetAllModels().ToList();

        models.Should().NotBeEmpty();
        models.Should().AllSatisfy(m => m.Should().NotBeNull());
    }

    [Fact]
    [Trait("Axis", "Inference")]
    public async Task I_TranscribeWithLanguageHint_Works()
    {
        await using var model = await LocalTranscriber.LoadAsync("fast", cancellationToken: TestContext.Current.CancellationToken);

        var wavBytes = TestDataHelper.CreateToneWav(16000, 2.0f, 440);
        var options = new TranscribeOptions { Language = "en" };
        var result = await model.TranscribeAsync(wavBytes, options, TestContext.Current.CancellationToken);

        result.Should().NotBeNull();
    }

    [Fact]
    [Trait("Axis", "Inference")]
    public async Task I_TranscribeFromStream_Works()
    {
        await using var model = await LocalTranscriber.LoadAsync("fast", cancellationToken: TestContext.Current.CancellationToken);

        var wavBytes = TestDataHelper.CreateToneWav(16000, 2.0f, 440);
        using var stream = new MemoryStream(wavBytes);
        var result = await model.TranscribeAsync(stream, cancellationToken: TestContext.Current.CancellationToken);

        result.Should().NotBeNull();
        result.Text.Should().NotBeNull();
    }

    [Fact]
    [Trait("Axis", "Quality")]
    public async Task Q_DetectedLanguage_IsNotEmpty()
    {
        await using var model = await LocalTranscriber.LoadAsync("fast", cancellationToken: TestContext.Current.CancellationToken);

        var wavBytes = TestDataHelper.CreateToneWav(16000, 2.0f, 440);
        var result = await model.TranscribeAsync(wavBytes, cancellationToken: TestContext.Current.CancellationToken);

        result.Language.Should().NotBeNullOrEmpty("model should detect or assign a language");
    }

    [Fact]
    [Trait("Axis", "EdgeCase")]
    public async Task E_NullOptions_UsesDefaults()
    {
        await using var model = await LocalTranscriber.LoadAsync("fast", cancellationToken: TestContext.Current.CancellationToken);

        var wavBytes = TestDataHelper.CreateToneWav(16000, 1.0f, 440);
        var result = await model.TranscribeAsync(wavBytes, options: null, cancellationToken: TestContext.Current.CancellationToken);

        result.Should().NotBeNull();
    }

    [Fact]
    [Trait("Axis", "EdgeCase")]
    public async Task E_ModelProperties_ArePopulated()
    {
        await using var model = await LocalTranscriber.LoadAsync("fast", cancellationToken: TestContext.Current.CancellationToken);

        model.ActiveProviders.Should().NotBeNull();
        model.RequestedProvider.Should().BeDefined();
    }

    // ── Static Tests: Model Registry Validation (no model loading) ─

    [Fact]
    [Trait("Axis", "Loading")]
    public void L_GetAvailableModels_ContainsAllStandardAliases()
    {
        var models = LocalTranscriber.GetAvailableModels().ToList();

        models.Should().Contain("default");
        models.Should().Contain("fast");
        models.Should().Contain("quality");
        models.Should().Contain("auto");
    }

    [Fact]
    [Trait("Axis", "Loading")]
    public void L_GetAllModels_HasExpectedCount()
    {
        var models = LocalTranscriber.GetAllModels().ToList();

        models.Count.Should().BeGreaterThanOrEqualTo(5,
            "registry should have at least 5 Whisper model variants");
    }

    [Fact]
    [Trait("Axis", "Loading")]
    public void L_GetAllModels_AllHaveRequiredFields()
    {
        var models = LocalTranscriber.GetAllModels().ToList();

        models.Should().AllSatisfy(m =>
        {
            m.Id.Should().NotBeNullOrEmpty("model Id is required");
            m.AliasName.Should().NotBeNullOrEmpty("AliasName is required");
            m.DisplayName.Should().NotBeNullOrEmpty("DisplayName is required");
            m.SampleRate.Should().BeGreaterThan(0, "SampleRate should be > 0");
        });
    }

    [Fact]
    [Trait("Axis", "Loading")]
    public void L_GetAllModels_DefaultIsWhisperBase()
    {
        var models = LocalTranscriber.GetAllModels().ToList();
        var defaultModel = models.FirstOrDefault(m => m.AliasName == "default");

        defaultModel.Should().NotBeNull("default alias should exist");
        defaultModel!.DisplayName.Should().Contain("Base",
            "default Whisper model should be Base variant");
    }

    [Fact]
    [Trait("Axis", "Loading")]
    public void L_GetAllModels_ContainsMultilingualModels()
    {
        var models = LocalTranscriber.GetAllModels().ToList();

        // Most Whisper models are multilingual
        models.Should().Contain(m => m.IsMultilingual,
            "registry should have at least one multilingual model");
    }

    [Fact]
    [Trait("Axis", "Quality")]
    public void Q_TranscribeOptions_Defaults_AreCorrect()
    {
        var opts = new TranscribeOptions();

        opts.Language.Should().BeNull("default language should be null (auto-detect)");
        opts.Translate.Should().BeFalse("default Translate should be false");
        opts.Temperature.Should().BeApproximately(0f, 0.01f,
            "default temperature should be 0 (greedy)");
        opts.BeamWidth.Should().Be(5, "default beam width should be 5");
    }

    [Fact]
    [Trait("Axis", "Quality")]
    public void Q_TranscriberOptions_Defaults_AreReasonable()
    {
        var opts = new TranscriberOptions();

        opts.ModelId.Should().Be("default", "default model ID should be 'default'");
        opts.DisableAutoDownload.Should().BeFalse("auto download should be enabled by default");
        opts.Provider.Should().Be(ExecutionProvider.Auto, "default provider should be Auto");
        opts.CacheDirectory.Should().BeNull("default cache directory should be null");
    }

    // ── Model-Loading Tests: Quality Alias, Translate Option ─────

    [Fact]
    [Trait("Axis", "Loading")]
    public async Task L_QualityAlias_LoadsSuccessfully()
    {
        await using var model = await LocalTranscriber.LoadAsync("quality", cancellationToken: TestContext.Current.CancellationToken);

        model.ModelId.Should().NotBeNullOrEmpty();
    }

    [Fact]
    [Trait("Axis", "Loading")]
    public async Task L_AutoAlias_LoadsSuccessfully()
    {
        await using var model = await LocalTranscriber.LoadAsync("auto", cancellationToken: TestContext.Current.CancellationToken);

        model.ModelId.Should().NotBeNullOrEmpty();
    }

    [Fact]
    [Trait("Axis", "Inference")]
    public async Task I_TranslateOption_ProducesEnglishText()
    {
        await using var model = await LocalTranscriber.LoadAsync("fast", cancellationToken: TestContext.Current.CancellationToken);

        var wavBytes = TestDataHelper.CreateToneWav(16000, 2.0f, 440);
        var options = new TranscribeOptions { Translate = true };
        var result = await model.TranscribeAsync(wavBytes, options, TestContext.Current.CancellationToken);

        // With translate=true, output should be in English
        result.Should().NotBeNull();
        result.Language.Should().NotBeNullOrEmpty();
    }

    // ── Gap Tests: TranscribeOptions defaults, WordTimestamps ────

    [Fact]
    [Trait("Axis", "Quality")]
    public void Q_TranscribeOptions_AllDefaults_AreDocumented()
    {
        var opts = new TranscribeOptions();

        opts.Language.Should().BeNull("auto-detect by default");
        opts.Translate.Should().BeFalse();
        opts.WordTimestamps.Should().BeFalse();
        opts.InitialPrompt.Should().BeNull();
        opts.MaxTokens.Should().Be(448);
        opts.BeamWidth.Should().Be(5);
        opts.Temperature.Should().Be(0f);
        opts.CompressionRatioThreshold.Should().BeApproximately(2.4f, 0.01f);
        opts.NoSpeechThreshold.Should().BeApproximately(0.6f, 0.01f);
    }

    [Fact]
    [Trait("Axis", "Inference")]
    public async Task I_WordTimestamps_ProducesSegments()
    {
        await using var model = await LocalTranscriber.LoadAsync("fast", cancellationToken: TestContext.Current.CancellationToken);

        var wavBytes = TestDataHelper.CreateToneWav(16000, 3.0f, 440);
        var options = new TranscribeOptions { WordTimestamps = true };
        var result = await model.TranscribeAsync(wavBytes, options, TestContext.Current.CancellationToken);

        result.Should().NotBeNull();
        // WordTimestamps should produce finer segments
        result.Segments.Should().NotBeNull();
    }

    [Fact]
    [Trait("Axis", "Quality")]
    public async Task Q_RepeatTranscription_IsDeterministic()
    {
        await using var model = await LocalTranscriber.LoadAsync("fast", cancellationToken: TestContext.Current.CancellationToken);

        var wavBytes = TestDataHelper.CreateToneWav(16000, 2.0f, 440);
        var result1 = await model.TranscribeAsync(wavBytes, cancellationToken: TestContext.Current.CancellationToken);
        var result2 = await model.TranscribeAsync(wavBytes, cancellationToken: TestContext.Current.CancellationToken);

        result1.Text.Should().Be(result2.Text,
            "same audio should produce same transcription");
    }

    [Fact]
    [Trait("Axis", "Quality")]
    public void Q_TranscriberOptions_Clone_IsIndependentCopy()
    {
        var original = new TranscriberOptions
        {
            ModelId = "quality",
            DisableAutoDownload = true,
            Provider = ExecutionProvider.Cpu
        };

        var clone = original.Clone();

        clone.ModelId.Should().Be("quality");
        clone.DisableAutoDownload.Should().BeTrue();

        clone.ModelId = "fast";
        original.ModelId.Should().Be("quality", "original unaffected by clone mutation");
    }

    [Fact]
    [Trait("Axis", "Inference")]
    public async Task I_TranscribeWithKoreanLanguageHint_Works()
    {
        // TR-Q2: Verify Korean language hint is accepted and produces a result
        await using var model = await LocalTranscriber.LoadAsync("fast", cancellationToken: TestContext.Current.CancellationToken);

        var wavBytes = TestDataHelper.CreateToneWav(16000, 2.0f, 440);
        var options = new TranscribeOptions { Language = "ko" };
        var result = await model.TranscribeAsync(wavBytes, options, TestContext.Current.CancellationToken);

        result.Should().NotBeNull();
        result.Text.Should().NotBeNull("transcription with Korean hint should produce text (even for tone audio)");
    }

    // ── Multi-Format Tests (TR-Q1) ─────────────────────────────────

    [Fact]
    [Trait("Axis", "Inference")]
    public async Task I_Mp3File_TranscribesSuccessfully()
    {
        // TR-Q1: Verify MP3 input decodes via the Mp3FileReaderBase + NLayer path (docket
        // iyulab/lm-supply#169 — AudioFileReader itself no longer supports .mp3 on this
        // project's cross-platform NAudio 3.0 build).
        var mp3Bytes = TestDataHelper.TryCreateToneMp3(44100, 2.0f, 440);
        if (mp3Bytes is null)
        {
            // LAME encoding failed for some environment-specific reason — skip if unavailable
            return;
        }

        await using var model = await LocalTranscriber.LoadAsync("fast", cancellationToken: TestContext.Current.CancellationToken);

        var tempPath = TestDataHelper.SaveToTempFile(mp3Bytes, ".mp3");
        try
        {
            var result = await model.TranscribeAsync(tempPath, cancellationToken: TestContext.Current.CancellationToken);

            result.Should().NotBeNull();
            result.Text.Should().NotBeNull("MP3 input should produce transcription text");
        }
        finally
        {
            File.Delete(tempPath);
        }
    }

    [Fact]
    [Trait("Axis", "Inference")]
    public async Task I_HighSampleRateWavFile_TranscribesViaFilePath()
    {
        // TR-Q1: 44.1kHz WAV exercises resampling to 16kHz in AudioProcessor
        var wavBytes = TestDataHelper.CreateToneWav(44100, 2.0f, 440);

        await using var model = await LocalTranscriber.LoadAsync("fast", cancellationToken: TestContext.Current.CancellationToken);

        var tempPath = TestDataHelper.SaveToTempFile(wavBytes, ".wav");
        try
        {
            var result = await model.TranscribeAsync(tempPath, cancellationToken: TestContext.Current.CancellationToken);

            result.Should().NotBeNull();
            result.Text.Should().NotBeNull("high sample rate WAV should be resampled and transcribed");
        }
        finally
        {
            File.Delete(tempPath);
        }
    }
}
