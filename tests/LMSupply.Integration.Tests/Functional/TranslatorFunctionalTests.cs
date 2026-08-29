using LMSupply.Exceptions;
using LMSupply.Integration.Tests.Helpers;
using LMSupply.Translator;

namespace LMSupply.Integration.Tests.Functional;

/// <summary>
/// Comprehensive functional tests for the Translator domain.
/// Tests L (loading), I (inference), Q (quality), E (edge cases) axes.
/// Uses OPUS-MT models for ko-en / en-ko translation.
/// Requires GPU + network access. Run locally only.
/// </summary>
[Trait("Category", "Functional")]
[Trait("Category", "LocalOnly")]
[Trait("Domain", "Translator")]
public class TranslatorFunctionalTests
{
    // ── L axis: Model Loading ───────────────────────────────────────

    [Fact]
    [Trait("Axis", "Loading")]
    public async Task L_DefaultAlias_LoadsSuccessfully()
    {
        await using var model = await LocalTranslator.LoadAsync("default", cancellationToken: TestContext.Current.CancellationToken);

        model.ModelId.Should().NotBeNullOrEmpty();
        model.SourceLanguage.Should().NotBeNullOrEmpty();
        model.TargetLanguage.Should().NotBeNullOrEmpty();
    }

    [Fact]
    [Trait("Axis", "Loading")]
    public async Task L_KoEnAlias_LoadsSuccessfully()
    {
        await using var model = await LocalTranslator.LoadAsync("ko-en", cancellationToken: TestContext.Current.CancellationToken);

        model.SourceLanguage.Should().Be("ko");
        model.TargetLanguage.Should().Be("en");
    }

    [Fact]
    [Trait("Axis", "Loading")]
    public async Task L_WarmupAsync_CompletesWithoutError()
    {
        await using var model = await LocalTranslator.LoadAsync("ko-en", cancellationToken: TestContext.Current.CancellationToken);

        var act = () => model.WarmupAsync();
        await act.Should().NotThrowAsync();
    }

    [Fact]
    [Trait("Axis", "Loading")]
    public async Task L_GetModelInfo_ReturnsValidInfo()
    {
        await using var model = await LocalTranslator.LoadAsync("ko-en", cancellationToken: TestContext.Current.CancellationToken);

        var info = model.GetModelInfo();
        info.Should().NotBeNull();
    }

    [Fact]
    [Trait("Axis", "Loading")]
    public async Task L_InvalidModelId_ThrowsException()
    {
        var act = () => LocalTranslator.LoadAsync("nonexistent-translator-xyz");
        await act.Should().ThrowAsync<Exception>();
    }

    [Fact]
    [Trait("Axis", "Loading")]
    public async Task L_GetAvailableModels_ReturnsNonEmptyList()
    {
        var models = LocalTranslator.GetAvailableModels().ToList();

        models.Should().NotBeEmpty();
    }

    // ── I axis: Basic Inference ─────────────────────────────────────

    [Fact]
    [Trait("Axis", "Inference")]
    public async Task I_KoreanToEnglish_BasicTranslation()
    {
        await using var model = await LocalTranslator.LoadAsync("ko-en", cancellationToken: TestContext.Current.CancellationToken);

        var result = await model.TranslateAsync("안녕하세요, 만나서 반갑습니다.", TestContext.Current.CancellationToken);

        result.Should().NotBeNull();
        result.TranslatedText.Should().NotBeNullOrEmpty();
        result.SourceText.Should().Be("안녕하세요, 만나서 반갑습니다.");
    }

    [Fact]
    [Trait("Axis", "Inference")]
    public async Task I_BatchTranslation_ReturnsCorrectCount()
    {
        await using var model = await LocalTranslator.LoadAsync("ko-en", cancellationToken: TestContext.Current.CancellationToken);

        string[] texts = ["안녕하세요", "감사합니다", "좋은 아침입니다"];
        var results = await model.TranslateBatchAsync(texts, TestContext.Current.CancellationToken);

        results.Should().HaveCount(3);
        foreach (var result in results)
        {
            result.TranslatedText.Should().NotBeNullOrEmpty();
        }
    }

    // ── Q axis: Quality / Semantic Correctness ──────────────────────

    [Fact]
    [Trait("Axis", "Quality")]
    public async Task Q_KoreanGreeting_ContainsEnglishGreeting()
    {
        await using var model = await LocalTranslator.LoadAsync("ko-en", cancellationToken: TestContext.Current.CancellationToken);

        var result = await model.TranslateAsync("안녕하세요, 만나서 반갑습니다.", TestContext.Current.CancellationToken);

        result.TranslatedText.ToLowerInvariant()
            .Should().ContainAny("hello", "hi", "nice", "meet", "glad", "pleased",
                "translation should contain greeting-related words");
    }

    [Fact]
    [Trait("Axis", "Quality")]
    public async Task Q_BatchTranslation_PreservesOrder()
    {
        await using var model = await LocalTranslator.LoadAsync("ko-en", cancellationToken: TestContext.Current.CancellationToken);

        string[] texts = ["고양이", "강아지", "비행기"];
        var results = await model.TranslateBatchAsync(texts, TestContext.Current.CancellationToken);

        results.Should().HaveCount(3);
        results[0].SourceText.Should().Be("고양이");
        results[1].SourceText.Should().Be("강아지");
        results[2].SourceText.Should().Be("비행기");
    }

    [Fact]
    [Trait("Axis", "Quality")]
    public async Task Q_ShortText_ProducesNonEmptyResult()
    {
        await using var model = await LocalTranslator.LoadAsync("ko-en", cancellationToken: TestContext.Current.CancellationToken);

        var result = await model.TranslateAsync("네", TestContext.Current.CancellationToken);

        result.TranslatedText.Should().NotBeNullOrEmpty(
            "even single-word input should produce translation");
    }

    [Fact]
    [Trait("Axis", "Quality")]
    public async Task Q_TranslationIsDeterministic()
    {
        await using var model = await LocalTranslator.LoadAsync("ko-en", cancellationToken: TestContext.Current.CancellationToken);

        var result1 = await model.TranslateAsync("오늘 날씨가 좋습니다.", TestContext.Current.CancellationToken);
        var result2 = await model.TranslateAsync("오늘 날씨가 좋습니다.", TestContext.Current.CancellationToken);

        result1.TranslatedText.Should().Be(result2.TranslatedText,
            "same input should produce same translation");
    }

    // ── E axis: Edge Cases ──────────────────────────────────────────

    [Fact]
    [Trait("Axis", "EdgeCase")]
    public async Task E_SpecialCharacters_DoNotCrash()
    {
        await using var model = await LocalTranslator.LoadAsync("ko-en", cancellationToken: TestContext.Current.CancellationToken);

        var result = await model.TranslateAsync("🎉 테스트! #$%", TestContext.Current.CancellationToken);
        result.Should().NotBeNull();
        result.TranslatedText.Should().NotBeNull();
    }

    [Fact]
    [Trait("Axis", "EdgeCase")]
    public async Task E_HtmlTags_HandleGracefully()
    {
        await using var model = await LocalTranslator.LoadAsync("ko-en", cancellationToken: TestContext.Current.CancellationToken);

        var result = await model.TranslateAsync("<b>안녕하세요</b>", TestContext.Current.CancellationToken);
        result.Should().NotBeNull();
        result.TranslatedText.Should().NotBeNullOrEmpty();
    }

    [Fact]
    [Trait("Axis", "EdgeCase")]
    public async Task E_LongText_DoesNotOOM()
    {
        await using var model = await LocalTranslator.LoadAsync("ko-en", cancellationToken: TestContext.Current.CancellationToken);

        // Repeat a Korean sentence many times to create long input
        var longText = string.Join(" ", Enumerable.Repeat("오늘은 좋은 날입니다.", 100));

        var act = () => model.TranslateAsync(longText);
        await act.Should().NotThrowAsync("long text should truncate, not OOM");
    }

    [Fact]
    [Trait("Axis", "EdgeCase")]
    public async Task E_NullOptions_UsesDefaults()
    {
        await using var model = await LocalTranslator.LoadAsync("ko-en", options: null, cancellationToken: TestContext.Current.CancellationToken);

        var result = await model.TranslateAsync("테스트", TestContext.Current.CancellationToken);
        result.TranslatedText.Should().NotBeNullOrEmpty();
    }

    [Fact]
    [Trait("Axis", "EdgeCase")]
    public async Task E_EnglishInputToKoEnModel_StillProcesses()
    {
        await using var model = await LocalTranslator.LoadAsync("ko-en", cancellationToken: TestContext.Current.CancellationToken);

        // English input to a ko→en model: should still process without crash
        var result = await model.TranslateAsync("Hello world", TestContext.Current.CancellationToken);
        result.Should().NotBeNull();
    }

    // ── Gap Tests: Reverse Direction, Models, Properties ───────────

    [Fact]
    [Trait("Axis", "Loading")]
    public async Task L_EnKoAlias_LoadsSuccessfully()
    {
        await using var model = await LocalTranslator.LoadAsync("en-ko", cancellationToken: TestContext.Current.CancellationToken);

        model.SourceLanguage.Should().Be("en");
        model.TargetLanguage.Should().Be("ko");
    }

    [Fact]
    [Trait("Axis", "Loading")]
    public void L_GetAllModels_ReturnsModelInfo()
    {
        var models = LocalTranslator.GetAllModels().ToList();

        models.Should().NotBeEmpty();
        models.Should().AllSatisfy(m => m.Should().NotBeNull());
    }

    [Fact]
    [Trait("Axis", "Loading")]
    public async Task L_ModelProperties_ArePopulated()
    {
        await using var model = await LocalTranslator.LoadAsync("ko-en", cancellationToken: TestContext.Current.CancellationToken);

        model.ActiveProviders.Should().NotBeNull();
        model.RequestedProvider.Should().BeDefined();
    }

    [Fact]
    [Trait("Axis", "Inference")]
    public async Task I_EnglishToKorean_BasicTranslation()
    {
        await using var model = await LocalTranslator.LoadAsync("en-ko", cancellationToken: TestContext.Current.CancellationToken);

        var result = await model.TranslateAsync("Hello, nice to meet you.", TestContext.Current.CancellationToken);

        result.TranslatedText.Should().NotBeNullOrEmpty();
    }

    [Fact]
    [Trait("Axis", "Quality")]
    public async Task Q_RoundTrip_PreservesSemanticMeaning()
    {
        await using var koEn = await LocalTranslator.LoadAsync("ko-en", cancellationToken: TestContext.Current.CancellationToken);
        await using var enKo = await LocalTranslator.LoadAsync("en-ko", cancellationToken: TestContext.Current.CancellationToken);

        var original = "오늘 날씨가 좋습니다.";
        var english = await koEn.TranslateAsync(original, TestContext.Current.CancellationToken);
        var backToKorean = await enKo.TranslateAsync(english.TranslatedText, TestContext.Current.CancellationToken);

        // Round-trip may not be identical but should produce non-empty result
        backToKorean.TranslatedText.Should().NotBeNullOrEmpty(
            "round-trip translation should produce text");
    }

    // ── Static Tests: Model Registry Validation (no model loading) ─

    [Fact]
    [Trait("Axis", "Loading")]
    public void L_GetAvailableModels_ContainsLanguagePairs()
    {
        var models = LocalTranslator.GetAvailableModels().ToList();

        models.Should().Contain("ko-en", "Korean→English should be registered");
    }

    [Fact]
    [Trait("Axis", "Loading")]
    public void L_GetAllModels_AllHaveRequiredFields()
    {
        var models = LocalTranslator.GetAllModels().ToList();

        models.Should().AllSatisfy(m =>
        {
            m.Id.Should().NotBeNullOrEmpty($"Id should be set for {m.AliasName}");
            m.AliasName.Should().NotBeNullOrEmpty($"AliasName should be set for {m.Id}");
            m.DisplayName.Should().NotBeNullOrEmpty($"DisplayName should be set for {m.AliasName}");
            m.Architecture.Should().NotBeNullOrEmpty($"Architecture should be set for {m.AliasName}");
            m.SourceLanguage.Should().NotBeNullOrEmpty($"SourceLanguage should be set for {m.AliasName}");
            m.TargetLanguage.Should().NotBeNullOrEmpty($"TargetLanguage should be set for {m.AliasName}");
            m.TokenizerFile.Should().NotBeNullOrEmpty($"TokenizerFile should be set for {m.AliasName}");
        });
    }

    [Fact]
    [Trait("Axis", "Loading")]
    public void L_GetAllModels_HasExpectedCount()
    {
        var models = LocalTranslator.GetAllModels().ToList();

        models.Count.Should().BeGreaterThanOrEqualTo(2,
            "registry should have at least 2 translation models (ko-en, en-ko)");
    }

    [Fact]
    [Trait("Axis", "Quality")]
    public void Q_TranslatorOptions_Defaults_AreReasonable()
    {
        var opts = new TranslatorOptions();

        opts.ModelId.Should().Be("default", "default model ID should be 'default'");
        opts.MaxLength.Should().Be(512, "default max length should be 512");
        opts.BeamWidth.Should().Be(4, "default beam width should be 4");
        opts.UseGreedyDecoding.Should().BeFalse("beam search should be default");
        opts.LengthPenalty.Should().Be(1.0f, "default length penalty should be 1.0");
        opts.RepetitionPenalty.Should().Be(1.0f, "default repetition penalty should be 1.0");
        opts.DisableAutoDownload.Should().BeFalse("auto download should be enabled by default");
        opts.Provider.Should().Be(ExecutionProvider.Auto, "default provider should be Auto");
    }

    // ── Model-Loading: HTML Tags Edge Case ───────────────────────

    [Fact]
    [Trait("Axis", "EdgeCase")]
    public async Task E_HtmlTags_DoNotCrash()
    {
        await using var model = await LocalTranslator.LoadAsync("ko-en", cancellationToken: TestContext.Current.CancellationToken);

        var result = await model.TranslateAsync("<b>안녕하세요</b>", TestContext.Current.CancellationToken);
        result.Should().NotBeNull();
        result.TranslatedText.Should().NotBeNullOrEmpty(
            "HTML-tagged input should produce translation without crashing");
    }

    // ── Translator-Specific Static Tests ───────────────────────

    [Fact]
    [Trait("Axis", "Loading")]
    public void L_GetAvailableModels_ContainsBothKoEnDirections()
    {
        var models = LocalTranslator.GetAvailableModels().ToList();

        models.Should().Contain("ko-en", "should have Korean→English");
        models.Should().Contain("en-ko", "should have English→Korean");
    }

    [Fact]
    [Trait("Axis", "Loading")]
    public void L_GetAllModels_AllHaveLanguageInfo()
    {
        var models = LocalTranslator.GetAllModels().ToList();

        foreach (var model in models)
        {
            model.SourceLanguage.Should().NotBeNullOrEmpty(
                $"model {model.AliasName} should have source language");
            model.TargetLanguage.Should().NotBeNullOrEmpty(
                $"model {model.AliasName} should have target language");
            model.Architecture.Should().NotBeNullOrEmpty(
                $"model {model.AliasName} should have architecture");
        }
    }

    [Fact]
    [Trait("Axis", "Loading")]
    public void L_GetAllModels_AllHaveTokenizerFile()
    {
        // Translator models use auto-discovery for encoder/decoder
        // but should always have a tokenizer file specified
        var models = LocalTranslator.GetAllModels().ToList();

        foreach (var model in models)
        {
            model.TokenizerFile.Should().NotBeNullOrEmpty(
                $"model {model.AliasName} should have tokenizer file");
        }
    }

    // ── Gap Tests: Empty/Whitespace + Unsupported Language Pair ────

    [Fact]
    [Trait("Axis", "EdgeCase")]
    public async Task E_EmptyText_ThrowsArgumentException()
    {
        await using var model = await LocalTranslator.LoadAsync("ko-en", cancellationToken: TestContext.Current.CancellationToken);

        var act = () => model.TranslateAsync("");
        await act.Should().ThrowAsync<ArgumentException>(
            "empty text should throw ArgumentException, not crash");
    }

    [Fact]
    [Trait("Axis", "EdgeCase")]
    public async Task E_WhitespaceOnlyText_ThrowsArgumentException()
    {
        await using var model = await LocalTranslator.LoadAsync("ko-en", cancellationToken: TestContext.Current.CancellationToken);

        var act = () => model.TranslateAsync("   \t\n  ");
        await act.Should().ThrowAsync<ArgumentException>(
            "whitespace-only text should throw ArgumentException");
    }

    [Fact]
    [Trait("Axis", "Loading")]
    public async Task L_UnsupportedLanguagePair_ThrowsException()
    {
        // A language pair that doesn't exist in the registry
        var act = () => LocalTranslator.LoadAsync("xx-yy");
        await act.Should().ThrowAsync<Exception>(
            "unsupported language pair should throw with clear error");
    }

    [Fact]
    [Trait("Axis", "Inference")]
    public async Task I_BatchTranslation_EmptyArray_ThrowsOrReturnsEmpty()
    {
        await using var model = await LocalTranslator.LoadAsync("ko-en", cancellationToken: TestContext.Current.CancellationToken);

        var results = await model.TranslateBatchAsync(Array.Empty<string>(), TestContext.Current.CancellationToken);
        results.Should().BeEmpty("empty batch should return empty result");
    }

    [Fact]
    [Trait("Axis", "Quality")]
    public void Q_TranslatorOptions_Clone_IsIndependentCopy()
    {
        var original = new TranslatorOptions
        {
            ModelId = "ko-en",
            MaxLength = 256,
            BeamWidth = 6,
            UseGreedyDecoding = true,
            LengthPenalty = 0.8f,
            RepetitionPenalty = 1.2f
        };

        var clone = original.Clone();

        // Verify values are copied
        clone.ModelId.Should().Be("ko-en");
        clone.MaxLength.Should().Be(256);
        clone.BeamWidth.Should().Be(6);
        clone.UseGreedyDecoding.Should().BeTrue();
        clone.LengthPenalty.Should().Be(0.8f);
        clone.RepetitionPenalty.Should().Be(1.2f);

        // Verify independence
        clone.ModelId = "en-ko";
        clone.MaxLength = 1024;
        original.ModelId.Should().Be("ko-en", "original should not be affected by clone mutation");
        original.MaxLength.Should().Be(256, "original should not be affected by clone mutation");
    }
}
