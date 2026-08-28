using System.Text.RegularExpressions;
using AwesomeAssertions;
using LMSupply.Transcriber.Models;

namespace LMSupply.Transcriber.Tests;

/// <summary>
/// Tests for the Transcriber language alias naming convention.
/// Convention: Language-specific aliases use {tier}-{lang} pattern (ISO 639-1 codes).
/// Examples: large-ko, fast-ko, quality-ja
/// Standalone language aliases (english) are also allowed for backward compatibility.
/// </summary>
public class LanguageAliasConventionTests
{
    private static readonly Regex LanguageAliasPattern = new(
        @"^(fast|default|quality|large|turbo|medium|distil)-[a-z]{2}$",
        RegexOptions.Compiled);

    [Fact]
    public void LargeKo_FollowsTierLangPattern()
    {
        LanguageAliasPattern.IsMatch("large-ko").Should().BeTrue();
    }

    [Fact]
    public void AllLanguageSpecificModels_HaveLanguageInfo()
    {
        var models = TranscriberModelRegistry.Default.GetAvailableModels();
        var languageSpecific = models.Where(m =>
            m.SupportedLanguages is { Count: > 0 } &&
            !m.IsMultilingual).ToList();

        languageSpecific.Should().NotBeEmpty("should have language-specific models");

        foreach (var model in languageSpecific)
        {
            model.SupportedLanguages.Should().NotBeNull(
                $"Language-specific model '{model.AliasName}' should list supported languages");
        }
    }

    [Fact]
    public void KoreanModel_HasCorrectLanguageMetadata()
    {
        var model = TranscriberModelRegistry.Default.Resolve("large-ko");

        model.SupportedLanguages.Should().Contain("ko");
        model.IsMultilingual.Should().BeFalse();
    }

    [Fact]
    public void EnglishModel_HasCorrectLanguageMetadata()
    {
        var model = TranscriberModelRegistry.Default.Resolve("english");

        model.SupportedLanguages.Should().Contain("en");
        model.IsMultilingual.Should().BeFalse();
    }

    [Fact]
    public void MultilingualModels_DoNotHaveLanguageSuffix()
    {
        var models = TranscriberModelRegistry.Default.GetAvailableModels();
        var multilingual = models.Where(m => m.IsMultilingual).ToList();

        foreach (var model in multilingual)
        {
            // Multilingual models should not have language codes in their alias
            model.AliasName.Should().NotMatchRegex(@"-[a-z]{2}$",
                $"Multilingual model '{model.AliasName}' should not have a language suffix");
        }
    }

    [Fact]
    public void LanguageAliases_ShouldNotConflictWithStandardTiers()
    {
        var standardTiers = new[] { "default", "fast", "quality", "large", "medium", "turbo", "distil", "english", "auto" };
        var aliases = TranscriberModelRegistry.Default.GetAliases();

        var languageAliases = aliases
            .Where(a => a.Name.Contains('-') && a.Kind == AliasKind.System)
            .Select(a => a.Name)
            .ToList();

        foreach (var langAlias in languageAliases)
        {
            standardTiers.Should().NotContain(langAlias,
                $"Language alias '{langAlias}' should not conflict with standard tier names");
        }
    }

    [Fact]
    public void AllSystemAliases_AreUnique()
    {
        var aliases = TranscriberModelRegistry.Default.GetAliases();
        var names = aliases.Select(a => a.Name).ToList();

        names.Should().OnlyHaveUniqueItems("all alias names should be unique");
    }

    [Fact]
    public void LanguageSpecificModels_AlwaysHaveSupportedLanguagesList()
    {
        // Models with language code in alias name should have SupportedLanguages
        var aliases = TranscriberModelRegistry.Default.GetAliases()
            .Where(a => LanguageAliasPattern.IsMatch(a.Name))
            .ToList();

        foreach (var alias in aliases)
        {
            var model = TranscriberModelRegistry.Default.Resolve(alias.Name);
            model.SupportedLanguages.Should().NotBeNullOrEmpty(
                $"Model with language alias '{alias.Name}' should specify supported languages");
        }
    }

    [Fact]
    public void LanguageAliasFormat_TierPart_ShouldMatchExistingTier()
    {
        // For {tier}-{lang} aliases, the tier part should correspond to a real tier
        var existingTiers = new HashSet<string>(StringComparer.OrdinalIgnoreCase)
            { "fast", "default", "quality", "large", "medium", "turbo", "distil" };

        var aliases = TranscriberModelRegistry.Default.GetAliases()
            .Where(a => LanguageAliasPattern.IsMatch(a.Name))
            .ToList();

        foreach (var alias in aliases)
        {
            var tierPart = alias.Name.Split('-')[0];
            existingTiers.Should().Contain(tierPart,
                $"Language alias '{alias.Name}' should use a recognized tier prefix");
        }
    }
}
