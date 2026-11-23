using AwesomeAssertions;
using LMSupply.Exceptions;
using LMSupply.Generator;
using LMSupply.Generator.Gguf;

namespace LMSupply.Integration.Tests;

/// <summary>
/// The rung vocabulary is one set of names in every domain (<see cref="ModelTier"/>), so what makes it one set is
/// checked across the domains together: a domain-by-domain test would let one domain drift from the others.
/// </summary>
public class ModelTierLadderTests
{
    // The size adjectives are not aliases. The image generator's two are step-count presets of one model, not rungs.
    private static readonly string[] RetiredAdjectives = ["fast", "quality", "large", "xlarge"];

    private static void CheckLadder<T>(string domain, ModelRegistryBase<T> registry) where T : class, IModelInfoBase
    {
        var tiers = registry.GetTiers();

        tiers.Select(t => t.Tier).Should().BeInAscendingOrder($"{domain}: the ladder lists lowest first").And.OnlyHaveUniqueItems($"{domain}: one model per rung");
        foreach (var rung in tiers)
        {
            registry.Resolve(rung.Alias).CatalogKey.Should().Be(rung.Model.CatalogKey, $"{domain}: the rung alias '{rung.Alias}' names the rung's model");
            rung.Model.Tier.Should().Be(rung.Tier);
        }

        if (domain != "imagegenerator")
        {
            registry.GetAliases().Where(a => a.Kind == AliasKind.System).Select(a => a.Name)
                .Should().NotContain(RetiredAdjectives, $"{domain}: a size adjective names no rung and no model");
        }

        // `default` is the basic rung wherever the domain has one.
        if (tiers.Any(t => t.Tier == ModelTier.Basic) && registry.TryResolve("default", out var defaultModel))
        {
            defaultModel!.CatalogKey.Should().Be(tiers.First(t => t.Tier == ModelTier.Basic).Model.CatalogKey, $"{domain}: 'default' is the basic rung");
        }
    }

    [Fact]
    public void EveryDomainsLadder_IsOneSetOfNames_AndDefaultIsItsBasicRung()
    {
        CheckLadder("embedder", LMSupply.Embedder.Utils.EmbedderModelRegistry.Default);
        CheckLadder("reranker", LMSupply.Reranker.Models.RerankerModelRegistry.Default);
        CheckLadder("transcriber", LMSupply.Transcriber.Models.TranscriberModelRegistry.Default);
        CheckLadder("segmenter", LMSupply.Segmenter.Models.SegmenterModelRegistry.Default);
        CheckLadder("detector", LMSupply.Detector.Models.DetectorModelRegistry.Default);
        CheckLadder("captioner", LMSupply.Captioner.Models.CaptionerModelRegistry.Default);
        CheckLadder("ocr-detection", LMSupply.Ocr.Models.OcrDetectionModelRegistry.Default);
        CheckLadder("imagegenerator", LMSupply.ImageGenerator.Models.ImageGeneratorModelRegistry.Default);
        CheckLadder("synthesizer", LMSupply.Synthesizer.Models.SynthesizerModelRegistry.Default);
        CheckLadder("generator-onnx", GeneratorModelRegistry.Default);
    }

    [Fact]
    public void TheGgufLadder_IsFiveRungs_NamedLikeEveryOtherDomain()
    {
        var tiers = GgufModelRegistry.GetTiers();

        tiers.Select(t => t.Alias).Should().Equal("gguf:entry", "gguf:basic", "gguf:balanced", "gguf:advanced", "gguf:frontier");
        tiers.Select(t => t.Tier).Should().Equal(ModelTier.Entry, ModelTier.Basic, ModelTier.Balanced, ModelTier.Advanced, ModelTier.Frontier);
        GgufModelRegistry.GetAliases().Should().NotContain(a => a.Contains("qwen3-") || RetiredAdjectives.Any(w => a.EndsWith("-" + w, StringComparison.Ordinal) || a == "gguf:" + w),
            "no alias carries a model family with a size word, or a bare size word");
    }

    [Theory]
    [InlineData("gguf:qwen3-fast", "gguf:entry")]
    [InlineData("gguf:qwen3-default", "gguf:basic")]
    [InlineData("gguf:xlarge", "gguf:frontier")]
    [InlineData("gguf:gemma4-default", "gguf:gemma4-e4b")]
    public async Task ARetiredGgufAlias_FailsAndSaysWhatReplacedIt(string retired, string replacement)
    {
        GgufModelRegistry.Resolve(retired).Should().BeNull("a retired name loads nothing");

        var load = () => LocalGenerator.LoadAsync(retired, new GeneratorOptions { DisableAutoDownload = true });

        (await load.Should().ThrowAsync<ArgumentException>()).WithMessage($"*{replacement}*");
    }

    [Fact]
    public async Task ARetiredBareName_IsNotLookedUpAsAHubRepository_ByTheGenerator()
    {
        var load = () => LocalGenerator.LoadAsync("fast", new GeneratorOptions { DisableAutoDownload = true });

        (await load.Should().ThrowAsync<ModelNotFoundException>()).WithMessage("*'fast' is no longer an alias; it is now 'entry'*");
    }

    [Fact]
    public async Task ARetiredName_FailsWithItsReplacement_InTheDomainsThatLoadFromTheirOwnCatalog()
    {
        var embedder = () => LMSupply.Embedder.LocalEmbedder.LoadAsync("fast", new LMSupply.Embedder.EmbedderOptions { DisableAutoDownload = true });
        var reranker = () => LMSupply.Reranker.LocalReranker.LoadAsync("quality", new LMSupply.Reranker.RerankerOptions { DisableAutoDownload = true });
        var captioner = () => LMSupply.Captioner.LocalCaptioner.LoadAsync("quality", new LMSupply.Captioner.CaptionerOptions { DisableAutoDownload = true });

        (await embedder.Should().ThrowAsync<ModelNotFoundException>()).WithMessage("*'entry'*");
        (await reranker.Should().ThrowAsync<ModelNotFoundException>()).WithMessage("*'balanced'*");
        (await captioner.Should().ThrowAsync<ModelNotFoundException>()).WithMessage("*'basic'*");
    }
}
