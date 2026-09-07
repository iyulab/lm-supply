using AwesomeAssertions;
using LMSupply.Embedder;
using LMSupply.Generator;
using Xunit;

namespace LMSupply.Embedder.Tests;

/// <summary>
/// Keeps <see cref="WellKnownModels"/> pointing at models the registries actually carry.
///
/// <para>
/// Three of the <c>Embedder</c> constants had drifted onto models the embedder registry does not
/// know: they still named the English-first small models from an earlier default policy, long after
/// the registry moved to multilingual ones. Nothing failed, because an unrecognised value is not an
/// error — it is treated as a raw HuggingFace repo id and downloaded, just without the registry's
/// dimensions, pooling mode, subfolder and query/passage prefixes. A consumer following the
/// library's own "well known" list therefore got a quietly worse pipeline than the alias path.
/// </para>
///
/// <para>
/// This assertion is deliberately a plain string comparison against the public registry
/// enumeration, so it needs no model download and runs on every CI machine. The suite that would
/// otherwise have caught the drift is excluded from CI, which is precisely why it did not.
/// </para>
/// </summary>
public class WellKnownModelsRegistryTests
{
    public static TheoryData<string, string> EmbedderConstants() => new()
    {
        { nameof(WellKnownModels.Embedder.Default), WellKnownModels.Embedder.Default },
        { nameof(WellKnownModels.Embedder.Fast), WellKnownModels.Embedder.Fast },
        { nameof(WellKnownModels.Embedder.Quality), WellKnownModels.Embedder.Quality },
        { nameof(WellKnownModels.Embedder.Large), WellKnownModels.Embedder.Large },
        { nameof(WellKnownModels.Embedder.Multilingual), WellKnownModels.Embedder.Multilingual },
        { nameof(WellKnownModels.Embedder.MultilingualLarge), WellKnownModels.Embedder.MultilingualLarge },
        { nameof(WellKnownModels.Embedder.BgeBase), WellKnownModels.Embedder.BgeBase },
    };

    [Theory]
    [MemberData(nameof(EmbedderConstants))]
    public void EmbedderConstant_NamesAModelTheRegistryCarries(string constantName, string repoId)
    {
        var known = LocalEmbedder.GetAllModels().Select(m => m.RepoId).ToHashSet(StringComparer.OrdinalIgnoreCase);

        known.Should().Contain(repoId,
            $"WellKnownModels.Embedder.{constantName} must resolve through the registry, not fall " +
            "through to an untuned raw repo id");
    }
}
