using AwesomeAssertions;
using LMSupply.Generator;
using Xunit;

namespace LMSupply.Generator.Tests;

/// <summary>
/// Every generator constant must be reducible to a repository id.
///
/// <para>
/// The model cache is indexed by directory names of the form <c>models--org--name</c>, so the set
/// of cached ids only ever contains <c>org/name</c> strings. A constant that is a registry alias
/// (<c>"phi-4-mini"</c>) is perfectly valid for loading but can never match that set — which is how
/// the host's model-registry endpoint came to report a cached model as not cached, and to return
/// something that was not a repository id in a field called <c>RepoId</c>.
/// </para>
///
/// <para>
/// The invariant guarded here is therefore not "the constant is a repo id" but "the constant is
/// either a repo id already, or the registry can turn it into one". <c>"default"</c> and
/// <c>"auto"</c> are the deliberate exceptions: they are hardware-dispatch sentinels resolved at
/// load time, so no single repository id exists for them.
/// </para>
/// </summary>
public class WellKnownModelsResolveTests
{
    /// <summary>Sentinels that dispatch on hardware rather than naming one model.</summary>
    private static readonly string[] HardwareDispatchSentinels = ["default", "auto"];

    public static TheoryData<string, string> GeneratorConstants() => new()
    {
        { nameof(WellKnownModels.Generator.Default), WellKnownModels.Generator.Default },
        { nameof(WellKnownModels.Generator.Fast), WellKnownModels.Generator.Fast },
        { nameof(WellKnownModels.Generator.Small), WellKnownModels.Generator.Small },
        { nameof(WellKnownModels.Generator.Quality), WellKnownModels.Generator.Quality },
        { nameof(WellKnownModels.Generator.Medium), WellKnownModels.Generator.Medium },
        { nameof(WellKnownModels.Generator.Large), WellKnownModels.Generator.Large },
    };

    [Theory]
    [MemberData(nameof(GeneratorConstants))]
    public void GeneratorConstant_IsARepoIdOrResolvesToOne(string constantName, string value)
    {
        if (HardwareDispatchSentinels.Contains(value, StringComparer.OrdinalIgnoreCase))
            return;

        var resolved = GeneratorModelRegistry.Default.TryResolve(value, out var model) && model is not null
            ? model.ModelId
            : value;

        resolved.Should().Contain("/",
            $"WellKnownModels.Generator.{constantName} must reduce to an org/name repository id; " +
            "anything else cannot be matched against the model cache index");
    }
}
