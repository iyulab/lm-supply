using AwesomeAssertions;
using LMSupply.Generator.Internal.Llama;

namespace LMSupply.Generator.Tests;

/// <summary>
/// GGUF aliases are a public surface addressed by string, and the type system does not defend it.
///
/// <para>
/// On 2026-08-06 a consumer upgrading across a minor version found <c>gguf:default</c> gone: the
/// registry had moved to family-scoped names. The removal itself was a reasonable design change —
/// what was not reasonable is that <b>nothing signalled it</b>. The compiler cannot: aliases are
/// strings. This package keeps no CHANGELOG. The umbrella's changelog-coverage gate skips packages
/// without one, by design. So three separate defences let it through and the consumer learned about
/// it from an exception at runtime.
/// </para>
///
/// <para>
/// This snapshot is the missing signal. It does <b>not</b> forbid removing or renaming an alias —
/// pre-1.0, that is a normal thing to do. It makes the removal <b>visible to the person doing it</b>:
/// the build goes red, they update the list, and the diff then says plainly which public name
/// disappeared. A reviewer sees it; a release note can quote it.
/// </para>
///
/// <para>
/// <b>When this fails:</b> read the diff. If the change is intended, update <see cref="KnownAliases"/>
/// in the same commit and mention the removed name where consumers will see it. Do not delete this
/// test — the whole value is that the list has to be edited deliberately.
/// </para>
/// </summary>
public class GgufAliasSurfaceSnapshotTests
{
    /// <summary>
    /// Every alias this package promises to resolve, as of 0.38.0. Sorted, so the diff on change is
    /// minimal and readable.
    /// </summary>
    private static readonly string[] KnownAliases =
    [
        "gguf:auto",
        "gguf:gemma4-balanced",
        "gguf:gemma4-default",
        "gguf:gemma4-fast",
        "gguf:gemma4-large",
        "gguf:gemma4-quality",
        "gguf:phi-4-mini",
        "gguf:qwen2.5-7b",
        "gguf:qwen3-balanced",
        "gguf:qwen3-default",
        "gguf:qwen3-fast",
        "gguf:qwen3-large",
        "gguf:qwen3-quality",
        "gguf:xlarge"
    ];

    [Fact]
    public void TheAliasSurface_MatchesItsRecordedSnapshot()
    {
        var actual = GgufModelRegistry.GetAliases().OrderBy(a => a, StringComparer.Ordinal).ToList();
        var expected = KnownAliases.OrderBy(a => a, StringComparer.Ordinal).ToList();

        actual.Should().Equal(
            expected,
            "aliases are a string-addressed public surface; adding or removing one has to be a "
            + "deliberate edit here so the change is visible in the diff rather than at a consumer's runtime");
    }

    [Theory]
    [MemberData(nameof(EveryKnownAlias))]
    public void EveryRecordedAlias_StillResolves(string alias)
    {
        // The list above could drift into fiction on its own — a name kept here after the registry
        // dropped it would make the snapshot assert against itself. Resolving each one keeps the
        // record tied to behaviour.
        GgufModelRegistry.Resolve(alias).Should().NotBeNull(
            "'{0}' is recorded as part of the public alias surface", alias);
    }

    public static TheoryData<string> EveryKnownAlias()
    {
        var data = new TheoryData<string>();
        foreach (var alias in KnownAliases)
        {
            data.Add(alias);
        }
        return data;
    }
}
