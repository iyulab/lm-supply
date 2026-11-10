using System.Globalization;
using LMSupply.Testing;
using LMSupply.Embedder.Utils;

namespace LMSupply.Embedder.Tests;

/// <summary>
/// What an embedder alias loads is part of the public behaviour: <c>default</c> moving from one repository to another, or a
/// model's pooling or query prefix changing, changes the vectors an application stores, and nothing in the type system
/// says so. This pins what each built-in alias resolves to — repository, subfolder, dimensions, input length, pooling,
/// lower-casing, licence and the three prefixes — and the pool <c>auto</c> draws from, so that a change makes the build
/// red until <c>catalog.snapshot</c> is edited in the same commit. Entries a user added to <c>aliases.json</c> are not part
/// of it.
///
/// <para>
/// <b>When this fails:</b> read the lines that differ. If the change is intended, copy the printed lines into
/// <c>catalog.snapshot</c> and add a CHANGELOG line for each changed alias (<c>embedder:default: A → B (why)</c>);
/// <c>scripts/check-catalog-notes.ps1</c> reports a snapshot change that has no such line. See <c>docs/CATALOG.md</c>.
/// </para>
/// </summary>
public class CatalogSnapshotTests
{
    // One line per alias, sorted: embedder:<alias> | repository | subfolder | dimensions | input length | pooling | lower-case | licence | query | passage | default prefix
    private static List<string> Render()
    {
        var registry = new EmbedderModelRegistry(DefaultModels.All);
        var lines = new List<string>();
        foreach (var alias in registry.GetAliases().Where(a => a.Kind == AliasKind.System && a.Name != "auto"))
        {
            var model = registry.Resolve(alias.Name);
            lines.Add(Line(
                "embedder:" + alias.Name, model.RepoId, model.Subfolder,
                model.Dimensions.ToString(CultureInfo.InvariantCulture), model.MaxSequenceLength.ToString(CultureInfo.InvariantCulture),
                model.PoolingMode.ToString(), model.DoLowerCase ? "lower" : "keep-case", model.License,
                Quoted(model.QueryPrefix), Quoted(model.PassagePrefix), Quoted(model.DefaultPrefix)));
        }

        // The order of the pool is part of what it says: the largest model that fits is the one chosen.
        lines.Add(Line("embedder:auto", "pool", string.Join(",", EmbedderModelRegistry.AutoCandidates.Select(m => m.AliasName))));
        return [.. lines.OrderBy(l => l, StringComparer.Ordinal)];
    }

    private static string Quoted(string? prefix) => prefix is null ? "" : "\"" + prefix + "\"";

    private static string Line(params string?[] fields)
    {
        var parts = fields.Select(f => f ?? "").ToList();
        while (parts.Count > 1 && parts[^1].Length == 0)
        {
            parts.RemoveAt(parts.Count - 1);
        }

        return string.Join(" | ", parts);
    }

    [Fact]
    public void TheCatalog_MatchesItsRecordedSnapshot() => CatalogSnapshot.Verify(Render());
}
