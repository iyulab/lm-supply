using System.Globalization;
using LMSupply.Testing;
using LMSupply.Hardware;
using LMSupply.Reranker.Models;

namespace LMSupply.Reranker.Tests;

/// <summary>
/// What a reranker alias loads is part of the public behaviour: <c>default</c> moving from one repository to another changes
/// the scores an application ranks by, and nothing in the type system says so. This pins what each built-in alias
/// resolves to — repository, ONNX and tokenizer files, input length, licence, language coverage, architecture and output
/// shape — and the model <c>auto</c> picks on each hardware tier, so that a change makes the build red until
/// <c>catalog.snapshot</c> is edited in the same commit. Entries a user added to <c>aliases.json</c> are not part of it.
///
/// <para>
/// <b>When this fails:</b> read the lines that differ. If the change is intended, copy the printed lines into
/// <c>catalog.snapshot</c> and add a CHANGELOG line for each changed alias (<c>reranker:default: A → B (why)</c>);
/// <c>scripts/check-catalog-notes.ps1</c> reports a snapshot change that has no such line. See <c>docs/CATALOG.md</c>.
/// </para>
/// </summary>
public class CatalogSnapshotTests
{
    // One line per alias, sorted: reranker:<alias> | repository | ONNX file | ONNX data file | tokenizer repository | tokenizer file | input length | licence | languages | architecture | output
    private static List<string> Render()
    {
        var registry = new RerankerModelRegistry(RerankerModels.All);
        var lines = new List<string>();
        foreach (var alias in registry.GetAliases().Where(a => a.Kind == AliasKind.System && a.Name != "auto"))
        {
            var model = registry.Resolve(alias.Name);
            lines.Add(Line(
                "reranker:" + alias.Name, model.Id, model.OnnxFile, model.OnnxDataFile, model.TokenizerRepoId, model.TokenizerFile,
                model.MaxSequenceLength.ToString(CultureInfo.InvariantCulture), model.License, model.IsMultilingual ? "multilingual" : "english",
                model.Architecture.ToString(), model.OutputShape.ToString()));
        }

        // What `auto` picks depends on the machine; the mapping from a hardware tier is what the catalog says.
        lines.Add(Line("reranker:auto", "tiers", string.Join(",", Enum.GetValues<PerformanceTier>().Select(t => t + "=" + RerankerModelRegistry.ForTier(t).AliasName))));
        return [.. lines.OrderBy(l => l, StringComparer.Ordinal)];
    }

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
