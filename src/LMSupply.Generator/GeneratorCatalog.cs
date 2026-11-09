using System.Security.Cryptography;
using System.Text;
using LMSupply.Core;
using LMSupply.Generator.Gguf;

namespace LMSupply.Generator;

/// <summary>
/// The generator catalogs as a whole — the GGUF aliases and the ONNX ones.
/// </summary>
public static class GeneratorCatalog
{
    /// <summary>
    /// A short stable identifier of what the catalogs say: which repository, file and quantization each alias loads, each
    /// model's known-issue tags and tier, the pool automatic selection draws from, and the ONNX entries. It changes when
    /// the catalog changes — an alias repointed, a tag added, a model retired — and only then, so an application can tell
    /// "the catalog changed since I last looked" by comparing it with the value it stored. Entries a user added to
    /// <c>aliases.json</c> are not part of it. The value is 16 lowercase hexadecimal characters.
    /// </summary>
    public static string Revision { get; } = Compute();

    private static string Compute()
    {
        var text = string.Join('\n', Render()) + '\n';
        return Convert.ToHexString(SHA256.HashData(Encoding.UTF8.GetBytes(text)))[..16].ToLowerInvariant();
    }

    /// <summary>
    /// The catalogs in their canonical text: one line per alias, sorted — <c>alias | repository | file | quantization |
    /// known-issue tags | tier</c> — plus one line for the automatic-selection pool. <see cref="Revision"/> hashes it, and the
    /// repository's <c>catalog.snapshot</c> records it.
    /// </summary>
    internal static IReadOnlyList<string> Render()
    {
        var lines = new List<string>();

        foreach (var alias in GgufModelRegistry.GetAliases().Where(a => !a.Equals("gguf:auto", StringComparison.OrdinalIgnoreCase)))
        {
            var model = GgufModelRegistry.Resolve(alias)
                ?? throw new InvalidOperationException($"'{alias}' is listed by GetAliases() but does not resolve.");
            lines.Add(Line(alias, model.RepoId, model.DefaultFile, model.QuantizationType,
                string.Join(",", model.KnownIssues.OrderBy(i => i, StringComparer.Ordinal)),
                model.Tier?.ToString()));
        }

        // The pool automatic selection draws from is part of what the catalog says; moving a model in or out of it
        // changes what "auto" can pick.
        lines.Add(Line("gguf:auto", "pool", string.Join(",", GgufModelRegistry.GetAutoSelectionAliases().OrderBy(a => a, StringComparer.Ordinal))));

        var onnx = new GeneratorModelRegistry(DefaultGeneratorModels.All);
        foreach (var alias in onnx.GetAliases().Where(a => a.Kind == AliasKind.System && a.Name != "auto"))
        {
            var model = onnx.Resolve(alias.Name);
            lines.Add(Line("onnx:" + alias.Name, model.ModelId, model.Subfolder, model.DefaultQuantization.ToString()));
        }

        return lines.OrderBy(l => l, StringComparer.Ordinal).ToList();
    }

    private static string Line(params string?[] fields)
    {
        var parts = fields.Select(f => f ?? "").ToList();
        while (parts.Count > 1 && parts[^1].Length == 0) parts.RemoveAt(parts.Count - 1);
        return string.Join(" | ", parts);
    }
}
