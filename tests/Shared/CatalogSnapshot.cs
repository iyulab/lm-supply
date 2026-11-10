using System.Collections;
using System.Globalization;
using System.Reflection;

namespace LMSupply.Testing;

/// <summary>
/// What a catalog alias resolves to, as text: one line per built-in alias, sorted — <c>domain:alias | Property=value | …</c> for every public
/// property of the model, so that a field added to a model's description is pinned from the day it exists. The text a project's
/// <c>catalog.snapshot</c> records, and the build fails when the registry and the file differ. Aliases a user added to
/// <c>aliases.json</c> and <c>auto</c>, which depends on the machine, are not part of it; free text for a reader (the description and the
/// display name) is not either.
/// </summary>
internal static class CatalogSnapshot
{
    private static readonly HashSet<string> Left = new(StringComparer.Ordinal) { "AliasName", "Description", "DisplayName" };

    public static List<string> Render<T>(string domain, ModelRegistryBase<T> registry) where T : class, IModelInfoBase
    {
        var lines = new List<string>();
        foreach (var alias in registry.GetAliases().Where(a => a.Kind == AliasKind.System && a.Name != "auto"))
        {
            var model = registry.Resolve(alias.Name);
            var fields = model.GetType().GetProperties(BindingFlags.Public | BindingFlags.Instance)
                .Where(p => p.GetIndexParameters().Length == 0 && !Left.Contains(p.Name))
                .OrderBy(p => p.Name, StringComparer.Ordinal)
                .Select(p => p.Name + "=" + Format(p.GetValue(model)));
            lines.Add(string.Join(" | ", [domain + ":" + alias.Name, .. fields]));
        }

        return [.. lines.OrderBy(l => l, StringComparer.Ordinal)];
    }

    /// <summary>Fails with the lines that differ when <paramref name="actual"/> is not what <c>catalog.snapshot</c> next to the test assembly records.</summary>
    public static void Verify(IReadOnlyCollection<string> actual)
    {
        var path = Path.Combine(AppContext.BaseDirectory, "catalog.snapshot");
        var expected = File.ReadAllText(path).Split('\n').Select(l => l.TrimEnd('\r')).Where(l => l.Length > 0 && !l.StartsWith('#')).ToList();
        var missing = expected.Except(actual, StringComparer.Ordinal).ToList();
        var added = actual.Except(expected, StringComparer.Ordinal).ToList();
        if (missing.Count + added.Count == 0)
        {
            return;
        }

        Xunit.Assert.Fail(
            "the catalog changed without catalog.snapshot being edited with it.\n"
            + "Recorded but no longer true:\n  " + string.Join("\n  ", missing)
            + "\nTrue now but not recorded:\n  " + string.Join("\n  ", added)
            + "\nReplace the first set with the second in catalog.snapshot, and add a CHANGELOG line for each changed alias (docs/CATALOG.md).");
    }

    private static string Format(object? value) => value switch
    {
        null => "",
        string text => text,
        IEnumerable items => "[" + string.Join(",", items.Cast<object?>().Select(Format)) + "]",
        IFormattable number => number.ToString(null, CultureInfo.InvariantCulture),
        _ => value.ToString() ?? "",
    };
}
