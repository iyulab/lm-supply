using AwesomeAssertions;

namespace LMSupply.Generator.Tests;

/// <summary>
/// What a catalog alias loads is part of the public behaviour: <c>gguf:qwen3-balanced</c> moving from one repository to
/// another changes the model an application downloads and runs, and nothing in the type system says so. The alias
/// names are pinned by <see cref="GgufAliasSurfaceSnapshotTests"/>; this pins what each name resolves to, so a repoint,
/// a different default file or quantization, a change of a known-issue tag, or a change of the automatic-selection
/// pool makes the build red until <c>catalog.snapshot</c> is edited in the same commit.
///
/// <para>
/// <b>When this fails:</b> read the lines that differ. If the change is intended, copy the printed lines into
/// <c>catalog.snapshot</c> and add a CHANGELOG line for each changed alias (<c>qwen3-balanced: A → B (why)</c>);
/// <c>scripts/check-catalog-notes.ps1</c> reports a snapshot change that has no such line. See
/// <c>docs/CATALOG.md</c>.
/// </para>
/// </summary>
public class CatalogSnapshotTests
{
    private static string SnapshotPath => Path.Combine(AppContext.BaseDirectory, "catalog.snapshot");

    [Fact]
    public void TheCatalog_MatchesItsRecordedSnapshot()
    {
        var expected = ReadLines(File.ReadAllText(SnapshotPath));
        var actual = GeneratorCatalog.Render();

        var missing = expected.Except(actual, StringComparer.Ordinal).ToList();
        var added = actual.Except(expected, StringComparer.Ordinal).ToList();

        (missing.Count + added.Count).Should().Be(0,
            "the catalog changed without catalog.snapshot being edited with it.\n"
            + "Recorded but no longer true:\n  " + string.Join("\n  ", missing)
            + "\nTrue now but not recorded:\n  " + string.Join("\n  ", added)
            + "\nReplace the first set with the second in catalog.snapshot, and add a CHANGELOG line for each changed alias.");
    }

    private static List<string> ReadLines(string text)
        => text.Split('\n').Select(l => l.TrimEnd('\r')).Where(l => l.Length > 0 && !l.StartsWith('#')).ToList();
}
