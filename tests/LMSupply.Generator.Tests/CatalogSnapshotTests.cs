using LMSupply.Testing;

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
    [Fact]
    public void TheCatalog_MatchesItsRecordedSnapshot() => CatalogSnapshot.Verify(GeneratorCatalog.Render());
}
