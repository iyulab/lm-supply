using LMSupply.ImageGenerator.Models;
using LMSupply.Testing;

namespace LMSupply.ImageGenerator.Tests;

/// <summary>
/// What a built-in alias of the ImageGenerator catalog loads is part of the public behaviour: an alias moving to another repository, or a
/// model's files, input size or licence changing, changes what an application downloads and runs, and nothing in the type system says
/// so. This pins every public property of each built-in model, so a change makes the build red until <c>catalog.snapshot</c> is
/// edited in the same commit. If the change is intended, copy the printed lines into <c>catalog.snapshot</c> and add a CHANGELOG line
/// naming the alias with its domain; <c>scripts/check-catalog-notes.ps1</c> reports a change that has none. See <c>docs/CATALOG.md</c>.
/// </summary>
public class CatalogSnapshotTests
{
    [Fact]
    public void TheCatalog_MatchesItsRecordedSnapshot() => CatalogSnapshot.Verify(CatalogSnapshot.Render("image-generator", new ImageGeneratorModelRegistry(ImageGeneratorModels.All)));
}
