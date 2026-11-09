# The model catalog: how it is kept current

LMSupply ships catalogs of models: aliases such as `gguf:qwen3-balanced` or `fast` that the library resolves to a
repository and a file, and a pool that `gguf:auto` picks from. The library chooses what each alias loads, so a consumer
that shows a model's name or size next to an alias depends on what the library did with the catalog since it last looked.
This page says when the catalog is reviewed and how a change reaches you.

## Review cadence

| What | Reviewed |
|---|---|
| Models of the generator (GGUF and ONNX), embedder and reranker catalogs: newer releases of a family, better quantizations, licence changes, entries to retire | at least monthly |
| Models of the other tasks (transcriber, synthesizer, OCR, translator, captioner, detector, segmenter, image generator) | at least every two months |
| Alias assignments: which model each tier alias (`fast`, `default`, `balanced`, `quality`, `large`) names, the default quantization, the automatic-selection pool | at least every two months, after the model review |

A review that finds nothing worth changing changes nothing; the cadence is how often the catalog is looked at, not a
promise of a change.

## Every change is a release-note line

A change of what an alias resolves to is behaviour, even though no signature changes. The release notes carry one line
for each of these, in the section of the release that ships it:

- an alias repointed to another repository or model: `qwen3-balanced: Qwen3 8B → Qwen 3.5 9B (why)`
- a different default file or quantization for an alias
- a known-issue tag added to or removed from a model (for example `thinking-enabled-by-default`)
- an alias retired or added, or a model moved into or out of the automatic-selection pool
- an entry of the ONNX generator catalog changed in the same ways

A consumer that keeps its own label for an alias can pin it with `GgufModelRegistry.Resolve(alias).RepoId` in a test,
and update the label in the same pass as the upgrade.

## For contributors

`tests/LMSupply.Generator.Tests/catalog.snapshot` records, for every alias of the GGUF and ONNX generator catalogs, the
repository, file, quantization and known-issue tags it resolves to. `CatalogSnapshotTests` fails when the registries and
the file differ, and prints the lines that changed. Edit the file in the same commit as the registry, then run

```
pwsh scripts/check-catalog-notes.ps1
```

which compares the snapshot with the one at the upstream branch and lists every changed alias that has no line naming it
in the `CHANGELOG.md` additions of the same range. The alias-name list in `GgufAliasSurfaceSnapshotTests` stays as the
record of the public alias surface.
