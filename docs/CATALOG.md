# The model catalog: how it is kept current

LMSupply ships catalogs of models: aliases such as `gguf:balanced` or `entry` that the library resolves to a
repository and a file, and a pool that `gguf:auto` picks from. The library chooses what each alias loads, so a consumer
that shows a model's name or size next to an alias depends on what the library did with the catalog since it last looked.
This page says when the catalog is reviewed and how a change reaches you.

## The ladder: one set of names in every domain

Every domain's catalog is a ladder of up to five rungs — `entry`, `basic`, `balanced`, `advanced`, `frontier`
(`ModelTier`), ordered by capability and cost — and the name of a rung is an alias of its model in that domain's
registry: `LocalEmbedder.LoadAsync("entry")`, `LocalTranscriber.LoadAsync("frontier")`, `LocalGenerator.LoadAsync("gguf:balanced")`.

- **A rung names one distinct model.** Presets of one model (the image generator's step counts), voices and language
  variants are not rungs. A domain fills only the rungs it has a model for; `registry.GetTiers()` lists them and
  `ResolveTier(tier, TierFallback.Lower | Higher)` takes the nearest one when the caller asks for that. An absent rung
  is `ModelNotFoundException` naming the rungs the domain holds — never a neighbour.
- **A rung's name carries no model family or generation**, so it stays right when the catalog moves to a newer model:
  `gguf:balanced` was Qwen3 8B and is Qwen 3.5 9B. What a rung loads changes only in a release, with a line in the
  notes (below). To stay on one model, name the model (`multilingual-e5-large`, `gguf:qwen3.6-35b-a3b-q4_k_m`) or give
  its repository id.
- **`default` is the model the domain recommends** — wherever the domain has a `basic` rung, that one.
- **The size adjectives `fast`, `quality`, `large` and `xlarge` are not aliases.** A retired name loads nothing, and the
  "model not found" message says what replaced it.
- The rungs are the library's words; an application shows its own. A rung is the domain's own step up — `frontier` of
  the transcriber (Whisper large-v3) and `advanced` of the reranker are not the same size.

## Review cadence

| What | Reviewed |
|---|---|
| Models of the generator (GGUF and ONNX), embedder and reranker catalogs: newer releases of a family, better quantizations, licence changes, entries to retire | at least monthly |
| Models of the other tasks (transcriber, synthesizer, OCR, translator, captioner, detector, segmenter, image generator) | at least every two months |
| Alias assignments: which model each rung alias (`entry`, `basic`, `balanced`, `advanced`, `frontier`) and `default` name, the default quantization, the automatic-selection pool | at least every two months, after the model review |

A review that finds nothing worth changing changes nothing; the cadence is how often the catalog is looked at, not a
promise of a change.

## Every change is a release-note line

A change of what an alias resolves to is behaviour, even though no signature changes. The release notes carry one line
for each of these, in the section of the release that ships it:

- an alias repointed to another repository or model: `balanced: Qwen3 8B → Qwen 3.5 9B (why)`
- a different default file or quantization for an alias
- a known-issue tag added to or removed from a model (for example `thinking-enabled-by-default`)
- an alias retired or added, or a model moved into or out of the automatic-selection pool
- an entry of the ONNX generator catalog changed in the same ways

A consumer that keeps its own label for an alias can pin it with `GgufModelRegistry.Resolve(alias).RepoId` in a test,
and update the label in the same pass as the upgrade.

## For contributors

`tests/LMSupply.Generator.Tests/catalog.snapshot` records, for every alias of the GGUF and ONNX generator catalogs, the
repository, file, quantization, known-issue tags and recommendation tier it resolves to; `GeneratorCatalog.Revision` is the hash of the same text, so it changes when the file would. `CatalogSnapshotTests` fails when the registries and
the file differ, and prints the lines that changed. Every other domain keeps a `catalog.snapshot` of its own in the same
way, in its test project. The embedder's and the reranker's record the fields that shape the vectors or scores — an embedder's
dimensions, pooling and prefixes, a reranker's architecture and output — plus the pool or tier mapping `auto` draws from; the
transcriber's, synthesizer's, OCR's (detection and recognition), translator's, captioner's, detector's, segmenter's and image
generator's record every public property of each built-in model, so that a property added later is pinned from the day it exists.
`auto` itself depends on the machine and is not recorded for those. Edit the file in the
same commit as the registry, then run

```
pwsh scripts/check-catalog-notes.ps1
```

which compares the snapshots with the ones at the upstream branch and lists every changed alias that has no line naming it
in the `CHANGELOG.md` additions of the same range. An embedder or reranker alias is named with its domain
(`embedder:default: BGE-M3 → … (why)`; for the two OCR catalogs `ocr-detection:default` or `ocr` beside the alias). The alias-name list in `GgufAliasSurfaceSnapshotTests` stays as the
record of the public alias surface.
