# Changelog

All notable changes to this project are documented in this file. Versions follow
[Semantic Versioning](https://semver.org/); while the major version is 0, a minor release may contain
breaking changes, and each one is marked **Breaking** with a migration note.

## [0.72.1] - 2026-09-23

### Fixed

- **ONNX models report `FinishReason = "length"` when a completion stops at `GenerationOptions.MaxTokens` or fills
  the model's maximum sequence length.** `GenerateChatWithToolsAsync` and the last chunk of `GenerateChatStreamAsync`
  answered `"stop"` for every completion without tool calls, so a response cut off at the token limit looked
  finished. They now say `"stop"` only when the model produced its end token or a stop sequence. The llama-server
  path already reported the server's reason and is unchanged.
- **A GGUF embedding repository applies the query/passage prefixes of the model it was converted from.**
  `EmbedQueryAsync` / `EmbedPassageAsync` on `nomic-ai/nomic-embed-text-v1.5-GGUF` (the GGUF example in the README)
  embedded bare text, while the ONNX build of the same model prefixed `search_query: ` / `search_document: ` — a GGUF
  file carries no sentence-transformers prompts and the GGUF path reported none. A repository named
  `<org>/<model>-GGUF` now takes the catalog entry of `<org>/<model>` (a local `.gguf` file or an unknown model still
  has none). The prefixes are part of the vector space, so `VectorSpaceRevision` changes for such a model: vectors
  stored before this release were made without the prefix and should be re-embedded.

## [0.72.0] - 2026-09-22

### Added

- **`LocalEmbedder.GetVectorSpaceRevisionAsync(modelIdOrPath, options?, ct)`** — the `VectorSpaceRevision` a
  load would report, from the cached files alone: no inference session, no download, no request. It goes
  through the same resolution as `LoadAsync` (model file, tokenizer as built from its files, pooling,
  normalization, sequence length, prefixes), so the two agree by construction for everything but the
  dimension, which the load reads from the ONNX graph and this method from the catalog entry or the
  repository's `config.json` (`hidden_size`); a model whose graph width differs from its declared hidden
  size gets a different value here, and the load traces a warning for it. Returns `null` when the answer
  needs a load: the model is not cached, the id is unknown, no dimension is declared, or the model is GGUF
  (llama-server decides its dimension). For a consumer that names its vector store after the embedding
  identity before the model is loaded and wants the revision in that name without forcing a warm-up.

## [0.71.0] - 2026-09-21

### Changed

- **The embedder truncates where the model says it does.** A sentence-transformers model declares
  its sequence length in `sentence_bert_config.json` (`max_seq_length` - 256 for
  `all-MiniLM-L6-v2`, not the 512 its architecture allows). The file was neither downloaded nor
  read, so a model loaded by repository id ran at the 512 default and its embeddings diverged from
  the reference implementation on inputs longer than the declared length. It is now downloaded
  (an existing cache picks it up on the next online load) and used when the caller left
  `EmbedderOptions.MaxSequenceLength` at its default: the model's declaration first, then the
  catalog entry for a known alias, then 512. **An explicit value still wins.** After loading,
  `MaxSequenceLength` holds the length in effect. **Behaviour change:** for a model that declares a
  length other than the one previously used, inputs longer than it embed differently - vectors
  stored from such inputs need re-embedding.
- **Breaking: `EmbedderOptions.MaxSequenceLength` is `int?` and `EmbedderOptions.PoolingMode` is
  `PoolingMode?`, both `null` by default.** `null` means "the model decides": its own declaration
  (`sentence_bert_config.json` for the length, `1_Pooling/config.json` for the pooling), then the
  catalog entry for a known alias, then 512 / `Mean`. A value is used as given — including exactly 512,
  which the `int` could not express. After loading, both hold the value in effect. Assignments compile
  unchanged; code that *reads* either property as a non-nullable value handles `null` (or reads
  `GetModelInfo()` after loading, which is where the effective values are). `LMSupply.Reranker` already
  used `int?` for the same option; the embedder now matches it.
- **A model loaded by repository id is pooled the way it declares, and reports a `ModelInfo`.** Without a
  catalog entry the loader pooled every such model with the option's default (`Mean`) whatever its
  `1_Pooling/config.json` said — a CLS model such as `BAAI/bge-small-en-v1.5` loaded by id produced
  vectors in a different space from the same model loaded by alias — and `GetModelInfo()` was `null`, so
  `EmbedQueryAsync`/`EmbedPassageAsync` applied no prefix. The loader now downloads and reads
  `1_Pooling/config.json`, `modules.json` and `config_sentence_transformers.json`, builds a `ModelInfo`
  from the model's own files (dimensions from the session, the effective length and pooling, and the
  `prompts` the repository declares — most declare none), and a catalog `ModelInfo` reports the length and
  pooling actually in effect. **Behaviour change:** a repository-id model whose file declares CLS or max
  pooling embeds differently from before — re-embed vectors stored from it. A pooling this library does
  not implement (weighted mean, last token) is treated as undeclared.

- **A model file the cache already holds at another place in the same snapshot is moved, not downloaded
  again.** 0.63.0 (2026-09-11) moved a subfolder's files from the snapshot root into the subfolder; the
  new path looked in one place only, so every install that had the model before 0.63.0 fetched it once
  more on the first load after updating — about 1 GB for the default embedder — and kept both copies
  (one dogfooding cache held 6.8 GB of byte-identical pairs). That release note was never written; this
  is it. The downloader now looks for a wanted file at the snapshot root (when the target is a
  subfolder) or in an immediate subfolder (when the target is the root) and moves a copy of the listed
  length into place: no request, one copy. Copies of another length are left alone; nothing is moved in
  offline (read-only) mode. Existing duplicate pairs are not removed by this — a reclaim call is a
  separate item.

- **Reranker `auto` on a Medium host takes the GGUF multilingual model when a llama-server binary is
  already cached.** Medium resolved to `quality` (bge-reranker-base), whose model card lists English and
  Chinese as its training languages — on a Korean corpus it ranked worse than no reranking (a consumer's
  measurement, #376). When a llama-server binary is already in the cache, `auto` now resolves to
  `multilingual-fast` (bge-reranker-v2-m3 Q4_K_M: a fifth of the download, a fraction of the CPU latency,
  the same ranking as the ONNX `multilingual`); without one it still resolves to `quality`, so `auto`
  never fetches a server binary on its own. Low, High and Ultra are unchanged. `LlamaServerDownloader.IsAnyServerCached()`
  is the probe (a file check, no request). **Behaviour change** for Medium hosts that already run
  llama-server: a different model, scores on the same 0..1 scale.
- **`EnvironmentDetector.DetectGpu()` / `DetectAllGpus()` / `DetectPlatform()` could return `null`
  from a non-nullable API** when `ClearCache()` ran on another thread: the cached value was assigned
  under the lock but read again outside it on the way out. Measured with a fact that runs 80
  concurrent detections: 2 of 6 runs threw `ArgumentNullException` before, 0 of 8 after. The NVML
  session (one process-wide handle whose init → enumerate → shutdown each detection owns) is
  serialized in the same change as a precaution; a vendor flip between two detections seen once on a
  dual-GPU machine was **not** reproduced by that fact with or without the serialization, so its cause
  is not established by this release.

### Added

- **`IEmbeddingModel.VectorSpaceRevision` — an opaque string that changes when, and only when, this
  library would produce different vectors for the same model id.** Three releases (`#195`, 0.70.0 and
  this one) changed the vectors a model id produces and said so only in prose; a consumer that stores
  vectors had no value to compare. The revision is derived from what the loader actually did — the
  tokenizer and its normalization convention, pooling, L2 normalization, the query/passage prefixes,
  the sequence length in effect, the model file opened (a quantization variant is a different space)
  and a per-component implementation epoch (WordPiece and SentencePiece each carry their own, raised
  only when a release changes the ids they produce for the same files) — so a WordPiece fix moves
  WordPiece models only, and a model loaded by repository id is covered by the same code as an alias.
  Not part of it: the execution provider/GPU, and for GGUF models the llama-server binary version.
  **Contract:** store it next to the vectors; when a later load reports a different value, the stored
  vectors are stale for that model. The interface member has a default implementation (`null`), so a
  consumer's own `IEmbeddingModel` keeps compiling. The canonical line behind the hash is traced at
  load (`[LocalEmbedder.vectorspace]`) so two revisions can be diffed. Teeth: golden-vector facts
  (`LocalOnly`, CPU provider) fail when the vectors move and the revision does not, and when the
  revision moves and the vectors do not — measured red on three mutations before shipping
  (normalization switched off · basic tokenization removed · an epoch raised alone).
- **`CacheManager.FindReclaimable(cacheDir)` / `Reclaim(cacheDir, files)` — free the duplicates a
  layout change left behind.** 0.63.0 moved a subfolder's files from the snapshot root into the
  subfolder and every existing cache downloaded them again, keeping both copies (one dogfooding cache
  measured 6.8 GB of byte-identical pairs). `FindReclaimable` lists, largest first, each root file whose
  same-name twin in a subfolder of the same snapshot has the same length and SHA-256 **and** is the copy
  a manifest lists as read — the measured case and nothing wider; a root file with no twin, a twin of
  another length or content, or one no manifest knows is never reported. The list is the dry run
  (`RepoId`, `Path`, `Size`, `TwinPath`, a sentence to show); `Reclaim` deletes what it is handed, after
  re-checking that each file still exists, lies inside the cache directory and still has its twin, and
  returns the bytes freed. After reclaiming, the model loads from the cache without a request
  (fixture-tested). Adopt-before-fetch (below) prevents new pairs; this removes the ones already there.
- **Breaking (`LMSupply.Text.Core`): `ISequenceTokenizer.Signature`.** The algorithm, normalization
  convention and implementation epoch of a tokenizer as a short ASCII string — the tokenizer's input to
  the revision above. Implementations of `ISequenceTokenizer`/`IPairTokenizer` outside this library
  add the property (any stable string that changes when the ids the tokenizer produces change).
- `EmbedderOptions.DefaultMaxSequenceLength`.
- **Special tokens typed into the input are ordinary text - now a tested guarantee.** `"a [SEP] b"`
  tokenizes as `a [ sep ] b`, so user content cannot inject a separator or classifier token; the
  special ids appear exactly where the tokenizer puts them. This differs from sentence-transformers,
  which maps such text to the special ids, and costs cosine agreement on inputs that contain them.
  The behaviour itself is unchanged from 0.70.0; it was a side effect and is now pinned for cased and
  uncased vocabularies.

## [0.70.0] - 2026-09-21

### Fixed

- **Breaking — WordPiece (BERT-family) models are tokenized the way they were trained.** The WordPiece
  path looked whitespace-separated words up in the vocabulary and did nothing else: no lowercasing, no
  accent stripping, no splitting of punctuation or CJK ideographs. On an uncased vocabulary that made
  every capitalized word (`The`, `Paris`), every word with punctuation attached (`dog?!`, `France.`)
  and every accented word (`Café`) an `[UNK]`, so embeddings and relevance scores matched the
  reference implementation only for lowercase, space-separated ASCII. BERT's basic tokenization now
  runs first, configured from what the model declares — `tokenizer.json` (`BertNormalizer`,
  `BertPreTokenizer`), then `tokenizer_config.json` (`do_lower_case`, `strip_accents`,
  `tokenize_chinese_chars`); a model that ships neither is treated as cased when its vocabulary holds
  uppercase pieces. Token ids now equal the HuggingFace `tokenizers` output, including for tab/newline
  separated words and for ASCII symbols such as `$ + =`.
  Affected: every model that loads through `vocab.txt` or a WordPiece `tokenizer.json` — in the
  embedder `bge-base-en-v1.5`, `bge-large-en-v1.5`, `e5-small-v2`, `e5-base-v2`, `all-mpnet-base-v2`,
  `nomic-embed-text-v1.5` and any such repository loaded by id (e.g. `all-MiniLM-L6-v2`); in the
  reranker `default`, `fast` and `ms-marco-l12`. The SentencePiece models (`default`/`quality` =
  bge-m3, `fast`/`large` = multilingual-e5, reranker `quality`/`large`/`multilingual`) are untouched.
  Migration: **vectors stored from an affected model were computed from the old token ids — re-embed
  them**, or queries and documents will disagree wherever text has capitals or punctuation. Reranker
  scores from the affected aliases change (towards the model's own), so re-check any threshold
  calibrated on them.

## [0.69.0] - 2026-09-21

### Added

- **`LMSupply.Reranker`: the alias `multilingual-fast`** loads bge-reranker-v2-m3 — the model behind
  `multilingual` — as a Q4_K_M GGUF through llama-server: about 440 MB instead of 2.3 GB and a fraction
  of the CPU latency, with the same 0..1 scores. It works on `LoadAsync`, `IsModelDownloaded` and
  `DownloadModelAsync`, accepts a quantization qualifier (`multilingual-fast:Q8_0`), and a user alias of
  the same name overrides it. `multilingual` and `auto` still resolve to ONNX: nothing starts a
  llama-server process unless you ask for this alias or a `gguf:` id.

### Changed

- **Breaking — `LMSupply.Reranker`: a GGUF reranker now scores on the same 0..1 scale as an ONNX one.**
  The llama-server route returned the classifier's raw logit (e.g. `2.8`, and negative for most
  candidates) where `RankedResult.Score` documents a relevance between 0 and 1 and the ONNX route has
  always delivered one. The logit now goes through the same sigmoid. Ranking is unchanged (the mapping
  is monotonic); a threshold you calibrated on the ONNX scale now means the same thing on a `gguf:`
  model, and a downstream "drop scores below 0" default no longer empties the result list.
  Migration: if you compared GGUF scores against a logit threshold `t`, compare against
  `1 / (1 + exp(-t))` instead.

### Fixed

- **`LocalReranker.IsModelDownloaded` answers for GGUF models.** It had no GGUF branch, so a `gguf:`
  model that was cached and loaded fine was always reported as not downloaded. It now looks where the
  GGUF loader looks and picks the file the loader picks — `true` exactly when a load with
  `DisableAutoDownload = true` would find its file. Git LFS pointers are not counted as a model.
- **`LocalReranker.DownloadModelAsync` fetches a GGUF model the way the loader does** — one file, into
  the loader's cache layout. It used to fall through to the raw-repository download, which pulls every
  quantization of the repository into a layout the GGUF loader never reads.
- **An interrupted GGUF reranker download no longer leaves a truncated model in the cache.** The
  reranker's GGUF downloader wrote straight to the final file name; it now uses the same resumable
  download as the embedder and generator (`.part` until complete, length checked against the
  repository listing), so a cached file of the wrong length is fetched again instead of being loaded.
- **`RerankerOptions.QuantizationHint` is honoured on the GGUF route** (e.g. `"Q8_0"`). It was read by
  nothing there; the route always asked for `Q4_K_M`, which stays the default.
- **Reranker model sizes and languages are stated as they are.** `quality` (bge-reranker-base) declared
  440 MB; the fp32 ONNX it downloads is 1.1 GB — `ModelInfo.SizeBytes`, and with it the pool's memory
  estimate, were off by 2.5×. The package README listed `large` and `multilingual` at half their download
  size. `quality` and `large` were described as multilingual; their model cards list English and Chinese,
  and the docs no longer recommend them for other languages.
- The GGUF error message, XML docs and `docs/reranker.md` named `BAAI/bge-reranker-v2-m3-GGUF` and
  `jinaai/...-GGUF`, neither of which exists on HuggingFace. They now name `gpustack/` repositories
  that do. The docs also said the largest quantization that fits in memory is chosen; the route asks
  for `Q4_K_M` (or your hint) first and only falls back to that rule.

## [0.68.3] - 2026-09-19

This file starts at 0.68.3. Changes in earlier releases were not recorded here; the commit history is
the record for them.
