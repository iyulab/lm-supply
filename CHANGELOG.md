# Changelog

All notable changes to this project are documented in this file. Versions follow
[Semantic Versioning](https://semver.org/); while the major version is 0, a minor release may contain
breaking changes, and each one is marked **Breaking** with a migration note.

## [0.79.1] - Unreleased

### Fixed
- **Loading a GGUF model on Windows no longer waits about two extra seconds per connection.** The llama-server listens on `127.0.0.1`, but the startup health poll and every client connected to `http://localhost:…`, which resolves to `::1` first; on Windows a refused IPv6 connect takes about 2 s (measured 2,042 ms against 2 ms). Both now use `LlamaServerProcess.LoopbackHost` (`127.0.0.1`), and `LlamaServerInfo.BaseUrl` returns that address.

## [0.79.0] - 2026-09-26

### Added

- **Transcription can say who spoke.** `TranscribeOptions.Diarize = true` labels each `TranscriptionSegment.Speaker`
  (`"S1"`, `"S2"`, … in speaking order) with local speaker diarization — pyannote segmentation-3.0 plus a WeSpeaker
  speaker embedding, fetched on first use into the model cache. `NumSpeakers` fixes the count; `SpeakerThreshold`
  tunes the estimate. Works on Whisper and Parakeet models; the streaming call rejects it (it needs the whole
  recording).

- **A streamed chat turn reports the server's token counts.** `GenerateChatStreamAsync`'s last chunk carries
  `ChatStreamChunk.Usage` (`PromptTokens`, `CompletionTokens`, `TotalTokens`) on the llama-server path — the same
  counts `GenerateChatWithToolsAsync` reports, hidden reasoning included, so a streaming consumer no longer has to
  re-tokenize the visible text. Null on the ONNX path, and when the output-token safety limit cuts the stream.

### Changed

- **Breaking (selection): automatic selection no longer lets a model take more than half of system RAM.** When nothing
  fits VRAM, `"default"`/`"auto"`/`"gguf:auto"` pick the largest model within min(system RAM − 4 GB, system RAM ÷ 2)
  (`GgufModelRegistry.SystemRamBudgetBytes`). Before, a 32 GB host with a small GPU got `gguf:qwen3-quality` — a
  ~20 GB download that then held most of the machine's memory; it now gets `gguf:qwen3-balanced`. A 64 GB host still
  gets `qwen3-quality`. Hosts where a model fits VRAM are unaffected. To keep the larger model, name it.
- `GenerateChatStreamAsync` (llama-server path) emits `FinishReason` on one final chunk, after any text or tool calls
  released when the stream ends. Before, it could arrive before those trailing chunks.

## [0.78.0] - 2026-09-26

### Added

- `ModelPreferences.ForProvider(provider)`: quantization preferences from `HardwareProfile.For(provider)` — an explicit
  `Cpu` ranks by the CPU tier and does not probe the GPU.

### Fixed

- **An explicit `Provider = Cpu` load by ONNX alias no longer loads the NVIDIA driver libraries.** The ONNX download
  step of the generator (preset aliases such as `"phi-4-mini"`), embedder, captioner, transcriber and translator read
  `HardwareProfile.Current` to rank quantizations, and that read probes the GPU (NVML, which brings `nvcuda.dll` with
  it). 0.76.0 removed the other probe sites; this was the last one on the ONNX paths.
- **A trimmed or AOT host — including a file-based `dotnet run app.cs`, which is AOT by default — can load and run a
  GGUF model.** Every JSON read and write in LMSupply (llama-server requests and responses, the server and runtime state
  files, download manifests, hub and NuGet listings, the VITS config) used reflection-based `System.Text.Json`, so such a
  host failed with "Reflection-based serialization has been disabled" before the first load. They now resolve through
  source-generated `JsonSerializerContext`s, and the core, llama, generator and synthesizer test suites run with
  reflection-based serialization off so a type left out fails in CI.

### Changed

- **With an explicit `Cpu`, the quantization ranking for a model without a registry subfolder follows the CPU tier
  (system memory), not the GPU's.** On a machine with a large GPU, a CPU load could previously prefer FP16 files
  meant for the GPU tier; it now prefers what the CPU tier ranks first. Pass `QuantizationHint` to choose explicitly.

## [0.77.0] - 2026-09-26

### Fixed

- **A streamed GGUF completion reports the server's token counts.** `GenerateChatCompleteResultAsync` and
  `GenerateCompleteResultAsync` estimated `TokenUsage` from the visible text (about four characters per token). A
  reasoning model's hidden reasoning never appears in that text, so a call that generated ~480 tokens and stopped at the
  limit reported ~60. The chat stream now asks for `stream_options.include_usage` and reads the server's `usage`
  (llama.cpp's `timings` when no usage object is sent); the text completion reads `tokens_evaluated`/`tokens_predicted`.
  `TokenUsage.IsEstimated` is true only when the server reported nothing and the counts are still an estimate.
- **A GGUF model whose llama-server exited recovers on the next call.** A server that was killed, crashed or ran out
  of GPU memory left the loaded model failing every later call with "connection refused" and nothing saying the server
  was gone. The next call now starts a new server with the configuration the model was loaded with, and logs a warning
  with the exit code. The call that was in flight when the server died still fails.
- **No false "initialized CPU-only" warning with llama.cpp b11146 and later.** Those builds no longer print the device
  list at the default verbosity, and its absence was read as a GPU runtime that failed to load, on every load, while the
  GPU was in use. When the startup log says nothing about devices, the binary's `--list-devices` answers instead, and
  the warning appears only when that lists no accelerated device.

## [0.76.0] - 2026-09-25

### Changed

- **Breaking: `"default"`, `"auto"` and `"gguf:auto"` select with one rule, from the profile of the load's provider.**
  `"default"`/`"auto"` read only the detected GPU and fell back to the smallest model when nothing fit VRAM, while
  `"gguf:auto"` also considered system RAM — the documented rule for all three. Now all three pick the largest model
  that fits VRAM, else the largest that fits system RAM (less 4 GB), else the smallest. With an explicit
  `Provider = ExecutionProvider.Cpu` the selection uses system RAM alone: the VRAM budget (including
  `LMSUPPLY_VRAM_BUDGET_MB`) does not apply and the GPU is not probed.
  - A host where no candidate fits VRAM (no GPU, integrated GPU, a small laptop GPU) now gets a larger model from
    `"default"`/`"auto"` than before, run mostly on the CPU. To keep the previous small model, name it:
    `gguf:qwen3-fast`.
  - `Cpu` + `"default"`/`"auto"`/`"gguf:auto"` on a GPU host now gets the model its RAM holds, not the one its GPU
    holds.

### Fixed

- **`GetModelInfo().ModelId` names the quantization that was loaded.** When a registry alias's default did not fit the
  memory budget and a smaller cached quantization stood in, the model info still named the default
  (`gguf:gemma4-balanced` loaded as Q4_0 reported `Gemma 4 E4B Instruct (Q8_0)`). It now names the loaded one
  (`Gemma 4 E4B Instruct (Q4_0)`).
- **`GeneratorModelInfo.KnownIssues` is populated for registry aliases.** It was looked up by the display name, which
  is not a registry key, so every GGUF load reported no known issues.
- **An explicit `ExecutionProvider.Cpu` no longer loads the NVIDIA driver libraries.** Every entry point's model pool,
  the runtime manager's initialization and several load-path reads probed the GPU whatever the provider, so a CPU
  load brought `nvml.dll` and the CUDA driver (`nvcuda.dll`) into the process. The GPU is now probed only when a GPU or
  `Auto` choice needs it: a CPU embedder load leaves both unloaded (module list), and an `Auto` load still probes.
- **A llama-server supplied through `ServerBinaryPath` loads on llama.cpp b11146 (v0.5.0) and later.** That build
  removed `--mmap`/`--no-mmap`/`--mlock`, and an external binary was treated as "unknown build" and sent `--mmap`
  (every preset asks for memory mapping), so the server exited with "invalid argument: --mmap" before `/health`. An
  external binary is now asked for its build with `--version` (unless `PinnedVersion` names it), so it gets the
  spelling it parses; when the build still cannot be read, memory mapping on — llama.cpp's default — is not sent at
  all. Binaries this library downloads were not affected: their build is known.

### Added

- **`HardwareProfile.For(ExecutionProvider)`** — the profile a load with that provider works with. For `Cpu` it is
  system memory and a CPU tier, built without probing the GPU; for any other provider it is `HardwareProfile.Current`.
  `HardwareDetector.GetRecommendation(ExecutionProvider)` answers from it.
- **`GeneratorModelInfo.RequestedModelId`, `RequestedFile`, `LoadedFile` and `IsQuantizationSubstituted`** — what the
  load was asked for, the alias's default file, the file it opened, and whether a smaller quantization stood in for the
  default. Set by GGUF (llama-server) loads. `IModelInfoBase.AliasName` now returns the requested id.

## [0.75.0] - 2026-09-24

### Fixed

- **`DownloadProgress.OverallPercentComplete` is byte-weighted.** It weighted every file of a multi-file download
  equally, so while the one large file of a model downloaded it reported the share of files done (18 % with the
  weights 82 % done). When every file's size is known up front (the repository listing or a verified manifest) it is
  now bytes done over bytes total; only when a size is unknown does it fall back to the file-count approximation.

### Added

- **`DownloadProgress.OverallBytesDownloaded` / `OverallTotalBytes`** — the bytes done and the size of the whole
  multi-file download (files already cached count as done), or `null` when a file's size is not known. The existing
  `BytesDownloaded` / `TotalBytes` are, as before, the current file's; their XML doc now says so.

## [0.74.0] - 2026-09-24

### Added

- **`LocalGenerator.IsModelDownloaded(modelId, options)` tells a consent gate whether a load would download
  anything.** It reads the cache only (no network, no runtime) and follows `LoadAsync`'s resolution: the
  file a `gguf:` alias opens on this host rather than any file of its repository, every shard of a split model,
  the model `default`/`auto` selects, a raw GGUF repository, a local path, and an ONNX model once a completed
  download of it is cached. The embedder and reranker already had this; a generator consumer had to work it out
  from the repository cache listing, which says "downloaded" for an alias whose file is not.
- **`LlamaServerPool.ReleaseIdleAsync()` stops every llama-server no model is using, now.** A disposed model's
  server stays pooled for `IdleTimeout` (10 minutes) so it can be reused, and on one GPU it keeps its memory. A host
  that switches models calls this after disposing the old one; it returns how many servers were stopped.

### Changed

- **A registry alias no longer loads another alias's cached file when its own would fit.** When a repository held
  only a smaller quantization (for example `gguf:gemma4-default`'s Q4_0), `gguf:gemma4-balanced` (Q8_0) loaded
  that file on a host where Q8_0 fits, silently. It now downloads its own file, and with `DisableAutoDownload` the
  load fails with `ModelNotFoundException` instead. On a host where the default does not fit, the cached smaller
  quantization is still used, as before.

- **Starting a llama-server on a GPU stops the idle servers of other models first** — for a generator, an embedder
  and a reranker alike. After switching from one GGUF model to another, the old model's server stayed resident
  beside the new one for ten minutes, and a generator was sized as if that memory were free (it now releases them
  before it measures). Idle servers of the same model are kept (the load may reuse them), and a server a live model
  is using is never stopped. The rule does not check whether the new server would fit beside the idle ones: a host
  that disposes its models between calls and alternates between two of them now starts a server on each switch
  (keep the models alive to keep their servers warm).

### Fixed

- **A pooled llama-server can no longer be stopped while a load is leasing it.** The idle sweep picked a server and
  stopped it in two steps; a load that leased it in between got a server that was shutting down. Retiring and
  leasing are now one atomic decision.
- **An importance-matrix file (`*-imatrix.gguf`) is never taken for a model.** Quantizers publish it next to their
  quantizations. It was counted as one, and as the smallest file it could be picked when no quantization fits.

- **A chat that starts with two system messages no longer fails on Qwen 3.x (llama-server backend).** A system prompt
  followed by a second system message (a conversation summary, for example) is valid in the OpenAI chat format, but
  Qwen 3.x chat templates reject any system message after the first, and the call failed with HTTP 500 ("System
  message must be at the beginning"). A leading run of system messages is now sent as one, joined by a blank line.
  The same applies when the model prepends its own tool prompt to a caller's system prompt. The token-budget trim
  counts the merged form. System messages later in the conversation are sent as before.
- **The Mistral prompt format keeps every system message.** A second system message was dropped, and when it was the
  first message left after the first one was taken out, the system prompt was dropped too. Every system message now
  rides in the next `[INST]` block, several in a row joined by a blank line.

## [0.73.0] - 2026-09-24

### Added

- **A raw prompt completion can say why it ended.** `ITextGenerator.GenerateCompleteResultAsync(prompt, options)`
  returns the same text as `GenerateCompleteAsync` in a `GenerationResult` whose new `FinishReason` is `"length"` when
  the model stopped at `MaxTokens` and `"stop"` when it finished — on both backends. Chat callers already had this
  through `GenerateChatWithToolsAsync` and `GenerateChatStreamAsync`; the prompt form had no way to tell a cut-off
  answer from a finished one. `GenerateWithUsageAsync` now returns the reason too.
- **A chat completion can say why it ended, without collecting a stream.** `ITextGenerator.GenerateChatCompleteResultAsync(messages,
  options)` is the chat twin: the same text as `GenerateChatCompleteAsync`, plus `FinishReason`. The string chat APIs cannot
  carry a reason, so a caller that needed one had to collect `GenerateChatStreamAsync` itself. On the llama-server
  backend the chat text path now reads the server's structured stream, which is where its `finish_reason` arrives
  (the text it returns is unchanged).
- `LlamaServerClient.GenerateStreamAsync` streams a raw completion as `CompletionStreamData` (text delta, and the
  finish reason on the last chunk, mapped from llama-server's `stop_type`).

### Changed

- **Breaking** for code that implements `ITextGenerator` / `IGeneratorModel` itself (a wrapper or proxy): add
  `GenerateCompleteResultAsync` and `GenerateChatCompleteResultAsync`. A wrapper delegates each to the model it wraps in one line. The members have no default
  implementation on purpose — a default could only report "no reason", and a wrapper that forgot to forward it would
  silently hide the reason its inner model knows.

### Fixed

- The raw completion stream no longer drops text the server sends on its final chunk.

### Documentation

- **`LMSupply.Text.Core`'s package README showed tokenizer calls that do not compile.** `TokenizerFactory` has no
  `CreateSentencePieceAsync` (the sequence tokenizer is `CreateSentencePieceSequenceAsync`), and the factories' length
  parameter is `maxSequenceLength`, so the README's `maxLength:` named arguments failed. Corrected; a test now checks
  every name the README, `docs/` and each package README use.

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
- **Reranker and translator model info report their input limit as `IModelInfoBase.ContextLength`.** Both declared it
  (`ModelInfo.MaxSequenceLength`, `TranslatorModelInfo.MaxLength`) but left the interface member at its default `null`,
  so a generic model listing (the console host's `/models` endpoint included) showed the limit as unknown. The embedder
  already reported it this way.

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
