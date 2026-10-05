# Changelog

All notable changes to this project are documented in this file. Versions follow
[Semantic Versioning](https://semver.org/); while the major version is 0, a minor release may contain
breaking changes, and each one is marked **Breaking** with a migration note.

## [0.106.1] - Unreleased

### Fixed
- **Cancelling a call now cancels it.** 19 method(s) that take a `CancellationToken` caught every exception to
  return a fallback (`null`, an empty result, a failure value) or to log and continue, and treated the caller's own
  cancellation the same way. They now let the caller's `OperationCanceledException` through; other failures behave
  as before. Affected: model and runtime downloads, update checks, GGUF metadata reading, the llama-server client/process, OCR and image generation.

## [0.106.0] - 2026-10-05

### Changed
- **Loading a tokenizer, a vocabulary, a download manifest or a generator from a path can be cancelled.**
  `TokenizerFactory.Create*Async` (7), `VocabularyLoader.LoadFrom*Async` (4), `DownloadManifest.ReadAsync` /
  `WriteAsync`, `LocalGenerator.LoadFromPathAsync` and `LlamaServerPool.ReleaseIdleAsync` take an optional
  `CancellationToken` and pass it to the file reads and loads under them. Source that calls them still compiles; a
  test that calls them inside an xUnit v3 test now gets xUnit1051 (pass `TestContext.Current.CancellationToken`),
  and a binary compiled against 0.105.x must be rebuilt.

## [0.105.2] - 2026-10-05

### Fixed
- **Exception messages are plain ASCII.** Five carried an em dash or an arrow (the llama-server «VRAM is insufficient»
  load failure, the ONNX generator's variant-search failure, the WordPiece vocabulary shape error, the transducer
  vocabulary check, the Whisper translate check); they now use `-` and `->`. A convention test over every LMSupply
  library assembly keeps exception messages ASCII.

## [0.105.1] - 2026-10-05

### Dependencies
- Microsoft.Extensions.AI.Abstractions 10.10.1; .NET 10.0.12 servicing for System.Numerics.Tensors, ASP.NET Core, EF Core and FileProviders pins; Microsoft.OpenApi 2.12.2 (required by Microsoft.AspNetCore.OpenApi 10.0.12).

## [0.105.0] - 2026-10-04

### Changed
- **Breaking:** `License` on `TranscriberModelInfo`, `DetectorModelInfo`, `SegmenterModelInfo`, `SynthesizerModelInfo` and
  `TranslatorModelInfo` is `string?` with no default — as on the Embedder, Reranker, Captioner and OCR entries since
  0.103.0. An entry nobody curated (a user-registered model, a repository or a local file the catalog does not know)
  reported a licence nobody declared: `"MIT"`, `"Apache-2.0"` or `"Unknown"`, from the type's default or the registry's
  fallback entry. It now reports `null`. Every built-in entry keeps its curated licence (a test per domain), except the
  `chinese` voice, whose model card states none — it is `null` instead of `"Unknown"`. A caller that printed `License`
  unconditionally should handle `null`.

## [0.104.0] - 2026-10-04

### Changed
- **The captioner's `default` and `auto` load Florence-2 base** (`onnx-community/Florence-2-base-ft`, MIT), the model
  `quality` already loaded. On 12 everyday photos ViT-GPT2 named 4 subjects wrongly and Florence-2 none, and its int8
  download (about 275 MB, chosen on most machines) is smaller than ViT-GPT2's (about 960 MB). **Breaking** for a caller
  that relied on `default` being ViT-GPT2 (for its exact captions, or because `Detail`/`NumBeams` behave per model): load
  `fast`, which is ViT-GPT2. `DefaultModels.Default` is now `DefaultModels.Florence2Default`; `DefaultModels.VitGpt2` is
  registered under `fast`, and `DefaultModels.VitGpt2Fast` is removed (use `VitGpt2`).

### Added
- **`LocalReranker.Describe(modelId)` and `LocalEmbedder.Describe(modelIdOrPath)`: what an id loads on this host — name,
  licence, repository and backend — without downloading.** They follow the same resolution as the load: a user alias,
  `auto` for this host, and a built-in GGUF alias such as `multilingual-fast`, which had no public name or licence (no
  registry entry describes a GGUF build). A GGUF build reports the licence of the model it was quantized from
  (`multilingual-fast` → Apache-2.0, bge-reranker-v2-m3). A model the catalog does not know reports its own name and
  a null licence. New shared types `ModelDescription` and `ModelBackend` (`LMSupply.Core`).

### Fixed
- A ViT-GPT2 model loaded from a local directory is identified as ViT-GPT2 rather than as whatever `default` names.

## [0.103.0] - 2026-10-04

### Changed
- **Every llama-server LMSupply launches now requires a per-process key.** The server listens on loopback, which keeps
  other machines out but not other processes or other users of the same machine; until now any of them could send
  prompts to the model or read the ones it held. Each server gets a random 256-bit key, handed to the process through
  its environment (not its command line), and every request LMSupply sends carries it. Nothing changes for callers of
  the generator, embedder or reranker. **Breaking** for a host that starts `LlamaServerProcess` itself and calls
  `Info.BaseUrl` with its own client: send `Authorization: Bearer <LlamaServerProcess.ApiKey>` (for an
  OpenAI-compatible client, set its API key to `ApiKey`), or set `LlamaServerConfig.RequireApiKey = false`.
  `--api-key`/`--api-key-file` in `AdditionalArgs` is refused while `RequireApiKey` is on.

### Fixed
- **An embedding model loaded by path from a copied repository is read like the same model loaded by id.** With
  `model.onnx` in `onnx/`, the repository's root files (`modules.json`, `1_Pooling/`, `config_sentence_transformers.json`,
  a root tokenizer) one level up were never read. And a model the catalog knows lost the catalog's declarations — a
  copied multilingual E5 model embedded queries and passages without its `query: `/`passage: ` prefixes, silently,
  since E5's repository does not declare them. A path load now reads the root files from the repository root, and when
  the files name their repository (the download manifest LMSupply writes, or `_name_or_path` in `config.json`) and the
  catalog knows it, the catalog's prefixes, pooling and sequence length apply.
- `docs/embedder.md` «Local Models» listed `vocab.txt` as required (the multilingual models use SentencePiece);
  `docs/llama.md` suggested an `HF_HUB_OFFLINE` variable nothing reads.

### Added
- **`License` on the Embedder, Reranker, Captioner and OCR registry entries** (and `OcrModelInfo.License` for a
  detection + recognition pair), as Generator, Transcriber, Detector, Segmenter, Synthesizer and Translator already had —
  the licence of the weights, curated, so a consent screen can state it without reading the repository card. For a
  conversion it is the converted model's licence: the `default` captioner (an ONNX export of a ViT-GPT2 model whose card
  states none) reports `Apache-2.0`. Every built-in entry carries one; a test keeps it that way.
- **`LlamaServerConfig.RequireApiKey`** (default `true`), **`LlamaServerProcess.ApiKey`**, and an `apiKey` parameter on
  `LlamaServerClient`.

## [0.102.0] - 2026-10-03

### Added
- **`LocalReranker.GetDownloadSizeBytesAsync(modelId, options)` and `LocalOcr.GetDownloadSizeBytesAsync(language,
  options)`** — what a first load would fetch, read from the repository listing without downloading, as Captioner,
  Embedder, Generator and Transcriber already answer. Reranker: the GGUF file at the load's quantization, or the graph
  and tokenizer files; OCR: the detection model plus that language's recognizer and dictionary (about 10 MB for
  English, about 87 MB for Japanese or Chinese). Offline (`DisableAutoDownload`) the listing comes from the cache only
  and an unlisted repository throws, as the load does.

### Changed
- The embedder's and the reranker's GGUF downloaders were two copies of one class; they are now one shared
  implementation, so selection, cache layout and the download plan cannot drift between them.

## [0.101.0] - 2026-10-03

### Added
- **`Clone()` on every options type**: new on `CaptionerOptions`, `OcrOptions` and `ImageGeneratorOptions`, and
  `EmbedderOptions.Clone()` is now public. The others already had one.

### Changed
- **Captioner: `NumBeams` is a beam search, `Temperature` a sampling temperature.** `NumBeams` above 1 used to sample
  at `Temperature` instead of searching, and at the default `NumBeams = 1` `Temperature` was never read. Now
  `NumBeams > 1` runs a beam search (ViT-GPT2 and Florence-2) and fills `CaptionResult.AlternativeCaptions` with the
  runners-up — a field that was always empty. **Breaking:** `CaptionerOptions.Temperature` is `float?`, null by
  default (greedy); setting it samples. A non-positive temperature, `NumBeams < 1`, or a temperature together with
  beams is refused at load. Migration: delete an assignment of `Temperature = 1.0f` (that was the old default).
- `CaptionResult.Confidence` is documented as what it is: exp of the mean token log-probability, between 0 and 1.

### Fixed
- **No `Local*` entry point changes the options it is given any more.** Most loaders wrote what they resolved into
  the caller's instance: a `:variant` qualifier into `QuantizationHint` (Captioner, Detector, Embedder, OCR, Reranker,
  Segmenter, Synthesizer, Transcriber, Translator), the resolved model id into `ModelId`, the language into
  `OcrOptions.LanguageHint`. An options object reused for a second load (`"quality:fp16"`, then `"default"`) carried
  the first load's variant into the second. Every entry point now works on a copy. A source-scan test fails on the
  mutating shape.

## [0.100.0] - 2026-10-03

### Added
- **Captioner: a second, current captioning model — alias `quality` (Florence-2 base, MIT).** On everyday photos it names
  the main subject where ViT-GPT2 tends to substitute a familiar scene (a lasagna as "bread with a fork and knife", a
  concert ticket as a sign reading "don't miss the holidays"). Same `ICaptionerModel` surface, same
  `DisableAutoDownload` behaviour. Its files are published in quantized variants; a load takes the one
  `QuantizationHint` (or a `"quality:fp16"` qualifier) names, else the one the hardware tier picks — int8, about 275 MB,
  on most machines. `default`, `fast` and `auto` stay ViT-GPT2.
- **`CaptionerOptions.Detail`** (`Brief` · `Detailed` · `Paragraph`) selects how much a caption says, on models that
  caption at several levels (`quality`). A paragraph needs a larger `MaxLength` than the default 50.
- **`LocalCaptioner.GetDownloadSizeBytesAsync`** answers what a load will download for the same id and options (the
  variant it would pick), without downloading — for a consent screen, as Embedder, Generator and Transcriber already
  offer.
- A local model directory holding the Florence-2 graphs (any one variant) loads as Florence-2.

### Changed
- **Options a captioning model cannot honour are refused at load, before anything is downloaded**
  (`NotSupportedException`): `Detail` other than `Brief` on ViT-GPT2, and `Prompt` on Florence-2, whose prompt selects
  a task rather than starting the caption.
- The Captioner package README and guide listed aliases that do not exist (`vit-gpt2`, `smolvlm`, `florence2`), a
  ~500 MB size for ViT-GPT2 (it downloads about 1 GB) and DirectML, which was removed in 0.67.0; corrected.

## [0.99.0] - 2026-10-03

### Removed
- **Breaking: `GenerationOptions.IncludePromptInOutput` is removed.** No generator ever acted on it — setting it left
  the output unchanged. Migration: delete the assignment; to show the prompt with the answer, prepend the prompt you
  already hold.

## [0.98.2] - 2026-10-02

### Fixed
- **A refused llama-server release lookup now says why.** When GitHub refuses the API request (HTTP 403/429 with an
  exhausted rate limit) the acquisition error names the limit, the time it resets and how to raise it, instead of
  "the latest-release lookup returned nothing"; any other refusal names its status and URL.
- **`GITHUB_TOKEN` (or `GH_TOKEN`) is used for release lookups when set**, which raises GitHub's limit from 60
  unauthenticated requests an hour per IP address. It goes to the GitHub API only — never with the build download —
  and an `HttpClient` passed in with its own `Authorization` header is left as it is.
- Cancelling a release lookup now cancels it; it used to be reported as "no release resolved".

## [0.98.1] - 2026-10-02

### Fixed
- **Packages now carry the license text.** Each `.nupkg` includes `LICENSE` next to the `MIT` expression,
  so an application that ships third-party notices can copy the copyright line from the package.

## [0.98.0] - 2026-10-01

### Changed
- **Breaking — downloads use the Hugging Face hub cache layout, so other Hugging Face tools find them.** The revision is
  resolved to a commit; each file is stored in `blobs/{id}` (SHA-256 for LFS files, else the Git blob id) and linked
  from `snapshots/{commit}/`, with `refs/{revision}` naming the commit. Without link support the new blob is moved into
  the snapshot (one copy); a blob already in the cache is reused without a request. When the commit cannot be resolved
  (offline, a failed request), or the model already has a `snapshots/{revision}/` directory from an earlier version,
  files are written there as before. GGUF models (Generator, Embedder, Reranker) are written the same way. Migration:
  code that read `snapshots/main/` or the `gguf-embeddings/` / `gguf-rerankers/` trees directly should use
  `CacheManager.GetSnapshotDirectories` / `GetModelFilePath`. Files already in those places are still read and are not
  moved.
- **Download manifests are kept in `models--{org}--{name}/.lmsupply/manifests/`**, out of the snapshots.
  `DownloadManifest.Read`/`ReadAsync` look there first and still read a `.lmsupply-manifest.json` inside the directory.
- **Breaking: `CacheManager.DeleteModel` deletes only what LMSupply owns** — its snapshots (in a commit snapshot shared
  with another tool, only the files its manifests list), blobs no remaining snapshot links to, refs naming a removed
  snapshot, and `.lmsupply/`. The repository directory is removed only when empty, and the method returns `false` when
  nothing there was LMSupply's. Before, it deleted the whole repository directory, other tools' files included.
- **`FindReclaimable`/`Reclaim` never touch a link or another tool's snapshot; `GetTotalCacheSize` counts a linked blob once.**

### Added
- **`RepoFile.Lfs` (`RepoFileLfs`) and `RepoFile.BlobId`** — the Git LFS entry of a listed file and its hub-cache blob name.

## [0.97.0] - 2026-10-01

### Fixed
- **`ThreadCount` is honoured by every ONNX model type.** The Embedder fixed its intra-op threads at the logical core
  count and ignored the option; OCR and the Captioner ignored it too. Measured with `multilingual-e5-small` on CPU
  (a 300-character sentence, 20 calls): `ThreadCount = 1` went from about 790 ms of process CPU per call to 54 ms.
- **With `ThreadCount` set, ONNX Runtime threads no longer spin-wait between runs.** A short call used to leave several
  cores at full load for seconds afterwards (about 15 s of CPU in 3 idle seconds); now idle stays idle. Without
  `ThreadCount` the ONNX Runtime defaults are kept, spinning included, for throughput.
- **Parakeet (Transcriber) and the Reranker apply `LogLevel`.**

### Changed
- **The Reranker without `ThreadCount` uses ONNX Runtime's thread defaults** instead of one intra-op thread per
  logical core plus half as many inter-op threads.

### Added
- **`SessionOptionsExtensions.ApplyCommonOptions`** (`LMSupply.Inference`): the one place the session's log level and
  threads come from `LMSupplyOptionsBase`, for code that creates its own ONNX sessions through `OnnxSessionFactory`.

## [0.96.0] - 2026-09-30

### Added
- **The native ONNX Runtime can be kept off the network: `RuntimeManager.Configure(new RuntimeManagerOptions { ... })`.**
  `RuntimeDirectory` loads the runtime from a directory the application ships (nothing is looked up, downloaded or
  updated; a provider whose libraries are missing there is refused, so a CPU-only bundle serves CPU).
  `DisableAutoDownload` takes the runtime from the cache only and throws on a miss, with no version lookup or update
  check. `PinnedVersion` fixes the version, which is then never resolved from nuget.org and never auto-updated. Until
  now a model's `DisableAutoDownload` stopped model downloads while the runtime was still fetched from nuget.org on
  first use and version-checked every 24 hours. Call `Configure` once, before the first model load.
- **`LocalEmbedder.GetDownloadSizeBytesAsync(modelIdOrPath, options)`**: the bytes a first load would fetch (the files
  that load picks on this host, at the repository's listed lengths; 0 for a local path), the same query the Generator
  and Transcriber already had, for a consent screen shown before any download. ONNX catalog aliases, repository ids and
  GGUF repositories are all answered; with `DisableAutoDownload` it reads the cache only.
- **A loaded embedding model is a Microsoft.Extensions.AI `IEmbeddingGenerator<string, Embedding<float>>`:**
  `model.AsEmbeddingGenerator(EmbeddingTextKind.Unspecified | Query | Passage)`. The text kind applies the model's
  query/passage prefix (E5), `EmbeddingGenerationOptions.Dimensions` is a Matryoshka truncation, a `ModelId` naming
  another model throws, and the generator does not own the model. `LMSupply.Embedder` now references
  `Microsoft.Extensions.AI.Abstractions` 10.9.0.

## [0.95.0] - 2026-09-30

### Added
- **The detector finds people's poses: `pose`, `pose-quality` and `pose-multi` return each person with the 17 COCO
  keypoints.** Backed by Google's MoveNet (Apache-2.0 weights, community ONNX conversion): `pose` is SinglePose
  Lightning, `pose-quality` SinglePose Thunder, `pose-multi` MultiPose Lightning (up to six people, each with a box).
  Keypoints are in COCO order, so `PoseSkeleton` indices address them, in original-image pixels, each with its own
  score. SinglePose reports one person whose confidence is the mean keypoint score and whose box is the keypoints'
  extent. About 5 / 10 / 24 ms per 800x533 frame on a desktop CPU.
- **`DetectorOutputLayout.MoveNetSinglePose` and `MoveNetMultiPose`, and `DetectorInputFormat.PaddedRgbInt32`**, so a
  MoveNet ONNX file the registry does not describe can be loaded by stating its layout: NHWC int32 RGB, resized with
  the aspect ratio kept and padded to the input size, with coordinates mapped back through the padding.

### Fixed
- **A model file named by a path inside the repository (`onnx/model.onnx`) is the file that loads.** The name was
  treated as the "any precision" placeholder, so a machine that preferred int8 downloaded the int8 build beside it
  instead; the path was also resolved against the ONNX folder a second time. Only the bare name `model.onnx` still
  leaves the precision to the machine.
- **Detector output tensors are released after each frame** instead of waiting for the finalizer.

## [0.94.0] - 2026-09-30

### Added
- **`CacheManager.GetSnapshotDirectories`, `FindSnapshotDirectory` and `TryGetContentLength`** — the snapshot directories a lookup reads, in order, and a content length that follows symbolic links.

### Fixed
- **Models downloaded into the shared cache by other Hugging Face tools are found and not downloaded again.** Lookups
  (`IsModelDownloaded*`, the loaders, `CacheManager.ModelFileExists`/`GetModelFilePath`/`GetMissingFiles`) now read the
  snapshot `refs/<revision>` names first and `snapshots/<revision>/` after it; that snapshot is never modified, and
  downloads are still written to `snapshots/<revision>/`. The ONNX generator's `IsModelDownloaded` still answers from the
  download manifest only.

### Changed
- **Documentation comments describe behaviour only.** XML documentation and code comments state what the code does
  and the condition that triggers it; references to external tracking and planning notes were removed.

## [0.93.1] - 2026-09-29

### Documentation
- **The package READMEs shown on nuget.org compile.** Every C# block in the repository README and in each package
  README is now compiled against the current API by a test. Fixed on the way:
  - Reranker: `result.Index` → `result.OriginalIndex`.
  - Vision.Core: `PreprocessProfiles` and a synchronous path overload that do not exist → `await PreprocessAsync(path, PreprocessProfile.ImageNet)`.
  - Text.Core: `EncodeSequence`/`EncodeBatch` need `CreateAutoSequenceAsync` (not `CreateAutoAsync`), and `Decode` takes `int` ids. The encoded types are documented as the `readonly struct`s they are (`Length`, not `ActualLength`).
  - Generator: the llama-server pinning example names its namespace, and the ONNX install line is a shell command.
  - Missing `using` lines (`LMSupply` for `ExecutionProvider`, `LMSupply.Generator.Models`), duplicated variable names, and the Synthesizer examples' `LMSUPPLY001` opt-in.

## [0.93.0] - 2026-09-29

### Added
- **`LocalTranscriber.IsModelDownloadedAsync` tells whether a load would open cached files only.** It checks the files the
  load picks (the same alias, `:variant` qualifier and quantization as `GetDownloadSizeBytesAsync`) at the lengths the
  repository lists, makes no request and loads nothing — the check the embedder, reranker and generator already had. A
  repository directory left behind without its model files answers `false`.
- **`LocalGenerator.GetDownloadSizeBytesAsync` is what the first load downloads on this host.** `"auto"`/`"default"` resolve
  to the model the selection picks (goal, `MaxContextLength`, `GpuLayerCount = 0` applied) and a GGUF alias to the
  quantization the memory budget picks; a split model counts every shard. It reads the repository listing and downloads
  nothing. The llama-server runtime is not counted.

### Fixed
- **`LocalGenerator.LoadAsync` and `DownloadModelAsync` no longer write a `:variant` qualifier into the caller's options.**
  They set `QuantizationHint` on the instance passed in, so reusing it for another id carried the first id's
  quantization; they now work on a copy.

## [0.92.1] - 2026-09-29

### Fixed
- **`LlamaOptions.GpuLayerCount = 0` sizes the auto model and quantization against system memory.** A load that keeps
  every layer on the CPU was still chosen as if the GPU held it: on a GPU with room for Qwen3 8B, `"auto"` picked 8B and
  ran it entirely on the CPU. It now selects like `Provider = Cpu`, and `AutoSelectionGoal` applies.

## [0.92.0] - 2026-09-29

### Added
- **`GeneratorOptions.AutoSelectionGoal` chooses what `"default"`, `"auto"` and `"gguf:auto"` take when no model fits VRAM.**
  `Quality` (default, unchanged) takes the largest model the system RAM budget holds; `Responsive` takes the smallest in the
  pool, for interactive use on a host whose GPU cannot hold the model. A candidate that fits VRAM is chosen the same way
  under either goal. `ModelSelectionResult.Goal` and the `[LocalGenerator.auto]` log line report it.

### Fixed
- **`"default"` and `"auto"` are sized for the requested context, like `"gguf:auto"`.** 0.91.0 sized `gguf:auto` and the
  quantization for `MaxContextLength`, but the other two names still sized every candidate for 4,096 tokens, so the same
  load could pick a model its context did not fit. `IsModelDownloaded` and `DownloadModelAsync` for these names follow.

### Removed
- **Breaking**: `GeneratorOptions.Verbose` and `TextGeneratorBuilder.WithVerboseLogging()`. Nothing read them — setting them
  never changed any output. The load already writes its selection, candidates and server lifecycle to `Trace`; a
  `TraceListener` (or its filter level) decides how much of it you see. Remove the assignment.

### Changed
- `TextGeneratorBuilder` hands the load a copy of all its options (`GeneratorOptions.Clone()`), not a hand-written subset.
  `TextGeneratorBuilder.WithAutoSelectionGoal` sets the new goal.

## [0.91.0] - 2026-09-29

### Changed
- **The quantization and model chosen before download size the KV cache like the server.** Both used a file-size estimate that
  assumed every attention head keeps K and V, several times the real cache of grouped-query and hybrid models. On an 8 GB GPU
  Qwen2.5-7B was downscaled from its default Q4_K_M to IQ4_XS; it now keeps Q4_K_M. A cache that holds only the smaller
  quantization now reports the model as not downloaded, and the next load fetches the default.
- **`gguf:auto` and the quantization choice are sized for the requested context** (`MaxContextLength`; unset = 4,096). On an
  8 GB GPU `gguf:auto` picks Qwen3 8B at 4,096 tokens (it picked Qwen 3.5 4B before) and Qwen 3.5 4B at 16,384.
- **Breaking**: `GgufModelInfo.NumLayers` and `HiddenSize` are replaced by `KvCacheBytesPerToken` and `SlidingWindowKvBytes`,
  read from each file's attention metadata, with `EstimateKvCacheBytes(contextLength)`. Several registered layer counts were
  wrong (Gemma 4 E4B: 34, the file has 42). `GgufModelRegistry.Resolve(alias, provider, contextLength)` and
  `GetAutoSelection(provider, contextLength)` take the context; `GgufModelDownloader.DownloadFromRegistryAsync` takes
  `contextLength`.

## [0.90.0] - 2026-09-29

### Added
- **`IGeneratorModel.CountTokensAsync(messages, options)` counts a chat request's prompt with its tools.** The tool
  definitions in `GenerationOptions.Tools` (with the tool choice and thinking setting) are rendered as generation sends
  them: on GGUF by the server's own chat template (`/apply-template`), matching the server's `prompt_tokens` exactly; on
  ONNX with the tool definitions generation injects. For 13 tools on Qwen 3.5 2B the prompt is 1,769 tokens, where the
  messages alone count 28. **Breaking** for implementers of `IGeneratorModel`: implement the new overload (a wrapper
  forwards it).
- **`GgufMetadata` carries the file's KV cache layout**: `KeyLength` / `ValueLength` (and their `…Swa` sliding-window
  counterparts), `SlidingWindow`, `SlidingWindowPattern`, `SharedKvLayers`, `FullAttentionInterval` and
  `HeadCountKvPerLayer` (when the file stores the KV head count per layer). For such a file `HeadCountKv` is now the
  largest per-layer count; it was null.

### Changed
- **Loading a model whose `llama-server` is still running shares it when it holds the requested context.** A request for
  8,192 tokens now shares a running 16,384-token server of the same model; before, only an equal context was shared and a
  second server was started. An idle server of the model with a smaller context is stopped before a new one is sized.

### Fixed
- **A GGUF model gets the context that fits its KV cache.** The context fit sized the cache from the file size, as if every
  attention head kept K and V, so grouped-query models were over-sized several times over: Qwen2.5-7B (IQ4_XS) on an 8 GB
  GPU was fitted to 6,163 tokens of a requested 16,384, which llama-server loads fully offloaded. The cache is now sized
  from the file's attention metadata — KV heads, K/V head dimensions, the cache type the server runs with, only the layers
  that keep a cache (hybrid recurrent and shared-KV layers do not), and a fixed window for sliding-window layers. For the
  three models measured, the estimate equals the size llama-server reports. The GPU layer fit uses the same size and the
  file's layer count.
- **Generation's context trim counts the tools.** Old turns were trimmed against a count of the messages alone, so a
  request with tools could pass the trim and then exceed the context on the server. The trim now counts the rendered
  prompt, tools included.
- **The VRAM budget sees the GPU as it is at load time.** Free VRAM was read once, when the process first detected the
  hardware, so a model loaded since was counted as free memory and `GeneratorModelInfo.VramFreeBytes` repeated the
  start-up figure. It is now read when the load sizes its server, after the idle servers it may displace are stopped
  (NVIDIA; other GPUs keep the start-up reading). Beside a 2B model in use, the budget for a 7B on an 8 GB GPU drops from
  6,959 to 6,112 MB.
- **GGUF metadata with 8- or 16-bit values is read correctly.** Such values were read as 4 bytes, which shifted every
  following key.

## [0.89.0] - 2026-09-28

### Added
- **`TranscribeOptions.MinSpeakers` / `MaxSpeakers` bound the estimated speaker count.** When you know how many people
  could speak (a meeting's attendees), set `MaxSpeakers`: a recording with fewer voices keeps its own count, where
  `NumSpeakers` would split one voice to reach the number.

### Changed
- **Diarization's default `SpeakerThreshold` is 0.4 (was 0.5).** On the four-speaker sherpa-onnx test recording, 0.5
  finds two speakers and 0.4 finds three. A meeting with two male voices that 0.5 merged is separated at 0.4. A
  recording of one voice stays one speaker at both. With the small-cluster rule below, a lower cut no longer turns a
  few stray windows into a speaker. Pass `SpeakerThreshold = 0.5f` for the previous cut.
- **Breaking — the Synthesizer is marked `[Experimental]` (diagnostic `LMSUPPLY001`).** It has no text-to-phoneme step yet,
  so its output is not intelligible speech (the known issue documented in 0.88.0). Every use of `LocalSynthesizer` and
  `ISynthesizerModel` now reports `LMSUPPLY001`, an error by default, so the status reaches the build and not only the
  README. To use the package anyway (loading, voice selection and the audio API work), suppress that one id:
  `<NoWarn>$(NoWarn);LMSUPPLY001</NoWarn>` or `#pragma warning disable LMSUPPLY001`.

### Fixed
- **`TranscribeOptions.NumSpeakers` gives the number of speakers asked for.** A cluster of a few stray embeddings
  used to take one of the requested slots and then disappear from the output, so asking for 3 speakers could give 2,
  and asking for 2 could give 1. Clustering now follows pyannote: such a small cluster joins the nearest real speaker,
  and a requested count counts real speakers only.
- **A recording of 10 seconds or less honours `NumSpeakers` and `SpeakerThreshold`.** It returned the segmentation
  window's local speakers without clustering, so neither option was read.
- **The console host's `/api/registry/models` names what each domain loads.** The transcriber, synthesizer, captioner, OCR,
  detector and segmenter lists were written by hand and had drifted to repositories the library never loads (for example
  Tesseract for OCR, YOLOv8 for the detector, SAM ViT for the segmenter). They now come from each domain's registry.

## [0.88.0] - 2026-09-28

### Fixed
- **Every Synthesizer voice was labelled MIT, which is the Piper engine's license, not the voices'.** Each entry's `License`
  now states its voice's license from the model card. `lessac` is the Blizzard 2013 research license. `fast` (Ryan),
  `british` (Semaine) and `korean` (KSS) are CC BY-NC-SA 4.0, which is non-commercial. `quality` (Amy) is unspecified,
  and `chinese` (Huayan) is unknown. Check a voice's license before shipping it.
- **The `korean` voice named a build that was never published** (`ko_KR-kss-x_low`), so every load failed. It now loads
  `ko_KR-kss-medium` (63 MB, 22,050 Hz). Loading is fixed, but Korean text still produces near-silence: see the known issue below.
- **MobileSAM prompts now follow SAM's input contract**, so masks land where the prompt is:
  - The image is resized by its longest side with the aspect ratio kept. It used to be stretched to 1024×1024, which
    misplaced masks on non-square images.
  - Both axes of the prompt coordinates use that one scale.
  - A first prompt sends `has_mask_input = 0`. It used to send 1, so the decoder conditioned on an empty mask.
  - A point prompt without a box gets SAM's padding point.
  - `multimask` returns the three multimask candidates instead of mixing in the single-mask token.
- A repository the Hub will not list (401) is reported as possibly nonexistent. Before, the message said only "private
  repository, set HF_TOKEN", but the Hub also answers 401 for a repository that does not exist.

### Added
- **`LocalSegmenter.LoadInteractiveAsync()`: interactive (point/box prompt) segmentation you can actually load.**
  `IInteractiveSegmenter` was public, but no public method returned one, and the `interactive` alias named a GitHub
  project (`ChaoningZhang/MobileSAM`) instead of a Hugging Face repository. The alias now loads the MobileSAM ONNX
  export `Acly/MobileSAM` (MIT). `LoadAsync("interactive")` throws and points here.

### Removed
- **Breaking: the Synthesizer alias `japanese` and `DefaultModels.JaJp`.** The voice it named (`ja_JP-jsut-medium`) was
  never published, so every load failed. Piper's only Japanese voice needs a Japanese phonemizer, which this package does
  not have (see the known issue).

### Known issue
- **Known issue — the output is not yet intelligible speech.** The synthesizer has no text-to-phoneme step: it maps letters to fixed ids instead of the phoneme ids the Piper voices were trained on, so what comes out is voice-like noise (a Whisper transcript of "The weather is beautiful today." read back "tube of warrior practitioner"), and text in non-Latin scripts comes out as near-silence. This has been the case since the Synthesizer was added. Loading, voices, WAV output and streaming work; the
  text-to-phoneme step is missing.

### Changed
- **Breaking: the default voice is LJSpeech** (`en_US-ljspeech-medium`, public domain). Before, it was Lessac, whose license
  does not allow commercial use. Lessac stays available as the alias `lessac`.

## [0.87.0] - 2026-09-28

### Removed
- **Breaking: the Translator alias `en-ko` and `DefaultModels.OpusMtEnKo`.** They pointed at `onnx-community/opus-mt-en-ko`,
  a repository that does not exist, so every load of the alias failed. No ONNX export of an OPUS-MT English-to-Korean model
  is published under `onnx-community` or `Xenova`. Migration: load a model you exported yourself by path or repository id.

### Fixed
- **Translator `SizeBytes` states the full-precision size.** Every alias said 300 MB, which matched no file set in the
  repositories. It now gives the full-precision encoder + merged decoder: 445.9 MB for `ko-en` and `zh-en`, 428.3 MB
  for `ja-en`. The doc says a load downloads the quantization the hint or hardware tier picks, often half of this or
  less. Memory estimates read this figure, so they are larger than before.
- The console host's model listing for the translator came from a hand-written list. That list named two repositories
  the library never loads (`facebook/nllb-200-distilled-600M` as `default`, and the missing `opus-mt-en-ko`). It now
  comes from the registry.

### Changed
- `GgufModelInfo.EstimatedSizeBytes` documents that it is the registry file's size. A load may download a smaller
  quantization when that file does not fit the memory budget.

## [0.86.0] - 2026-09-27

### Added
- **`LocalTranscriber.GetDownloadSizeBytesAsync`: how much a load will download, before it runs.** Given the options
  a load will use, it returns the bytes of the files that load fetches. The quantization follows `QuantizationHint`
  (or a `:int8` qualifier), otherwise the machine's hardware tier, as the load decides. It adds the diarization pair
  when `PreloadDiarization` is set. It reads the repository listing (cached, and reused by the load) and downloads
  nothing. `TranscriberModelInfo.SizeBytes` is the full-precision export's size and was never the download: `default`
  lists 290 MB while a load on a typical machine fetches the int8 pair, about 81 MB.
- **`HuggingFaceDownloader.PlanWithDiscoveryAsync` / `PlanModelAsync`: the files a download would fetch, with their
  listed sizes (`DownloadPlan`).** Each uses the same selection as its download method (`DownloadWithDiscoveryAsync` /
  `DownloadModelAsync`), so the plan and the download fetch the same files.

### Changed
- `TranscriberModelInfo.SizeBytes` documents what it is: the full-precision export's size used for model selection and
  memory estimates, not the download size. Its value is unchanged.

## [0.85.0] - 2026-09-27

### Added
- **A request that fails because llama-server exited now says so: `InferenceBackendExitedException`.** It carries the
  server's `ExitCode` and `RecentLog`, and wraps the transport failure the request saw. Before, the caller got only
  "connection refused" or "the response ended prematurely", which does not say the server is gone. This covers every
  generation, embedding and reranking call on a pooled llama-server, including a stream cut part-way through. It
  derives from `InferenceException`. A failure while the server is still running is rethrown as it was, after up to 2 s spent confirming the server has not exited. A generator's next call
  on the same model starts a new server, as before.

### Fixed
- **GGUF embedders and rerankers recover from a dead llama-server.** A pooled embedding or reranking server that
  exited (killed, crashed, out of memory) made every later call on that model fail on the dead port, until the model
  was reloaded. The next call now starts a new server with the same configuration, as the generator has done since
  0.77.0. The call that saw the server die still fails, now with `InferenceBackendExitedException`. All three model
  types share one replacement rule.

## [0.84.0] - 2026-09-27

### Added
- **`LlamaServerProcess.RecentLog`: what a server wrote last, readable after it dies.** The last 200 lines of the
  server's output (stderr and stdout), kept for the server's whole life. A server that dies in the middle of a request
  used to leave the caller only "connection refused"; read `RecentLog` with `ExitCode` to see why. The generator's
  restart warning for a dead server now includes the last 20 lines.

### Fixed
- **A long-running server no longer grows memory with its log.** The startup log buffer kept receiving every line the
  server wrote after it was ready, for its whole life, and nothing read it. It now stops at readiness
  (`Info.StartupLog` is unchanged); later output goes to the bounded `RecentLog`.
- **The server's stdout is read.** It was redirected and never read, so a build that writes to stdout would fill the
  pipe and stall the server.

## [0.83.0] - 2026-09-27

### Fixed
- **`LocalDetector`, `LocalSegmenter`, `LocalSynthesizer` and `LocalTranslator` report download progress.** Their
  `LoadAsync(…, IProgress<DownloadProgress>? progress, …)` accepted a progress sink and never passed it to the download,
  so a caller saw nothing while the model was fetched. This is the same defect 0.82.0 fixed in `LocalTranscriber`.
- **Listing a repository takes one request per 1000 files, not one per directory.** `ModelDiscoveryService` (used for
  file sizes before a download and for model discovery) walked the Hugging Face tree API one directory per request.
  On a repository with hundreds of directories this took about a minute before the first byte: the default
  `LocalSynthesizer` voice waited 56 s. It also used up the anonymous API quota (500 requests per 5 minutes per IP) in
  one load, so every later request from the same machine got HTTP 429. The listing is now one recursive request that
  follows the `Link` header's next page. A page that fails is an error. Before, a directory whose listing failed was
  silently left out of the result.

### Changed
- `ModelPathResolver.ResolveModelAsync` and `ResolveEncoderDecoderAsync` take an optional
  `IProgress<DownloadProgress>? progress` (before the `CancellationToken`). **Breaking** only for a caller that passes
  arguments positionally or was compiled against 0.82.0: recompile, and name the token if you pass it positionally.

## [0.82.0] - 2026-09-27

### Added
- **The speaker-diarization models can be fetched ahead and checked without a request.** Set
  `TranscriberOptions.PreloadDiarization = true` and `LocalTranscriber.LoadAsync` fetches and loads the pair that
  `TranscribeOptions.Diarize` uses (pyannote segmentation-3.0 + WeSpeaker ResNet34) together with the transcription
  model. It reports through the same `progress` and follows the same `DisableAutoDownload` rule. With downloads disabled
  and the pair not cached, the load fails with `ModelNotFoundException`, not a later diarized call.
  `LocalTranscriber.IsDiarizationDownloadedAsync(options)` says whether a diarized call would open both files from the
  cache. It makes no request and writes nothing. `LocalTranscriber.DiarizationDownloadSizeBytes` is the pair's size
  (32,523,463 bytes). Before, the pair could only arrive silently on the first diarized call, and nothing named it.

### Fixed
- **`LocalTranscriber.LoadAsync` reports download progress.** It accepted an `IProgress<DownloadProgress>` and never
  passed it to the download, so a caller received no progress for the transcription model (Whisper or Parakeet).
- A load that fails after the model object is created (warm-up, or the diarization preload) now disposes it.

## [0.81.1] - 2026-09-27

### Fixed
- **`ModelPool` can be disposed synchronously.** It implemented only `IAsyncDisposable`, so a host that disposed it with
  `using` or through a synchronously disposed container scope got an exception instead of the models being released.
  It now implements `IDisposable` too, blocking on `DisposeAsync`.

## [0.81.0] - 2026-09-27

### Added
- **The llama-server request timeout can be set per load.** `LlamaOptions.RequestTimeout` (`TimeSpan?`, null keeps the
  5-minute `LlamaServerConfig.DefaultRequestTimeout`, `Timeout.InfiniteTimeSpan` means no limit) reaches the server
  client. Before, a CPU-only tool-calling loop whose round took longer than 5 minutes failed at exactly 300 s with no
  setting to change it. The limit covers a whole non-streamed completion, or a streamed one until the server starts answering,
  and expiry still throws `TaskCanceledException` with an inner `TimeoutException`.

### Fixed
- **Loads that share a pooled server keep their own request timeout.** `ServerLease.Client` is now a view per lease
  over the pooled connections. Before, the server's client carried the limit of whichever caller started it, for every
  later caller of the same model. This includes the embedder and reranker, which lease the same way.
- **Passing `LlamaOptions` without `GpuLayerCount` keeps the VRAM fit.** An unset count already meant "all layers" at
  launch, but the fit that lowers it to what VRAM holds only ran for an explicit `-1`. Setting one unrelated property,
  such as a timeout or `Threads`, therefore offloaded every layer regardless of VRAM and relied on the out-of-memory
  retry. Unset now means `-1` everywhere.

## [0.80.0] - 2026-09-26

### Added
- **A generation result reports how fast the server generated it.** `GenerationResult.Timings`
  (`GenerateCompleteResultAsync`, `GenerateChatCompleteResultAsync`), `ChatCompletionResult.Timings` and the last chunk
  of `GenerateChatStreamAsync` (`ChatStreamChunk.Timings`) carry llama-server's own `timings` as `GenerationTimings`: `CompletionTokensPerSecond`,
  `PromptTokensPerSecond`, `CompletionDuration`, `PromptDuration`, `CachedPromptTokens` and `PromptTokensEvaluated`.
  These are the server's values, not derived from `Usage`. A rate computed from the visible text overstates a reasoning
  model's decode speed, and `Usage.PromptTokens` includes prompt-cache hits the server never evaluated. Null on the
  ONNX path, on server builds without `timings`, and when `MaxTokens` cuts the stream client-side. LMSupply.Llama
  exposes the raw object as `LlamaServerTimings` on `ChatStreamData.Timings`, `ChatCompletionFullResponse.Timings` and
  `CompletionStreamData.Timings`.

### Fixed
- **`GenerateChatWithUsageAsync` returns the backend's own count, finish reason and timings.** It estimated usage from
  the streamed text and did not mark the result `IsEstimated`. A reasoning model's hidden tokens were therefore
  missing from what looked like a measured count, and the finish reason was dropped. The lm-supply console's chat
  endpoint served that estimate. The method now delegates to `GenerateChatCompleteResultAsync`, as its raw twin
  `GenerateWithUsageAsync` has done since 0.73.0.

## [0.79.3] - 2026-09-26

### Changed
- **Every LMSupply package is marked `IsAotCompatible`, and builds clean under the trimming/AOT analyzers.** 0.78.0 made the JSON paths work in trimmed/AOT hosts at run time; the analyzers still flagged 32 calls (the `JsonSerializerOptions` overloads) in LMSupply.Core, LMSupply.Llama and LMSupply.Synthesizer. Those now use the `JsonTypeInfo` overloads, taken from the same options, so behaviour is unchanged. A trimmed or AOT consumer no longer sees warnings from these packages, and a new reflection-only call fails LMSupply's own build.

## [0.79.2] - 2026-09-26

### Fixed
- **Transcribing from a stream now hears the audio at the right speed, and accepts MP3.** `TranscribeAsync(Stream …)` decoded a WAV as it was — no mono mix-down, no resampling to 16 kHz — so anything but 16 kHz mono (a 44.1 kHz stereo recording, say) reached the model as audio at the wrong speed and came back as noise. An MP3 stream threw, though the file overload decodes MP3. The stream path now does what the file path does, recognising MP3 from its first bytes.

## [0.79.1] - 2026-09-26

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
  (one measured cache held 6.8 GB of byte-identical pairs). That release note was never written; this
  is it. The downloader now looks for a wanted file at the snapshot root (when the target is a
  subfolder) or in an immediate subfolder (when the target is the root) and moves a copy of the listed
  length into place: no request, one copy. Copies of another length are left alone; nothing is moved in
  offline (read-only) mode. Existing duplicate pairs are not removed by this — a reclaim call is a
  separate item.

- **Reranker `auto` on a Medium host takes the GGUF multilingual model when a llama-server binary is
  already cached.** Medium resolved to `quality` (bge-reranker-base), whose model card lists English and
  Chinese as its training languages — on a Korean corpus it ranked worse than no reranking. When a llama-server binary is already in the cache, `auto` now resolves to
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
  library would produce different vectors for the same model id.** Three releases (an earlier SentencePiece token-id fix, 0.70.0
  and this one) changed the vectors a model id produces and said so only in prose; a consumer that stores
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
  subfolder and every existing cache downloaded them again, keeping both copies (one measured cache
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
