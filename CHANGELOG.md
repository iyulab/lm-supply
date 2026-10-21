# Changelog

All notable changes to this project are documented in this file. Versions follow
[Semantic Versioning](https://semver.org/); while the major version is 0, a minor release may contain
breaking changes, and each one is marked **Breaking** with a migration note.

## [Unreleased]

## [0.135.0] - 2026-10-22

### Changed
- **Breaking: every download whose source cannot be reached at all now fails with `ModelHostUnreachableException` naming
  that host** — GGUF models, llama-server builds and the ONNX Runtime packages, not only the ONNX
  model files. A GGUF load surfaced the raw `HttpRequestException`; a llama-server build whose release GitHub could not
  answer (no connection, or the API timeout) failed as `LlamaServerAcquisitionException` with `ReleaseNotResolved`; an ONNX
  Runtime package with no cached version failed as `InvalidOperationException`. Each now raises the typed exception with
  `Host` = `huggingface.co` (or the `HF_ENDPOINT` host), `api.github.com` or the build's download host, or `api.nuget.org`,
  and the network error as its inner exception. A source that answers with an error (404, 403, 5xx) fails as before.
  Migration: code that caught `HttpRequestException`, `LlamaServerAcquisitionException` or `InvalidOperationException`
  for an unreachable source catches `ModelHostUnreachableException` (a `ModelDownloadException`).

### Added
- **Detector: a host can ask before a download.** `DetectorArtifacts` is the detector's `IModelArtifacts` (detection, face,
  plate and pose models alike), and `LocalDetector` gains `GetDownloadSizeBytesAsync`, `GetRemainingDownloadBytesAsync`,
  `IsModelDownloadedAsync`, `DownloadModelAsync` and `DeleteModel`: the one file a load picks from the repository, at the
  length the repository lists, planned by the same discovery as the load (`ModelPathResolver.PlanModelAsync`). Until now a
  host could only read the registry's approximate `SizeBytes`. See [Before a download](docs/detector.md#before-a-download).
- **`RuntimeManager.IsRuntimeAvailableOfflineAsync`, `GetRuntimeDownloadBytesAsync` and `DownloadRuntimeAsync`** answer, for
  one runtime package and provider, whether a load would find it without downloading, how many bytes it would download, and
  fetch it without loading it; `OnnxRuntimeArtifacts` answers the same for the provider an ONNX session of an
  `ExecutionProvider` is created on. `LlamaServerUpdateService.GetRemainingDownloadBytesAsync(backend)` answers for llama-server.
- **Generator: a loaded generator is a Microsoft.Extensions.AI `IChatClient`** — `generator.AsChatClient()` returns a
  `LocalChatClient`, so `ChatClientBuilder` pipelines, `UseFunctionInvocation()` and agent frameworks use a local model
  with no adapter of their own: instructions, images, tool calls and results, reasoning (`Reasoning.Effort` drives
  `ThinkingMode`, reasoning returns as `TextReasoningContent`), sampling options, `ToolMode`, JSON `ResponseFormat`,
  library-only options through `RawRepresentationFactory`, and usage with cached prompt tokens, whole or streamed. The
  client does not own the generator, and `GetService<IGeneratorModel>()` hands out the loaded model (its context
  length). See [Microsoft.Extensions.AI](docs/generator.md#microsoftextensionsai).
- **Synthesizer: the Korean voice speaks text.** A built-in Korean phonemizer turns Korean text into the phonemes the
  `korean` voice was trained on, in pure .NET with no download: figures are read as Sino-Korean numbers and Latin
  letters by their Korean names, each word is said by the rules of the standard pronunciation that follow from its
  spelling (liaison, the seven final sounds and cluster reduction, aspiration, palatalization, tensing, nasal and
  liquid assimilation), and the symbols written are the ones the voice is heard to say best. Spoken and transcribed by
  Whisper, everyday sentences and sentences with figures read back at a character error rate of about 18%.
  `SynthesizeAsync` and the other text methods on the Korean voice, which threw `NotSupportedException`, now return
  speech; a phonemizer set in `SynthesizerOptions.Phonemizer` still replaces the built-in one. The Chinese voice still
  needs phonemes or a phonemizer. See [Korean text](docs/synthesizer.md#korean-text) for what is not handled.

### Changed
- **Breaking — `IModelArtifacts.GetRemainingDownloadBytesAsync` is `GetRemainingDownloadAsync`, and the model kinds count
  the runtime a load runs on.** A consent prompt stated the model alone while the download — or the first load — also
  fetched a runtime: llama-server for a GGUF model (about 1.2 GB more on CUDA with its runtime companion), the ONNX Runtime
  native package for an ONNX one (about 160 MB for the CPU package of 1.31.0). It returns a `RemainingDownload`: `ModelBytes` (what the old method
  returned), `RuntimeBytes` and `TotalBytes`. `RuntimeBytes` is the llama-server build a load would run (the pinned version or
  the latest) and its CUDA companion, read from the release listing; or the ONNX Runtime package for the provider the session
  would be created on (and ONNX Runtime GenAI for an ONNX generator), as the package feed states its length. It is 0 when
  they are cached or supplied by the application, and null when that cannot be told or a load would not download them.
  `IsAvailableOfflineAsync` of an ONNX model now also requires its runtime in the cache — a load with downloads disabled
  fetched the ONNX Runtime package on a machine without it — and `DownloadAsync` fetches it. An ONNX generator without
  `LMSupply.Generator.Onnx` registered is not available offline: such a load fails. Migration:
  `await artifacts.GetRemainingDownloadBytesAsync(id)` → `(await artifacts.GetRemainingDownloadAsync(id)).ModelBytes`, or
  `.TotalBytes` for what the person agrees to download; an `IOnnxGeneratorBackend` implementation adds
  `IsRuntimeAvailableOfflineAsync`, `GetRuntimeDownloadBytesAsync` and `DownloadRuntimeAsync`.
- **Embedder and Reranker GGUF files written before 0.98.0 are moved into the Hugging Face cache layout.** The first
  lookup of a repository moves its files from the private `gguf-embeddings/` or `gguf-rerankers/` tree into the
  repository's `snapshots/main/` and removes the emptied tree, so other Hugging Face tools find them and nothing is
  downloaded again. A cache that cannot be written keeps its tree, which is still read. Both the move and the reading
  of those trees end after 0.150.0, as does the removal of the pre-0.112 `.discovery-cache/` directory and the absolute
  build paths the llama-server state keeps for releases before 0.122.

### Fixed
- **Synthesizer: English words the dictionary lacks are stressed and ended as English says them.** The
  letter-to-sound rules stressed every word of three or four syllables on its first one (`agenda`, `origami`,
  `avocado`), read a final i after a consonant as in «it» and a final ai as in «rain», reduced the stressed vowel of
  words such as `corona`, and said each letter of a doubled consonant (`bassetti` with two s sounds). A word of three
  syllables or more ending in a consonant and a, i or o is now stressed on the syllable before the last (one ending in
  -ica on the one before that), endings that do not move the stress (`-s`, `-ed`, `-ing`, `-er`, `-ly`, `-ness`,
  `-ment`, ...) are set aside before the stress is placed, a final i after a consonant is «ee», a final ai «eye», and a
  doubled consonant one sound. Read against the shipped dictionary, the rules now stress 86% of its words as it does
  (was 79%) — 91-95% of the long words ending in a consonant and a, i or o (was 4-30%) — and say 27% of them exactly
  (was 23%).
- **Synthesizer: a two-syllable English word with a prefix the dictionary lacks is stressed on its root.** The
  letter-to-sound rules stressed the root only after be-, de- and re-, and there also when the letters after them
  cannot begin a syllable (`beckon`, `dentist`, where the vowel is the word's own). The root now takes the stress
  after un-, dis-, mis-, ex-, en-, im-, ab-, af-, ap-, at-, ob-, sup- and trans- as well (`unpaid`, `exceed`,
  `dismiss`), and after be-, de- and re- only before a consonant or cluster a syllable can begin with. Read against the
  shipped dictionary, the rules now stress 87% of its words as it does.
- **Synthesizer: a gh or ough inside an English word the dictionary lacks is said as English says it.** The
  letter-to-sound rules read a gh after a consonant as g and h (`Afghan` with an h), kept silent a gh between single
  vowels (`Callaghan`, `Allegheny`), and said every ough as in «rough». A gh there is now a g, and an ough is said as in
  «tough» after t, n and an initial r, as in «dough» after d, as in «cough» after c, and as in «bough» otherwise. Of the
  shipped dictionary's words with a gh inside, the rules now say 48% with its sounds (was 44%).
- **Synthesizer: the unstressed vowels of an English word the dictionary lacks are reduced as English reduces them.**
  The letter-to-sound rules gave an unstressed short a, e or o its full sound (`Attenborough` with the «e» of «bet»,
  `Madison` with the «o» of «lot», `emoji` with the «e» of «bet»), an unstressed i before a vowel as in «it»
  (`accordion`), and no vowel a secondary stress. Such an a, e or o is now a schwa — at the start of the word, a schwa
  for a and the «i» of «it» for e — an i before a vowel the «ee» of «see», and an unstressed diphthong keeps a
  secondary stress (`Holloway`). Read against the shipped dictionary, the rules now say 34% of its words exactly (was
  27%).
- **Synthesizer: the stressed vowel of an English word the dictionary lacks follows its spelling.** The
  letter-to-sound rules said a stressed o before one consonant and a vowel as the o of «lot» (`emoji`, `Alamosa`), one
  before two consonants as the u of «cut» (`accommodate`), an ei as in «receive» (`Beilman`), and an e before a
  consonant and -er or -ing as in «see». A stressed o there is now «oh», and the o of «lot» before two consonants, an
  ei is «eye», and that e the e of «bet».
- **Synthesizer: an s, z or ch in an English word the dictionary lacks is said as English says it.** The
  letter-to-sound rules said a z after t as z (`Horowitz`), a final s after the vowel of a plural as s (`cities`,
  `Fridays`), and an s between vowels as z before a final -on, -a, -i or -o and in -sive (`Acheson`, `Mesa`,
  `abusive`). Each is now said as the shipped dictionary has it, and so is a ch before l, r, t or s: a k (`Chloe`,
  `Christ`), not the ch of «church».
- **Generator: a streamed tool call on a Gemma 4 GGUF model is no longer lost.** `GenerateChatStreamAsync` ended the
  turn with finish reason `tool_calls` but no `ChatToolCallDelta`: the stream kept only the calls its Gemma 4 wrapper
  parser found in the text, and llama-server builds that parse the wrapper themselves send the call as structured
  deltas, with no wrapper in the text. Server deltas now win wherever they are present, for every chat format, and the
  wrapper parser fills in the chunks where the server sent none. **Breaking** for a custom `IChatFormatter` that
  overrode `SuppressServerToolCallsWhenParserActive`: the member is removed — delete the override.
- **A cached GGUF model loads with `HTTP_PROXY` / `HTTPS_PROXY` (or a system proxy) set.** The llama-server readiness
  check and the API client sent their loopback requests to the proxy, which cannot reach the caller's loopback, so the
  local server was never seen as ready and the load did not return. Requests to a loopback llama-server now bypass the
  proxy; a `LlamaServerClient` pointed at a remote server keeps the default proxy behaviour.
- **A Hub download works with a cache directory written in another form.** With `HF_HUB_CACHE` or `HF_HOME` written
  with forward slashes on Windows (`C:/…`), or through a `.` segment anywhere, every download failed with «Path
  traversal detected in file path»: the check compared each file's full path with the directory as written. Every
  check that a path lies inside a cache directory now compares both in full form, so the same directory written either
  way contains the same files; a sibling directory whose name only begins with the cache's no longer counts as inside
  it, and outside Windows a directory differing only in case is another directory.

## [0.134.0] - 2026-10-21

### Added
- **Model downloads and Hub requests honour `HF_ENDPOINT`**, the Hub base URL variable `huggingface_hub` reads — a
  mirror or an internal hub for a machine that cannot reach huggingface.co. Every Hub URL uses it: file downloads,
  repository listings, model metadata and revision lookups. Unset, nothing changes; a value that is not an absolute
  http(s) URL fails the download with a message naming the variable instead of falling back to huggingface.co.
- **`ModelHostUnreachableException`: a model that is not cached and whose host cannot be reached at all fails with
  its own type**, carrying `Host`, so a host can tell «this network cannot reach the model host» (retrying will not
  help) from a host that answered. **Breaking** only for code that checks the exact type: it derives from
  `ModelDownloadException`, so `catch (ModelDownloadException)` still catches it — an exact-type assertion such as
  `Assert.ThrowsAsync<ModelDownloadException>` becomes `ThrowsAnyAsync`, or names the new type.

## [0.133.0] - 2026-10-21

### Added
- **`ITextGenerator.GenerateStreamAsync(prompt)` streams a raw-prompt completion in structured chunks**
  (`TextStreamChunk`), as `GenerateChatStreamAsync` does for chat: text deltas with their log probabilities, the
  server's progress reading a long prompt before the first token (`PromptProgress`, llama-server backend), and a final
  chunk with `FinishReason` and, on llama-server, `Usage` and `Timings`. Until now a raw-prompt stream said nothing
  while a long prompt was read, so a caller could not tell reading from a stalled backend. **Breaking** for a type that
  implements `ITextGenerator` (or `IGeneratorModel`) itself: it must add the member — forwarding to an inner
  generator, or yielding its text as `TextStreamChunk.Text` and a last chunk with the finish reason.

### Fixed
- **The README and guides no longer call ONNX Runtime 1.30 the current runtime** (it is 1.31). The CUDA 13
  requirement is stated as what it is — true of every GPU build since 1.27 — and the version LMSupply uses is named
  in one place, `docs/GPU_PROVIDERS.md`.
- **A GGUF model that started with fewer GPU layers after running out of GPU memory reports the layers it runs
  with.** When llama-server failed to start for lack of GPU memory, the load retried with about a quarter fewer layers,
  but `GeneratorModelInfo.GpuLayers` kept the count first planned — or none, when the caller had set the count
  through `LlamaOptions.GpuLayerCount`. It is now the count the server started with.

## [0.132.0] - 2026-10-20

### Fixed
- **A llama-server whose owner was killed is stopped by the next process that starts one, on Linux and macOS.** On
  those systems nothing ended a server when the process that started it was killed or crashed — its exit hooks never
  ran — so the server kept its port and the model's memory until stopped by hand. Each server now leaves an owner record
  (`owners/` in the llama-server cache directory) that a clean stop removes, and before a server starts, any recorded
  server whose owner is gone is stopped. A process is matched by its id, start time and name, so a reused id is never
  taken for a server. Windows already ended such servers with their owner.
- **An embedding or reranking llama-server keeps no RAM prompt cache.** Such a request always processes its whole
  input, so a prompt it leaves in the cache is never read back, yet llama-server builds before the upstream fix cached it
  anyway and held up to the cache limit (8 GiB by default) for nothing. LMSupply now starts these servers with
  `--cache-ram 0`; a generation server keeps llama-server's default.
- **A partially offloaded GGUF model reports its RAM share without counting the KV cache twice.** When not every layer
  fits in GPU memory, `GeneratorModelInfo.EstimatedRamBytes` (and the offload trace) added the KV cache to a total that
  already contained it, so the RAM share read one KV cache too high — more than the whole model file for a model with only
  a few layers left on the CPU. The number of GPU layers is unchanged.

### Changed
- **ONNX Runtime 1.31.0 and ONNX Runtime GenAI 0.18.0; the OpenVINO execution provider plugin goes to 1.6.1.** GenAI
  0.18.0 requires ONNX Runtime 1.31.0 or later, so the two move together. OpenVINO plugins 1.7.0 and 1.8.0 cannot create
  their provider on ONNX Runtime 1.31.0 (they look up experimental functions that release moved to its stable API), so
  `ExecutionProvider.OpenVino` uses plugin 1.6.1, which can; it moves forward once a later plugin release works with 1.31. ONNX Runtime 1.31.0 creates a session from a copy of its
  `SessionOptions`, so options released while a session is being created no longer matter. Its native builds now use
  the 1DS telemetry uploader on Windows too; LMSupply keeps setting `ORT_DISABLE_TELEMETRY=1` first, which that uploader
  honours. Migration: a consumer that pins `Microsoft.ML.OnnxRuntime`, `Microsoft.ML.OnnxRuntime.Gpu` or
  `Microsoft.ML.OnnxRuntimeGenAI` below these versions gets NU1605 on restore — raise the pin.

## [0.131.0] - 2026-10-20

### Fixed
- **A downloaded model is sized by its content, not by its cache link** — a model file in the cache is a snapshot
  entry linking to its blob, and its size was read from the link itself: 0 bytes on Windows, a few dozen bytes elsewhere.
  Loading a GGUF model whose layers did not all fit in GPU memory threw `DivideByZeroException` on Windows, and on other
  systems GPU offload, the context-length bound and the memory estimate of every model type (`EstimatedMemoryBytes`,
  registry `SizeBytes`) were computed from that size. A GGUF `block_count` of 0 now reads as unknown
  (`GgufMetadata.LayerCount` is null), and `MemoryEstimator.EstimateForGguf` rejects a non-positive file size or layer
  count with `ArgumentOutOfRangeException` instead of dividing by it.
- **`Reranker` and `MemoryAwareGenerator` implement `IDisposable` as well** — a container disposed with `Dispose()` (a `using` provider or scope) no longer throws «only implements IAsyncDisposable» when it created one.

### Changed
- Dependencies: `Microsoft.Extensions.AI.Abstractions` 10.11.0 (`LMSupply.Embedder`), `NAudio` 3.3.0
  (`LMSupply.Transcriber`). Migration: a consumer that pins either below these versions gets NU1605 (package downgrade)
  on restore — raise the pin.

## [0.130.0] - 2026-10-20

### Added
- `TranscribeOptions.Seed` makes a transcription reproducible. A window that fails the quality checks — audio without
  speech often does — is decoded again by sampling, and those draws differed on every call, so the same audio could
  read differently twice. With a seed, the same audio and options give the same text each time; without one,
  nothing changes.

### Changed
- **Breaking (macOS): `ExecutionProvider.Auto` no longer picks CoreML for ONNX sessions — they run on the CPU.**
  With the models LMSupply loads, CoreML took the host process down instead of failing a session, where no fallback
  can catch it: an invalid memory access on every synthesis with a Piper voice (CoreML's NeuralNetwork format), and a
  Metal assertion while compiling another model when the newer ML Program format was tried. `ExecutionProvider.CoreML`
  still runs a session on CoreML when asked for. llama-server's Metal backend (GGUF models) is unaffected. On an Apple
  Silicon Mac `auto` now also sizes models to the system's memory: the GPU reported no memory, the hardware tier came
  out `Low` on every Mac, and `auto` picked the smallest models — for the reranker, an English-only one. The GPU now
  carries the working-set budget Metal states for it (`recommendedMaxWorkingSetSize`).
  Migration: none for most callers. A caller that relied on `Auto` meaning CoreML on a Mac passes
  `ExecutionProvider.CoreML` and checks its own models on it.

### Fixed
- **Creating an ONNX session no longer risks reading freed session options.** The options were left for the garbage
  collector, and the ONNX Runtime binding passes only their native handle while a session is created, so a collection
  during a slow session creation could free them mid-call — seen once as a process crash (a bus error inside ONNX
  Runtime's session initialization) in a long run of loads. The options are now released once the session exists.
- **A load that runs on the CPU ranks quantized variants as the CPU runs them.** The tier from system memory was read
  with the GPU's order, so a CPU machine with 32 GB or more put FP16 first: the ONNX Runtime CPU provider has FP16 kernels
  for only part of its operators, and some FP16 exports do not load there at all — Florence-2's merged FP16 decoder is
  refused as an invalid model. On the CPU, full precision and 8-bit now come first and FP16 last
  (`ModelPreferences.ForCpuTier`), for an explicit `ExecutionProvider.Cpu` and for `Auto` where the hardware recommends
  the CPU.

## [0.129.0] - 2026-10-20

### Added
- **`LlamaServerVersionRequirements.MinimumSecureBuild` (b8146) and `IsBelowSecurityFloor`**: the oldest llama-server build
  that no published llama.cpp security advisory affects — b8146 fixes GHSA-3p4r-fq3f-q74v (CVE-2026-27940), the newest fix
  among them. `LlamaServerUpdateOptions.AllowVulnerableBuild` runs an older build anyway, with a traced warning.
- `LlamaServerStateManager.RollbackAsync(backend, platform, eligible, cancellationToken)`: a rollback to the newest previous
  version the predicate accepts.

### Changed
- **Breaking: a llama-server build below b8146 is no longer run.** When nothing is pinned and downloads are allowed, the latest
  build replaces it. Otherwise acquisition fails — a `PinnedVersion` below it, a `ServerBinaryPath` whose build reads or is
  declared as below it, a cached build below it on a load with downloads disabled — and a rollback passes over such builds.
  Migration: pin or ship b8146 or later; or set `LlamaServerUpdateOptions.AllowVulnerableBuild = true` to keep running the
  older build. A binary whose build cannot be read is run as before.
- Dependencies: `System.Numerics.Tensors` 10.0.13, `Microsoft.ML.Tokenizers` 2.0.1, `NAudio` 3.2.0.
  Migration: a consumer of `LMSupply.Captioner`, `LMSupply.Embedder`, `LMSupply.ImageGenerator` or `LMSupply.Ocr` that pins
  `System.Numerics.Tensors` below 10.0.13 gets NU1605 (package downgrade) on restore — raise that pin to 10.0.13 or later,
  together with any other .NET 10.0.x servicing pins kept on one band.
- **ONNX Runtime no longer sends its own telemetry from a process that uses LMSupply.** The official ONNX Runtime
  builds for Linux, macOS and Windows outside Windows apps upload trace events and a persistent device identifier to
  Microsoft by default. LMSupply now sets `ORT_DISABLE_TELEMETRY=1` before it first creates a session or touches the
  ONNX Runtime environment (in the native environment as well, where ONNX Runtime reads it on Linux and macOS), so the
  uploader is never created. A process that has set the variable itself, to any value, keeps it — set
  `ORT_DISABLE_TELEMETRY=0` before the first load to keep ONNX Runtime's default. Child processes inherit the variable.
  This also removes a crash at process exit on macOS, where the uploader's thread could abort the process while a
  response arrived during shutdown.

### Removed
- **Breaking: `DetectorModels.RtDetrR18`, `RtDetrR50`, `RtDetrR101` and `EfficientDetD0`.** They were other names for
  RT-DETR v2 models, and `EfficientDetD0` named an architecture it did not return (it was RT-DETR v2 Mini-Small).
  Migration: `RtDetrR18` → `DetectorModels.RtDetrV2S`, `RtDetrR50` → `RtDetrV2M`, `RtDetrR101` → `RtDetrV2L`,
  `EfficientDetD0` → `RtDetrV2MS` (alias `fast`).
- **Breaking: `WellKnownModels.Generator.Phi35Mini`**, a second name for `WellKnownModels.Generator.Medium` (the same
  model id). Migration: use `Medium`. The `phi-3.5-mini` alias and `DefaultGeneratorModels.Phi35Mini` stay.

### Fixed
- **The console host's system status and metrics stream report the process's memory as it is now.** `GET /api/system/status`
  (`processMemoryMB`) and each tick of `GET /api/system/metrics/stream` (`ramUsageMB`) repeated the working set the host had
  the first time it was asked, and a metrics stream still open when the host shut down failed with «No process is associated
  with this object». Both now read the figures live, and the CPU share shared by status reads and open streams is sampled
  atomically.
- **`scripts/test.ps1` reads its counts on a machine whose UI language is not English.** The test platform translates
  its summary keys there, so no count matched and the run reported no summary; the script now runs with
  `DOTNET_CLI_UI_LANGUAGE=en`.

## [0.128.0] - 2026-10-19

### Added
- **The speaker-diarization pair is a lifecycle kind of its own: `DiarizationArtifacts`** (`Kind = "diarization"`, one model id,
  `DiarizationArtifacts.DefaultModelId` = `"default"`). A `ModelLifecycle` that registers it beside `TranscriberArtifacts` holds
  «transcription available» and «speaker labels available» apart: the pair gets its own status, its own shared and
  cancellable download, and its own delete (the transcriber model is kept). With `DisableAutoDownload` its download only
  confirms the pair is cached.

### Changed
- **Breaking: `TranscriberArtifacts` covers the transcription model only.** `IsAvailableOfflineAsync`,
  `GetRemainingDownloadBytesAsync` and `DownloadAsync` no longer include the speaker-diarization pair when the options set
  `PreloadDiarization`, so a cached model reads as available before the pair is installed. Migration: register
  `new DiarizationArtifacts(options)` in the same `ModelLifecycle` and treat the transcriber as ready when both kinds are.
  `PreloadDiarization` still preloads the pair at load time, and `LocalTranscriber`'s static size, cache and download methods
  are unchanged.

### Fixed
- **Several loads of one cached model at once no longer fail with «The process cannot access the file
  `.lmsupply-manifest.json` because it is being used by another process».** Every load with downloads allowed rewrote the
  snapshot's download manifest even when every file was cached, and concurrent writers (or a reader beside them) collided on
  the file. A load that finds nothing new now leaves the manifest untouched; a manifest that does change is written to a file
  of its own and replaces the old one in one step (writers in one process take turns, a writer in another process is
  retried), and readers open it shared, so they see the old manifest or the new one, never a partial one. Loads with
  `DisableAutoDownload` never wrote it and are unchanged.

## [0.127.0] - 2026-10-18

### Added
- **A shared download can end when its last waiter leaves: `ModelLifecycle.DownloadAsync(kind, modelId, DownloadWaitMode, ct)`.**
  With `DownloadWaitMode.CancelWhenAbandoned`, a caller that stops waiting cancels the download if nobody else is waiting on
  it and no `DownloadWaitMode.KeepRunning` caller joined it (partial files stay, as with `CancelDownload`). The download
  counts its waiters itself, so two callers that start it in the same instant, or one that starts it and one that joins, no
  longer need a status read to decide who may stop it. A caller that arrives while an abandoned download is still winding
  down waits for it and starts a new one. The existing `DownloadAsync(kind, modelId, ct)` keeps today's behaviour
  (`KeepRunning`).

## [0.126.0] - 2026-10-18

### Added
- **A streamed chat on the llama-server backend reports how far the server has read the prompt:
  `ChatStreamChunk.PromptProgress`** (`PromptProgress`: `TotalTokens`, `CachedTokens`, `ProcessedTokens`, `Elapsed`). Reading a
  long prompt could leave `GenerateChatStreamAsync` silent for minutes before the first token, so a host guarding a turn
  could not tell a prompt still being read from a backend that had stopped. The stream now yields a chunk carrying only
  `PromptProgress` when the server starts reading and after each batch it evaluates (`return_progress`, sent on every
  streamed chat; a build that predates it ignores it). A consumer that reads only `Text` sees no change. Null on the ONNX
  backend, which reads the prompt in one step. `LlamaServerClient`'s `ChatStreamData.PromptProgress` carries the server's
  `prompt_progress` (`LlamaPromptProgress`).

## [0.125.0] - 2026-10-17

### Changed
- **Breaking: the console host's `POST /v1/chat/completions` rejects a message role it does not know.** A role outside
  OpenAI's set (`system`, `developer`, `user`, `assistant`, `tool`, `function`) was read as a user message; it is now a
  400 naming the message (`messages[i].role`), before any model loads. `developer` — OpenAI's newer name for the system
  role — is the system prompt, and `function` is read like `tool`. Migration: send one of those roles.

### Added
- **A host manages rerankers, transcribers, captioners and OCR like generators and embedders: `RerankerArtifacts`,
  `TranscriberArtifacts`, `CaptionerArtifacts` and `OcrArtifacts`** (`IModelArtifacts`, so a `ModelLifecycle` takes them; OCR's
  id is a language code, as for `LoadForLanguageAsync`). Each answers whether a load with downloads
  disabled finds everything in the cache — for a GGUF reranker its llama-server too, for a transcriber that preloads speaker
  diarization its model pair too — the bytes still to fetch, the download that makes it so, and the delete.
  `LocalTranscriber.DownloadModelAsync(id, options, progress)` and `LocalCaptioner.DownloadModelAsync(id, options, progress)`
  download what a load needs without loading it (each load and its download share one path),
  `LocalCaptioner.IsModelDownloadedAsync(id, options)` answers from the cache whether that load would make no request, and
  `LocalReranker.DeleteModel(id, cacheDir)` / `LocalTranscriber.DeleteModel(id, options)` / `LocalCaptioner.DeleteModel(id,
  options)` delete LMSupply's files of the model the id resolves to. For OCR: `LocalOcr.IsLanguageDownloadedAsync`,
  `DownloadForLanguageAsync` (the load's download path, no session) and `DeleteLanguageModel` (the language's recognizer only —
  the detection model every language shares stays) — never a local path, llama-server, a reranker's separate tokenizer
  repository or the shared diarization pair.
- **`LMSupply.Decider`: typed decisions from a local model.** `LocalDecider.LoadAsync` loads a GGUF model and
  `IDeciderModel.DecideAsync` answers named questions about a state — `DecisionQuestion.Choice` (your options, optionally
  «none of these»), `YesNo` and `Score` (2–10 levels) — with a probability for every answer (`Decision`, `DecisionResult`;
  `Confidence`, `LabelMass`, `Path`, `Calibrated`). A model whose llama-server reports the `decisions` output modality
  answers natively through `/v1/systemone` (every question in one request); any other instruct model is read from the
  log probabilities of one generated token (thinking off), renormalized over the answer labels, with the questions of a call
  sent in order so the server reuses the state's evaluation. `DeciderOptions.Temperature` applies a temperature per question
  kind (p ∝ p^(1/T)). Decisions that need arithmetic or several steps of reasoning are out of scope. See
  [docs/decider.md](docs/decider.md).

## [0.124.0] - 2026-10-17

### Changed
- **Breaking** — `IGeneratorModel` has a new member, `TokenizeAsync(string, CancellationToken)`. Migration: an
  implementation of your own (a wrapper or a test double) adds it — a wrapper delegates to the model it wraps.
- **Breaking: `DisableAutoDownload` keeps a GGUF load off the network entirely.** It covered model files only, so a GGUF
  generator, embedder or reranker loaded with it still downloaded llama-server when none was cached, and a cached server
  still ran the background update check — «downloads disabled» was not «no request». Now the server comes from the
  llama-server cache, an external `ServerBinaryPath` or a cached `PinnedVersion` only; a missing server throws
  `ModelNotFoundException` (`ModelId` `"llama-server"`), and no update check runs at load or in the background. A cached server
  older than the model needs is reported (`LlamaServerVersionRequirements`) instead of updated. Migration: download the
  model once with downloads allowed (its server is cached with it), or point `LlamaServerUpdateOptions.ServerBinaryPath` at a
  bundled server. `LlamaServerUpdateService.AcquireServerAsync(backend, localFilesOnly, …)` is the acquisition a load makes
  (a `LlamaServerInstallation`; throws instead of returning a failed result). The ONNX Runtime keeps its process-wide switch
  (`RuntimeManagerOptions.DisableAutoDownload`).

### Added
- **A host can ask whether a model is ready to load offline, and download exactly that on consent:
  `LMSupply.Lifecycle.IModelArtifacts`**, with `GeneratorArtifacts` and `EmbedderArtifacts` (built with the options the load
  will use). `IsAvailableOfflineAsync` answers from the cache whether a load with `DisableAutoDownload` would find everything —
  the model files and, for a GGUF model, llama-server; `GetRemainingDownloadBytesAsync` gives the model-file bytes still to
  fetch; `DownloadAsync` fetches the files and, for GGUF, llama-server, so the model is available offline afterwards
  (`DownloadModelAsync` alone does not fetch the server).
- **`ModelLifecycle` keeps the models a host manages, across calls.** Over the providers of the model kinds it uses: each
  model's status (`Missing`, `Downloading` with the latest `DownloadProgress`, `Ready`, `Failed` with the exception the download
  threw), `WatchAsync` — a stream of every change — and `DownloadAsync`, shared by every caller asking for the same model and
  cancellable by any caller with `CancelDownload` (a caller's own token only stops its wait). It never downloads on its own.
- **A host can delete a model by the id it loads.** `ModelLifecycle.DeleteAsync(kind, id)` removes the files a load of that
  id opens and publishes the status it leaves (`Missing`, or `Ready` when another tool's files remain); it is refused while
  that model downloads, and a download is refused while it is being deleted. Providers implement `IModelArtifacts.Delete`;
  `LocalGenerator.DeleteModel(id, options)` and `LocalEmbedder.DeleteModel(id, cacheDir)` resolve the id as
  `IsModelDownloaded` does. A `gguf:` registry alias deletes only the file it loads, so aliases that share a repository
  (`gguf:gemma4-default` / `gguf:gemma4-balanced`) keep each other's files — `CacheManager.DeleteModelFiles(cacheDir, repoId,
  files)` is that file-level delete, and the last file takes the repository's private state with it. A local path and
  llama-server are never deleted.
- **A GGUF model reports the log probability of each token it generates.** `GenerationOptions.TopLogprobs` (null by
  default) asks for them: 0 for each generated token's own value, n for the n most likely candidates at each position
  as well. They arrive as `TokenLogprob` entries (`TokenId`, `Token`, `Logprob`, `Bytes`, `TopLogprobs` — the shape of
  OpenAI's `logprobs.content`) on `GenerationResult.Logprobs` (raw prompt and chat), `ChatCompletionResult.Logprobs` and,
  token by token, `ChatStreamChunk.Logprobs`; reasoning tokens are included. The values are the model's raw next-token
  distribution, taken before sampling, so temperature, top-k/top-p/min-p, penalties, logit bias, a grammar and a JSON
  schema do not change them, and the same prompt gives the same values whatever the seed. llama-server cannot combine
  them with tools on a streamed completion, so that combination throws `ArgumentException` before the request is sent
  (`GenerateChatWithToolsAsync`, which is not streamed, returns them). The ONNX backend and `SpeculativeDecoder` throw
  `NotSupportedException` when `TopLogprobs` is set. On `LlamaServerClient`, `ChatCompletionOptions.TopLogprobs` and
  `CompletionOptions.TopLogprobs` do the same, and `ChatStreamData`, `CompletionStreamData` and
  `ChatCompletionFullChoice` carry `LlamaTokenLogprob` entries.
- **The console host's `POST /v1/chat/completions` takes `logprobs` and `top_logprobs`** with OpenAI's meaning
  (`top_logprobs` 0–20, only with `logprobs: true`) and returns `choices[].logprobs.content` — `token`, `logprob`,
  `bytes`, `top_logprobs` — streamed (each chunk's tokens) and not. The fields were accepted and ignored. A request
  outside those rules, with `draft_model`, or on a streamed strict `json_schema` response is a 400, and so is a model
  whose backend cannot report log probabilities (ONNX).
- **A model's token ids for a text.** `IGeneratorModel.TokenizeAsync(text)` returns them — on a GGUF model through
  llama-server's `/tokenize`, without the tokens the model adds around a whole prompt, and with text that spells a
  special token (such as `<think>`) read as that token; on an ONNX model through its tokenizer. Use it to find the id of
  the token whose log probability you read: ids differ between models, and `"B"` and `" B"` are different tokens.
  `LlamaServerClient.TokenizeAsync(text, addSpecial)` is the client-level call.

### Fixed
- **Every model file is reported `Complete` once, and a load completes its runtime once.** Two parts of the
  progress rule («`Complete` — the file is in place, downloaded or found in the cache; once per file») held for GGUF
  models only. Files fetched through the Hugging Face downloader (every ONNX model, and the reranker's files) ended on
  their last `Downloading` report, and found in the cache they were not reported at all, so a warm ONNX load reported
  `Loading` for files it never said were there. Now each such file gets one `Complete` report with its size, with the
  multi-file figures, under the name its download and its `Loading` report use (its path in the repository, e.g.
  `onnx/model.onnx`); a GGUF file that was downloaded gets one too. And a load that creates several runtime sessions (an
  image generator's three, a transcriber's two) reported the cached ONNX Runtime `Complete` once per session; every
  `LoadAsync` now reports it once. A reporter reused for the next load sees that load's runtime step again.
- **`LocalEmbedder.IsModelDownloaded` and `LocalEmbedder.DownloadModelAsync` handle GGUF models.** `IsModelDownloaded` looked
  for `model.onnx` only, so a cached GGUF embedder (`gguf:org/repo`, a `-GGUF` repository, a `.gguf` path) always read as not
  downloaded; `DownloadModelAsync` sent a GGUF id down the ONNX discovery path. Both now take the GGUF branch the load takes:
  the file an offline load opens counts as downloaded, and the download fetches that file.
- **The console host's `POST /v1/audio/speech` uses the `model` a request names.** `voice` defaulted to `alloy`,
  which maps to the default voice, so a request that named a synthesizer model and no voice was spoken by the default
  voice. An OpenAI voice name sent as `voice` still picks the default voice, as before.
- **The ONNX chat path keeps `NoRepeatNgramSize`, and the reasoning helpers keep every option.** Merging the chat
  formatter's stop sequences built a new `GenerationOptions` that listed the options it carried and dropped
  `NoRepeatNgramSize`; `GenerateChatWithReasoningAsync` and `GenerateWithReasoningAsync` likewise dropped `Thinking`,
  `Seed`, `MinP`, the penalties, `Grammar`, `JsonSchema` and `Tools`. Both now copy every option. The reasoning helpers
  return text only, so they throw `NotSupportedException` when `TopLogprobs` is set.
- **`SpeculativeDecoder` gives the draft model every option the caller set.** Its copies of `GenerationOptions` listed
  the options they carried, so the draft model was not given `Seed`, `MinP`, `Thinking`, the frequency and presence
  penalties, `Grammar`,
  `JsonSchema` or `Tools`, and drafted tokens the target then had to reject. Only the draft's temperature and the
  length of each speculation now differ from what the caller set.

## [0.123.0] - 2026-10-16

### Added
- **A cache listing can tell which repositories LMSupply downloaded.** `CacheManager.HasOwnFiles(cacheDir, repoId)` and
  `CachedModelInfo.HasOwnFiles` are true exactly when `DeleteModel` would delete something — an own snapshot, the private
  download state or the metadata file; false for a repository only other tools wrote into the shared hub cache.
- **The synthesizer speaks English text out of the box.** `SynthesizeAsync` and the other text methods work for the
  English voices of the catalog (`default`, `lessac`, `fast`, `quality`, `british`) with no phonemizer set: a built-in
  English phonemizer reads the text (numbers, years, times of day, dollar amounts, percentages, a closed set of
  abbreviations, pause punctuation), pronounces each word from the CMU Pronouncing Dictionary — or, for a word it does not
  have, from its regular endings, as a compound, or by the letter-to-sound rules of NRL Report 7948 — and writes the
  phonemes in the voice's own accent: American for `en-us`, British-style (non-rhotic, with the British vowels the
  spelling decides) for `en` and `en-gb-x-rp`. It writes only symbols the voice's `phoneme_id_map` has; for a few it has
  a fixed substitute, and a voice that lacks any other phoneme of its accent throws `NotSupportedException` for text.
  Spoken by each English voice and transcribed by Whisper, a set of plain sentences, sentences with figures and words not
  in the dictionary reads back at a word error rate of 0.01-0.08 (at most 0.03 on the plain sentences and figures). The dictionary is embedded in the assembly (about
  0.9 MB), so text works offline from the first use; its licence ships in the package (`THIRD-PARTY-NOTICES.md`,
  `licenses/cmudict/LICENSE`). A phonemizer set in `SynthesizerOptions.Phonemizer` still replaces it for any voice.

### Fixed
- **Package READMEs no longer name DirectML as a GPU provider.** It was removed in 0.67.0; the feature lists of the
  Detector, Embedder, Reranker, Segmenter, Synthesizer, Transcriber and Translator packages, and the provider lists of
  Core, Generator, Generator.Onnx, ImageGenerator and Ocr, still named it.
- **A GGUF chat model in the cache lists as `Generator`, not `Unknown`.** `CacheManager.DetectModelType` (and so
  `CachedModelInfo.DetectedType`) recognised a generator only by the ONNX GenAI `genai_config.json`; a repository holding
  `*.gguf` files now reads as a generator unless its name already says embedder, reranker, transcriber or synthesizer
  (`bge-m3-GGUF` stays an embedder).

### Changed
- **Breaking** — **a GGUF embedder, reranker or token count that llama-server refuses now says why.** These calls threw
  `HttpRequestException` («Response status code does not indicate success: 500») and dropped the server's answer. An input
  longer than the embedder's batch (which the embedders size to the model's sequence length) now throws
  `ContextLengthExceededException` with `TokenCount` and `MaxContextLength` — e.g. 12,639 tokens against 8,192 — and any
  other refusal throws `InferenceBackendException` carrying the status code and the server's body, as chat calls already
  did. Migration: catch `ContextLengthExceededException` / `InferenceBackendException` (both `InferenceException`) where you
  caught `HttpRequestException` around `EmbedAsync`, `RerankAsync` or a GGUF token count. A connection failure is still an
  `HttpRequestException` (or `InferenceBackendExitedException` when the server process exited).
- **Text for a voice in a language other than English still throws `NotSupportedException`**, now saying that the
  built-in phonemizer covers English voices only (the Korean and Chinese voices need phonemes or a phonemizer of yours).

### Removed
- **Breaking** — `LMSupply.Generator.Models.ToolDefinition`, a second public spelling of `ChatToolDefinition` (with
  implicit conversions both ways). Migration: use `ChatToolDefinition`, which `GenerationOptions.Tools` already takes.
- **Breaking** — the synthesizer is no longer `[Experimental]`: `LocalSynthesizer` and `ISynthesizerModel` no longer
  report `LMSUPPLY001`, and `SynthesizerDiagnostics` (its `ExperimentalId` and `ExperimentalUrl` constants) is removed.
  Migration: delete `LMSUPPLY001` from `NoWarn` and `#pragma warning disable` lines, and any reference to
  `SynthesizerDiagnostics`.

## [0.122.0] - 2026-10-15

### Fixed
- **A relocated cache keeps its llama-server builds.** The state file recorded each build as an absolute path, so after
  the cache root moved (`LMSUPPLY_CACHE_DIR`, a moved profile, a renamed drive) the retention sweep found no reference to
  the builds under the new root and deleted them — every backend's install and rollback builds — while the server kept
  running the copy under the old root if it still existed. Builds are now recorded by version and backend directory and
  resolved against the cache root in use; a state file written by an earlier release is migrated on read. A recorded
  build that is not under the root counts as not installed everywhere (`GetUpdateInfoAsync` returns null,
  `CheckAndApplyUpdateAsync` acquires as on a first load, `RollbackAsync` skips previous builds that are gone), and
  adopting a cached build keeps the entry's failed versions and the rollback builds that still exist.
- **The retention sweep reclaims a backend directory no entry references** even when another backend of the same build
  version is kept (a `b*/cuda12` directory survived while the CPU entry referenced `b*/cpu`).

### Changed
- **Breaking** — **no two top-level public types of the LMSupply packages share a name**, so an application that imports
  several packages' namespaces no longer hits CS0104. A test now holds this for every package. Renames — migration is the
  new name:
  - `ModelInfo` → `EmbedderModelInfo` (Embedder), `RerankerModelInfo` (Reranker), `CaptionerModelInfo` (Captioner) and
    `GeneratorModelEntry` (Generator — `GeneratorModelInfo` is the loaded model's info).
  - `DefaultModels` → `CaptionerModels`, `DetectorModels`, `ImageGeneratorModels`, `RerankerModels`, `SegmenterModels`,
    `SynthesizerModels`, `TranscriberModels`, `TranslatorModels`.
  - ImageGenerator `GenerationOptions` → `ImageGenerationOptions`; Reranker `ModelArchitecture` → `RerankerArchitecture`;
    the llama-server wire `ToolDefinition` → `LlamaToolDefinition`.
  - Generator's own `Quantization` enum is gone: the memory estimator uses `LMSupply.Core.Download.Quantization`
    (`FP32` → `Default`, `FP16` → `Fp16`).
- **Breaking** — **one `BoundingBox`.** The Detector (`X1`/`Y1`/`X2`/`Y2`) and OCR (`int X`/`Y`/`Width`/`Height`) boxes are
  replaced by `LMSupply.Vision.BoundingBox` (`float X`/`Y`/`Width`/`Height`, with `Right`, `Bottom`, `IoU`, `Clip`).
  Migration: add `using LMSupply.Vision;`; Detector `X2`/`Y2` → `Right`/`Bottom`; OCR coordinates are now `float` (cast
  where you need whole pixels); `BoundingBox.FromPolygon` → `OcrPolygon.Bounds`.
- **Breaking** — **`WellKnownModels.Embedder` and `WellKnownModels.Reranker` are removed from LMSupply.Generator.** The
  Generator package cannot read the embedder and reranker registries, and its copies had drifted:
  `WellKnownModels.Embedder.Quality` and `.Large` named other models than the registry's `quality` and `large` aliases, so
  `LoadAsync(WellKnownModels.Embedder.Quality)` and `LoadAsync("quality")` loaded different models. Migration: use the
  registry aliases (`"default"`, `"fast"`, `"quality"`, `"large"`) or `LocalEmbedder.Registry` / `LocalReranker.Registry`.

- **Breaking** — **`LlamaServerVersionState` and `VersionEntry` record a build backend instead of a path.**
  `InstalledPath`, `PendingPath` and `VersionEntry.Path` are replaced by `InstalledBuildBackend`, `PendingBuildBackend`
  and `VersionEntry.BuildBackend` (null means the entry's `Backend`); the directory is
  `LlamaServerStateManager.GetBuildDirectory(version, buildBackend)`. `CreateInitialStateAsync` and
  `MarkUpdateReadyAsync` take the build backend in place of the directory. Migration: pass the backend directory name
  (e.g. `"cuda12"`, or `"cpu"` for a CPU fallback build) and resolve paths with `GetBuildDirectory`. The state file still
  carries the absolute paths, written for earlier releases that share the cache.

## [0.121.0] - 2026-10-14

### Changed
- **Breaking** — **A GGUF embedder loaded from a repository takes the largest quantization that fits in memory, as the
  guide says.** It always preferred `Q4_K_M` when that fit (bge-m3: Q4_K_M with FP16 in the repository and 7.9 GB of VRAM
  free), which costs embedding quality for little memory on models this size. An online load now fetches the larger
  file even when a Q4_K_M copy is cached (an offline load, `DisableAutoDownload`, keeps using what is cached), so its
  vectors and `VectorSpaceRevision` differ from a Q4_K_M load. Migration: to keep a specific quantization, load the
  `.gguf` file by its path.
- **Breaking** — **A GGUF embedder pools the way its file says and reads inputs up to the length it was trained to.**
  With no `PoolingMode` set it was always mean-pooled, and its context and batch were 512 tokens: bge-m3 (a CLS model)
  produced vectors with cosine 0.71–0.75 to the same model loaded as ONNX, and a passage over 512 tokens was refused with
  HTTP 500. The pooling now comes from the file's `pooling_type` and the context from its `context_length` (up to 8,192
  unless `MaxSequenceLength` says otherwise), with batch sizes that hold a whole input. bge-m3 GGUF FP16 now gives cosine
  1.0000 to the ONNX model and embeds a 3,479-character passage. `VectorSpaceRevision` changes for GGUF embedders that
  relied on the defaults — re-embed stored vectors.
- **Breaking** — **`PoolingMode.Max` is refused for GGUF embedders** (`NotSupportedException`); llama-server has no max
  pooling and it was sent as last-token pooling, a different vector space without a word.
- **Breaking** — **`GgufMetadata` and `GgufMetadataReader` moved to `LMSupply.Llama`** (namespace `LMSupply.Llama.Gguf`),
  shared by the generator and the GGUF embedder; `GgufMetadata.ToMemoryConfig` is now an extension in
  `LMSupply.Generator.Gguf`. Migration: `using LMSupply.Llama.Gguf;`. Property access through `GeneratorModelInfo.GgufMetadata`
  is unchanged.

### Added
- **`PoolingMode.Last`** — last-token pooling, read from a sentence-transformers `pooling_mode_lasttoken` file and from a
  GGUF `pooling_type` of 3. Decoder-based embedders (Qwen3-Embedding) were read as «unknown» and mean-pooled.
- **`GgufMetadata.PoolingType`** and **`GgufMetadata.ParameterCountOrEstimate`**.

### Fixed
- **ONNX sessions use CUDA only where the ONNX Runtime GPU build can run, and the warning says what it needs.** ONNX
  Runtime 1.27 and later (LMSupply uses 1.30) publish their GPU build for CUDA 13, but LMSupply checked for any installed
  CUDA, chose CUDA for any driver that reported CUDA 11 or newer, and told the user to install CUDA 12 — which cannot load
  the provider (it depends on `cublasLt64_13.dll`). The CUDA check now looks for the CUDA 13 runtime and cuDNN 9; a driver
  that supports only CUDA 12 keeps CUDA out of `Auto` and of the explicit-CUDA attempt, with a warning naming the driver's
  CUDA version and the one needed; the README and every guide state the same prerequisites (driver for CUDA 13, CUDA 13
  runtime, cuDNN 9; GGUF models need only the driver). Hosts that could run CUDA before still do.
- **A batch embedding call runs batched inference.** `EmbedAsync(IReadOnlyList<string>)` padded the whole batch to its
  longest text and then ran the model one text at a time, so every text paid the batch's padding and nothing ran
  together — measured on CPU with bge-m3, 64 passages per call embedded at 2.2 passages/s against 3.5/s one per call.
  Texts are now sorted by token count and run in groups of similar length as one `[batch, length]` inference each, a
  group holding at most 8,192 padded tokens (or one text of the model's maximum length); each text is pooled over its
  own tokens. Vectors are the same as embedding each text alone (cosine > 0.9999 over 40 texts of mixed length). The
  token embeddings are also copied out of the output tensor in one block instead of element by element.

### Documentation
- **The guides under `docs/` now compile against the current API, and are checked on every build.** They showed calls
  that do not exist: `LocalEmbedder.LoadFromPathAsync` (a path goes to `LocalEmbedder.LoadAsync`), `IReranker` and
  `embedder.CosineSimilarity` (`IRerankerModel`, `LocalEmbedder.CosineSimilarity`), `OnnxGeneratorModelFactory.CreateAsync`
  with a `Progress<double>` (`LoadAsync` with an `IProgress<DownloadProgress>`), `ModelInfo.ModelId` / `ModelPath` /
  `ActiveProviders` (the model's own `ModelId`, `ActiveProviders`, `RequestedProvider`), `GpuInfo.Name` / `VramBytes`
  (`DeviceName`, `HardwareProfile.GpuMemoryGB`) and `CaptionResult.ProcessingTimeMs`; and `using var` on a model, which is
  `IAsyncDisposable` only (`await using`). The transcriber and synthesizer interface listings show the real signatures.

## [0.120.0] - 2026-10-13

### Added
- **`DownloadPhase.Loading`: a load says when it starts bringing the model into memory.** Once the files are in place,
  a load reports `Loading` for the file — when llama-server is started to read the weights, and when an ONNX Runtime
  session is created — so a host can say «loading the model» for what is usually the longest part of a warm load. The
  report counts nothing and is the last report of the file; the load is over when `LoadAsync` returns. A GGUF load that
  shares a server already running for the model reports no `Loading`. An ONNX file is named by its path in the
  repository (`unet/model.onnx`), the name its download is reported under. `Loading` is a new member at the end of the
  enum, so the values of the others do not move; a `switch` over `DownloadPhase` that must be exhaustive needs the case.
- **`DownloadProgress.IsIndeterminate`** is true when a report measures nothing: a step with no total to be a share of
  (`Preparing`, `Loading`, a download whose size the server did not announce). `PercentComplete` and
  `OverallPercentComplete` are zero for such a report, and that zero means «not measured»; show the step as busy.
- **`IModelInfoBase.CatalogKey` names a model within its catalog, and every registry resolves it.** A repository can hold
  several models, and their catalog entries then share an `Id`. The key is the id when the id names one model; otherwise it
  is the id, `:` and what tells the models apart. For the synthesizer's Lessac voice it is
  `rhasspy/piper-voices:en/en_US/lessac/medium` (the repository and the voice's folder). `Resolve`, `TryResolve`,
  `TryResolveCatalog`, `RegisterAlias` and `LocalSynthesizer.LoadAsync` take a key, so a voice can be named by something
  other than its alias. An id that several models share resolves as before, to the first of them registered:
  `rhasspy/piper-voices` is the default voice. The console host's `GET /v1/models/{model}` finds a model by its key as
  well. `CatalogKey` is a default interface member returning `Id`, so implementations outside this library keep compiling.

### Changed
- **Breaking** — **A GGUF load no longer reports its load stage as `Extracting`, and progress reports carry no invented
  byte counts.** After the file's `Complete` report, `LocalGenerator`, `LocalEmbedder` and `LocalReranker` reported the
  model file as `Extracting` with `30/100` (the embedder also `80/100`), then as `Extracting` with `0/0`, then a second
  `Complete` with `100/100` — none of them a measurement, so a progress bar jumped to 30 %, fell to zero and completed
  a file twice. A load now reports `Loading` once, with no byte counts, and does not complete the file again. Also:
  a cached file's `Complete` carries the size of the file (it was `1/1`); a cached runtime's `Complete` carries no byte
  counts (it was `1/1`, or `100/100` for llama-server); `PercentComplete` is 100 for any `Complete` report; and the
  opening llama-server report of the embedder and the reranker is `Runtime` / `Preparing`, as the generator's already
  was (it was a `Model` / `Downloading` report with nothing moving). Migration: code that showed `Extracting` for a
  model file as the load stage reads `Loading`; code that skipped reports with no total to avoid a bar at zero tests
  `IsIndeterminate`; code that read `BytesDownloaded == TotalBytes == 1` as «cached» reads `Phase == Complete` with no
  `Downloading` report before it.
- **Breaking** — **`GetAliases()` reports the target of a system alias as the model's `CatalogKey`.**
  `AliasInfo.TargetModelId` was the model's `Id`. The two are equal unless several models share the id, which today is the
  synthesizer: its seven voice aliases all reported `rhasspy/piper-voices`, and each now reports its own voice
  (`rhasspy/piper-voices:en/en_US/lessac/medium` for `lessac`). Migration: code that matched an alias to a model with
  `TargetModelId == Id` compares with `CatalogKey`; the target can be passed to `Resolve` or `LoadAsync` as it is.

### Fixed
- **`LocalSynthesizer.GetAllModels()` returns the seven voices of the catalog; it returned one.** A registry listed one
  entry per `Id`, so that a model registered under several aliases is listed once. Every voice has the id of the repository
  they all live in, `rhasspy/piper-voices`, so the default voice stood for all seven. A registry now lists one entry per
  `CatalogKey`. `LocalSynthesizer.Registry.GetAvailableModels()`, the console host's registry listing and its
  `GET /v1/models` show the seven voices too; the other domains list what they listed before. Code that took the only
  element of the list as the default voice resolves `default` instead.
- **The console host answers 400 for a JSON body that does not parse.** Its error middleware answered 422 for every
  request the framework could not read, whatever status the framework gave it; `POST /v1/audio/speech`,
  `/v1/chat/completions`, `/v1/embeddings`, `/v1/rerank` and `/v1/translate` did so for malformed JSON. The response now
  carries the framework's status, the error code `invalid_request` (it was `unprocessable_entity`) and a message naming
  the parameter that could not be read.

### Dependencies
- **`LMSupply.Synthesizer` no longer references `NAudio`.** Nothing in the package used it. An application that used
  NAudio only because this package brought it in references `NAudio` itself.

## [0.119.0] - 2026-10-12

### Added
- **The synthesizer runs a voice by the contract its config declares.** A Piper voice's `.onnx.json` names the phonemes the
  voice was trained on and the id of each, its inference defaults, its sample rate and its speaker count. The package
  downloaded that file and encoded text with a letter table of its own instead. Phonemes are now encoded with the voice's
  `phoneme_id_map` in the layout the voice was trained on: the start symbol, a pad, every phoneme followed by a pad, the end
  symbol, after decomposing the phonemes (NFD). A phoneme the voice has no id for is skipped. From phonemes for «The weather
  is beautiful today. The quick brown fox jumps over the lazy dog.», the default voice's audio is transcribed by Whisper
  word for word.
- **`ISynthesizerModel.SynthesizePhonemesAsync` and `SynthesizePhonemesStreamingAsync` synthesize from phonemes.** They take
  one string per sentence, in which every Unicode code point is one phoneme (a space and the clause punctuation `, ; : . ? !`
  included). Each sentence is synthesized on its own; streaming yields one chunk per sentence.
  `ISynthesizerModel.PhonemeScheme` says which phonemes the voice takes: its `phoneme_type` (`espeak` or `text`), the
  phonemizer voice its config names (`en-us`, `en-gb-x-rp`, `ko`, ...) and its language.
- **`IPhonemizer` and `SynthesizerOptions.Phonemizer`: the text-to-phoneme step is a seam.** An implementation is given the
  text and the voice's `PhonemeScheme` and returns phonemes in the form `SynthesizePhonemesAsync` takes; with one set, the
  text methods speak through it. A voice whose phoneme type is `text` needs none: its phonemes are its characters,
  lower-cased and decomposed.
- **`SynthesisResult.Phonemes` and `UnknownPhonemes`, `AudioChunk.UnknownPhonemes`.** The result carries the phonemes that
  were synthesized (the phonemizer's output, for text) and the ones that were skipped because the voice has no id for them.

### Changed
- **Breaking** — **synthesizing text with a voice that needs a phonemizer throws `NotSupportedException`; it no longer
  returns audio.** Every voice in the catalog was trained on eSpeak NG phonemes, and the package has no text-to-phoneme step
  for them. `SynthesizeAsync`, `SynthesizeToFileAsync`, `SynthesizeToStreamAsync` and `SynthesizeStreamingAsync` used to
  return voice-like noise for any text; now they throw, and the message names the voice's phoneme type and language and the
  two ways on. Migration: set `SynthesizerOptions.Phonemizer` to an `IPhonemizer` for the voice's scheme, or pass phonemes
  to `SynthesizePhonemesAsync`. The package stays `[Experimental]` (`LMSUPPLY001`) until it ships a phonemizer; what the
  diagnostic means is now this, not unintelligible output.
- **Breaking** — **`SynthesizeOptions.NoiseScale` and `NoiseWidth` are `float?`, and unset means the voice's own default.**
  They defaulted to 0.667 and 0.8, which overrode what a voice declares for itself: the default voice declares a noise
  width of 0.333. `Speed` now divides the length scale the voice declares, and a speed that is not a positive, finite
  number is refused with `ArgumentOutOfRangeException`. Migration: code that reads the two properties handles `null`; set
  them to keep 0.667 and 0.8.
- **Breaking** — **`ISynthesizerModel` has three more members** (`PhonemeScheme`, `SynthesizePhonemesAsync`,
  `SynthesizePhonemesStreamingAsync`). Migration: an implementation of the interface outside the package adds them.
- **Breaking** — **a voice whose config is missing, or has no `phoneme_id_map`, fails to load.** The config was optional
  and a voice loaded without it. It is the voice's contract now, so a load without it throws `FileNotFoundException`, and a
  config that is not valid JSON, has no `phoneme_id_map`, or lacks the pad, start or end symbol throws
  `ModelLoadException`. Migration: keep the voice's `.onnx.json` next to its model.
- **`SynthesizeStreamingAsync` yields one chunk per sentence the phonemizer returns.** It used to split the text at `.`, `!`
  and `?` itself and drop that punctuation. `AudioChunk.IsFinal` is set on the last chunk only; it was also set on any
  earlier sentence equal to the last one.
- **The console host's `POST /v1/audio/speech` answers 501 with the reason** when the voice needs a phonemizer the host does
  not have, which is every catalog voice today. It answered 200 with noise.
- **Breaking** — **a GGUF file found in the cache is reported as `DownloadPhase.Complete`, never as `Downloading`.**
  `GgufModelDownloader` and `SingleFileGgufDownloader` reported a cache hit as a 1/1-byte download in the default
  `Downloading` phase. So a host showing progress printed «Downloading …: 100%» on every warm start, and a tracker of «did
  this load download» counted a download each time. A sharded load's per-shard report before the cache check is now
  `Preparing`. `DownloadProgress.Phase` documents the rule: `Downloading` means bytes are moving. Migration: a consumer that
  inferred a cache hit from a first report already at 100 % can read `Phase == Complete` instead.

### Fixed
- **A voice's sample rate is the one its config declares.** `audio.sample_rate` was never read (the key is snake_case and
  was matched against `SampleRate`), so every voice reported 22050 Hz. The `fast` voice produces 16000 Hz audio:
  `ISynthesizerModel.SampleRate`, `SynthesisResult.SampleRate` and the WAV header said 22050, and the file played fast and
  high. Where a catalog entry and the config disagree, the config is used and `GetModelInfo()` reports its values.
- **The `british` voice has four speakers and takes a speaker.** Its catalog entry said one, so the speaker input its model
  requires was never supplied and synthesis failed in ONNX Runtime. `SynthesizerModelInfo.NumSpeakers` is 4 for it, the
  speaker count is read from the voice's config, and a `SynthesizeOptions.SpeakerId` the voice does not have is refused
  with `ArgumentOutOfRangeException`. A single-speaker voice ignores the option.
- **The translator no longer puts an unknown token in front of every input.** `TranslatorTokenizer` created its SentencePiece
  tokenizer with the model's automatic `<s>` on, and a Marian `vocab.json` has no `<s>`, so every source sentence was encoded as
  `<unk>` followed by its pieces, even with `addSpecialTokens: false`. opus-mt-ko-en: `[1, 1114, …]`; the Marian reference is
  `[1114, …]`. The encoding now matches the reference id for id. Translations of three test sentences stayed the same or
  improved: «이 문서는 사용자 가이드입니다.» was «This document is a custom guide.» and is now «This document is your guide.».

## [0.118.0] - 2026-10-12

### Added
- **`RuntimePackageRegistry.PackageConfig.RequiresExactVersion`** marks a runtime package whose native library must be exactly
  the version its managed assembly was built against. A version the caller names is then never replaced by another cached
  copy or by a newer download. The ONNX Runtime GenAI packages carry it.

### Changed
- **`RankedResult`'s comparison operators are documented as rank comparisons.** `a < b` means `a` ranks first (its score is
  higher), because `CompareTo` sorts in ranked order. Behaviour is unchanged. To compare scores, compare `Score`.

### Fixed
- **`LocalTranslator` translates with a merged decoder export.** The default model (`onnx-community/opus-mt-ko-en`) ships
  only `decoder_model_merged*.onnx`. Every translation with it failed in ONNX Runtime with «Missing Input:
  past_key_values.N.encoder.value», because the decoder set `use_cache_branch` but not the past key/value inputs a merged
  graph also declares. Both greedy and beam-search decoding now supply them; the values are unused with caching off.
- **The console's Caption page shows the caption.** The host returns `choices[0].caption` (OpenAI-style), and the page read
  a top-level `caption` that the response does not have.
- **The ONNX generator's native GenAI runtime now matches the managed GenAI package.** It used to be downloaded at the newest
  version on nuget.org whatever managed version the application ran, so the two drifted apart whenever GenAI shipped a new
  minor. Now it is provisioned at the `Microsoft.ML.OnnxRuntimeGenAI` version the application resolved (read from its
  deps.json, or the version LMSupply was built against). With downloads off, only that version is accepted from the cache.
  A `RuntimeManagerOptions.PinnedVersion`, which names an ONNX Runtime version, no longer applies to the GenAI runtime.

## [0.117.1] - 2026-10-11

### Fixed
- **`LocalGenerator.LoadFromPathAsync` loads an ONNX model from a downloaded repository folder.** Given a folder whose GenAI
  model sits in a variant subfolder (`cpu-int4/`, or two levels down such as
  `cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4/`), it now finds that subfolder the way a load by model id does,
  preferring the requested provider's variant. Before, it handed the runtime the outer folder and failed with «Error opening
  …\genai_config.json». A folder with no model anywhere now gets a `FileNotFoundException` that says what was searched.

### Dependencies
- **`Microsoft.ML.OnnxRuntimeGenAI` 0.13.2 → 0.17.1** (LMSupply.Generator.Onnx). The ONNX generator integration suite (Phi-3.5
  and Phi-4 mini on CUDA and CPU: load, generate, stream, warm-up, provider selection, load from path) passes on it.

## [0.117.0] - 2026-10-11

### Breaking
- **Public types that nothing in the libraries reads are removed.** `ModelPrecision`, `ComputePrecision`, `OptimizationLevel`,
  `PrecisionConfig`, `ModelVariantResolver` (with `FilePatterns`, `ModelVariantInfo`), `OrtValueExtensions`,
  `InferenceBufferPool`, `OrtValueInput` (LMSupply.Core), `GeneratorBackendType` (LMSupply.Generator) and `LlamaBackend`
  (LMSupply.Llama). Model variants are chosen by `ModelPreferences` (`Quantization`), tensors are built with `TensorUtils`,
  and the backend enums in use are `ModelBackend` and `LlamaServerBackend`. Migration: use those, or copy the removed type
  if you depended on it.

### Changed
- **The nuget.org «Release Notes» of every package link to the GitHub release of its version**, whose notes are this CHANGELOG's section for that version.

## [0.116.0] - 2026-10-11

### Changed
- **The packages from this repository depend on each other at exactly the same version** (`[x.y.z]`), not a floor.
  A consumer that moves one of them while a family member it depends on resolves at an older version now gets restore warning NU1608 naming
  the pair (an error where warnings are errors) — before, the mixed versions restored silently and could fail at run time.
  A member that reaches you only through another package's floor is not covered: pin every member you load and move them together.

### Breaking
- **A cancelled `SpeculativeDecoder.GenerateAsync` throws `OperationCanceledException`.** A cancel that landed between
  two speculation rounds ended the token stream normally, so `await foreach` (and `GenerateCompleteAsync`) returned the
  partial text as a finished generation. Migration: catch `OperationCanceledException` where you cancel.
- **`TokenizationException` is removed.** Nothing threw it or named it. Migration: none expected (catch
  `LMSupplyException` for any LMSupply failure).
- **A runtime provider name with no package now fails instead of running on CPU.** `RuntimePackageRegistry.GetPackageConfig`
  returned the CPU package for any name it did not know, so a misspelt provider (`"cuda13"`) loaded the CPU runtime
  without a word and cached it a second time under the misspelt name. It now returns `null`, and
  `RuntimeManager.EnsureRuntimeAsync`, `OnnxNuGetDownloader.DownloadAsync` and
  `RuntimeManager.ResolveRuntimeForProviderAsync` throw `InvalidOperationException` naming the package type and
  provider. To migrate, pass a provider `RuntimePackageRegistry.GetSupportedProviders` lists (`cpu`, `cuda`, `cuda11`,
  `cuda12`; `openvino` for the plugin package type), or `"auto"`.
- `RuntimePackageRegistry.Providers.CoreML` is removed: CoreML has no package of its own. Its provider ships in the base
  ONNX Runtime package, which CoreML sessions now provision as `cpu`. A runtime cached under `runtimes/onnxruntime/coreml/`
  by an earlier version is no longer read; on macOS the base package is used from (or downloaded once into)
  `runtimes/onnxruntime/cpu/`, and the old directory can be deleted.
- **A cancelled `RuntimeUpdateService.CheckAndApplyUpdateAsync` (and `RuntimeManager.CheckAndApplyUpdateAsync`, which
  calls it) throws `OperationCanceledException` instead of returning `NoUpdateNeeded`.** Cancelling the call during the
  version check reported «no update needed» as if the check had answered. The `VersionCheckTimeout` and a failed version
  check still return `NoUpdateNeeded`. Migration: catch `OperationCanceledException` if you relied on the result.

- **Cancelling a model load throws `OperationCanceledException`.** Two loads wrapped the caller's cancellation in their
  own failure type. Old → new, when the caller cancels: a GGUF embedding model loaded through `LocalEmbedder` (the
  llama-server dimension probe) threw `InvalidOperationException` («Failed to determine embedding dimensions») →
  `OperationCanceledException`; a reranker's ONNX session creation threw `InferenceException` → `OperationCanceledException`.
  Other load failures keep their types. Migration: handle `OperationCanceledException` where you handle cancellation.

## [0.115.1] - 2026-10-09

### Fixed
- **An encoder-decoder model downloaded in a quantized form loads again with downloads disabled.** When the repository
  listing was no longer cached, the download manifest (which lists only the files that were fetched, e.g.
  `encoder_model_int8.onnx` and `decoder_model_merged_int8.onnx`) was not recognised as an encoder-decoder export, so a
  Whisper load with `DisableAutoDownload = true` looked for the full-precision `encoder_model.onnx` and threw
  `FileNotFoundException`, while `LocalTranscriber.IsModelDownloadedAsync` answered `true`. Both now pick the cached pair the
  first load downloaded. `ModelDiscoveryService.IsEncoderDecoderModel` uses the same rule as the file selection: a role
  prefix (`encoder_model`, `encoder`, `decoder_model_merged`, `decoder_with_past_model`, `decoder_model`, `decoder`) alone or
  followed by `_<suffix>`; a name that only ends in `encoder.onnx` (`vision_encoder.onnx`) no longer counts as the encoder.

## [0.115.0] - 2026-10-09

### Added
- **ONNX sessions can run on an Intel GPU: `ExecutionProvider.OpenVino`.** On a host whose only GPU is an Intel one,
  ONNX sessions (embedder, reranker, transcriber, OCR, vision, …) ran on CPU: CUDA and CoreML do not serve it, and
  DirectML left ONNX Runtime 1.25. Intel's OpenVINO execution provider now runs them on the GPU, on Windows x64 and
  Linux x64. Select it explicitly; `Auto` does not pick it.
  - It is an ONNX Runtime plugin (`Intel.ML.OnnxRuntime.EP.OpenVINO`). On first use LMSupply downloads it into the
    runtime cache — about 120 MB to download, about 200 MB on disk with the OpenVINO runtime it carries — and
    registers it with ONNX Runtime. Its version (1.7.0) is pinned by LMSupply, independently of the ONNX Runtime
    version and of `RuntimeManagerOptions.PinnedVersion`. Like the base runtime, the download is not counted by
    `GetDownloadSizeBytesAsync`. Consumers add no package.
  - The first load of a model compiles it for the GPU, which can take tens of seconds. OpenVINO's model cache keeps
    the compiled model next to the plugin in the runtime cache, so later loads of that model, in any process, skip
    it. Each cached model takes disk space of the order of the model's size.
  - The session uses the plugin's GPU device (`deviceId` selects one of several). With no Intel GPU, or one the
    plugin does not expose, the session runs on CPU and `ActiveProviders` lists only `CPUExecutionProvider`, as an
    explicit `Cuda` request does on a host without CUDA.
  - Text generation refuses it with `NotSupportedException`: neither ONNX Runtime GenAI nor llama-server has an
    OpenVINO path. llama-server keeps using Vulkan on Intel GPUs under `Auto`.
  - Measured on an Intel Iris Xe (Windows): the default embedder's vectors match the CPU session's to float rounding
    (cosine above 0.999). On Linux the plugin loads and registers; its GPU path has not been measured.
- `RuntimePackageRegistry.PackageConfig.PinnedVersion`, `RuntimePackageRegistry.PackageTypes.ExecutionProviderPlugin`
  and `RuntimeManager.EnsureExecutionProviderPluginAsync`: a runtime package can carry its own pinned version, and an
  execution provider plugin is provisioned next to the base runtime and returned as a library path.

## [0.114.0] - 2026-10-08

### Added
- **A retrieval index can key on a revision that only the retrieval paths move.** `IEmbeddingModel.RetrievalVectorSpaceRevision`
  (and `LocalEmbedder.GetRetrievalVectorSpaceRevisionAsync` before the load) covers `EmbedQueryAsync` and `EmbedPassageAsync`
  only — the prefix `EmbedAsync` applies is not part of it. `VectorSpaceRevision` still covers every path. After 0.112.0
  gave E5 and Nomic models a default prefix, `VectorSpaceRevision` moved for them although their passage and query vectors
  did not; their retrieval revision is the value `VectorSpaceRevision` had on 0.111.0, so an index built then and keyed on
  it does not re-embed. For a model without a default prefix the two values are equal. `RetrievalVectorSpaceRevision` is a
  default interface member returning `null`, so implementations outside this library keep compiling.

## [0.113.0] - 2026-10-08

### Security
- **`LMSupply.Vision.Core` no longer depends on SixLabors.ImageSharp.** Version 3.1.12 carries advisories
  GHSA-j3p4-wp97-rph4, GHSA-j9gm-c75j-xc9q and GHSA-jjfr-hcj7-qf5w (high) and GHSA-gwg2-r3hj-4w44 and
  GHSA-wmxv-xphr-5c9g (moderate), so NuGet audit failed restore for every project that references the vision packages
  with warnings as errors. The only patched line (4.x) requires a license key at build time. Decoding and PNG encoding
  now use SkiaSharp (MIT); resizing is implemented in LMSupply.

### Breaking
- **`IImageLoader.LoadAsync` returns `RgbImage` and `IImagePreprocessor.Preprocess` takes `RgbImage`**, replacing
  ImageSharp's `Image<Rgb24>`. `RgbImage` holds 8-bit RGB pixels (`Pixels`, `GetRow`, an `[x, y]` indexer) and offers
  `Resize`, `Crop`, `Letterbox` and `Clone`. Code that loaded images with ImageSharp itself can pass the encoded bytes
  to `ImageLoader.Decode` instead.
- **Accepted input formats change.** JPEG, PNG, WebP, GIF (first frame), BMP and ICO decode.
  **TIFF, TGA, PBM and QOI no longer decode.** Data that is not
  a supported image, that is truncated, or whose header declares more pixels than can be held in memory (about 536
  million) now throws `InvalidDataException`, before any pixel buffer is sized from it (it used to throw ImageSharp's
  `UnknownImageFormatException` or `InvalidImageContentException`). To migrate, convert those files to PNG before
  passing them in.

### Changed
- **Resize output matches the previous implementation to within one level per channel.** The bicubic (Catmull-Rom)
  kernel widens with the scale factor when shrinking, as before. Fixture tests compare it against reference output from
  the previous implementation.
- **JPEG decoding differs slightly from before.** Two decoders do not produce identical pixels from the same JPEG:
  chroma-subsampled (4:2:0) files differ by about 0.5 levels on average (at most 6); 4:4:4 files differ by at most 3.
- **`ResizeMode.Fit` centers the letterboxed image with the offset rounded down on both axes.** It used to round the
  vertical offset half-to-even, so content in a custom `Fit` profile can move by one pixel. No built-in profile uses
  `Fit`.
- **Native libraries come with the package.** Windows and macOS natives come with SkiaSharp. Linux (glibc and musl,
  x64 and arm64) natives come with `SkiaSharp.NativeAssets.Linux.NoDependencies`, which needs no fontconfig.
  **Consumers add nothing per platform.** An application that already references `SkiaSharp.NativeAssets.Linux` (the
  fontconfig build) now gets both Linux native packages, which ship the same `libSkiaSharp.so`, so `dotnet publish`
  reports a duplicate output. Keep the one you reference and exclude the other with `ExcludeAssets="all"`.

### Added
- `ImageLoader.Decode(ReadOnlySpan<byte>)`, `ImageLoader.Identify` (dimensions without decoding) and
  `ImageLoader.EncodePng(RgbImage)`.

## [0.112.1] - 2026-10-08

### Fixed
- **`GetRemainingDownloadBytesAsync` answers 0 for a fully cached model loaded from a subfolder.** A tokenizer or config
  file the repository has only at its root (e.g. `sentence_bert_config.json` beside `onnx/model.onnx` in the E5 family)
  is stored beside the subfolder's model, where the load reads it; the remaining-size check looked for it at the
  repository root and counted it as still to download. `PlannedFile.CachePath` says where the download stores each
  file, and the check reads it there.

## [0.112.0] - 2026-10-08

### Breaking
- **`OnnxSessionFactory.CreateAsync`/`CreateWithInfoAsync` take `configureOptions` and `progress` as required
  parameters** (pass `null` for either). Session creation can download the native runtime on first use; a reporter the
  caller holds was easy to drop by omission, and most domains did. `provider` is required too.
- **`LocalImageGenerator.LoadAsync` takes `IProgress<DownloadProgress>`** instead of `IProgress<float>`, like every
  other load: file names, phases, byte counts and the runtime step instead of one fraction that restarted per file.
- **`IGeneratorModelFactory.DownloadModelAsync` takes `IProgress<DownloadProgress>`; `ModelDownloadProgress` is
  removed.** Its adapter in the ONNX generator factory swapped the total and downloaded byte counts.
- **`EmbedAsync` applies the model's prefix for text with no retrieval role** — `query: ` for the E5 family (its model
  card prescribes it for every task other than retrieval, and warns of degraded results without a prefix),
  `clustering: ` for nomic-embed-text-v1.5 (whose card requires a task prefix). It used to embed the bare text: a
  vector these models were never trained to produce, matching neither the query nor the passage side. Models without
  a prefix convention (BGE, GTE, MPNet) are unchanged. The new `ModelInfo.DefaultPrefix` carries it; a model loaded by
  repository id reads it from `config_sentence_transformers.json`'s `default_prompt_name`, as sentence-transformers
  does. `VectorSpaceRevision` changes for the models affected, so stored vectors are re-embedded; revisions of the
  others stay as they were.
- **`EmbedRawAsync`** (four overloads) embeds the text exactly as given, for text that already carries the
  instruction the model expects. It replaces `EmbedAsync` as the member an `IEmbeddingModel` implementation provides;
  `EmbedAsync`, `EmbedQueryAsync` and `EmbedPassageAsync` apply the model's conventions over it.
- **`EmbeddingTextKind.Unspecified` is now `Default`** (the model's default prefix, and still the generator's default);
  **`EmbeddingTextKind.Raw`** is the bare path.
- **`LocalGenerator.LoadFromPathAsync(path, options, progress, cancellationToken)`** — `progress` comes before the
  token, as in `LoadAsync`. **`LocalSegmenter.LoadInteractiveAsync(id, options, progress, cancellationToken)`** — the
  same.

### Added
- **`TextGeneratorBuilder.WithDownloadProgress`**: the builder reports the model and runtime downloads `BuildAsync`
  makes, as `LocalGenerator.LoadAsync` does. Its ONNX path now fetches the model before the runtime, the order the
  load by id keeps.
- **`DownloadProgress.Kind`** (`DownloadKind.Model` / `DownloadKind.Runtime`): a load reports the model download and
  a first-run runtime download (the native ONNX Runtime, llama-server) through one callback, and a caller can now tell
  them apart.
- **`GetRemainingDownloadBytesAsync`** beside every `GetDownloadSizeBytesAsync` (Embedder, Reranker, Generator,
  Captioner, OCR, Transcriber): what a load with the same id and options would still download now — the files the cache
  does not hold at the length the repository lists. `GetDownloadSizeBytesAsync` stays the whole first-run download, so
  a consent screen asked again for a model already on disk; this answers 0 once it is cached, and the missing part of a
  partly cached model. `DownloadPlan.GetRemainingBytes(cacheDir)` is the shared rule, including GGUF files in the tree
  earlier versions wrote.

### Fixed
- **The Hugging Face hub root holds only the hub layout.** The repository file listing LMSupply caches moved from
  `.discovery-cache/` at the hub root — which `hf cache scan` reported as an invalid cache directory — to the
  repository's own `models--{org}--{name}/.lmsupply/listings/`, beside the download manifests, where Hugging Face tools
  do not look and which they delete with the repository. The old directory is removed when found.
- **Loads that start together on a cold cache download the native ONNX Runtime once** — an image generator creating its
  sessions in parallel, or two models loaded with `Task.WhenAll`. Each used to fetch its own copy, and the last to
  finish deleted the runtime directory the first had already loaded to put its copy in place (the load failed on
  Windows). A runtime already in the cache is never replaced by a second copy of itself, and the package is staged
  beside the cache instead of in the system temp directory, so a cache on another drive works. llama-server builds
  get the same treatment: concurrent first runs no longer extract over each other, and the executable appears only
  once the rest of the build is in place.
- **`EmbedQueryAsync` on bge-base-en-v1.5 and bge-large-en-v1.5 applies the retrieval instruction their model card
  recommends** (`Represent this sentence for searching relevant passages: `); passages and other text stay bare, as the
  card says. Queries were embedded without it. A GGUF repository of either model (`…-GGUF`) takes it too. Their
  `VectorSpaceRevision` changes.
- **The runtime step of a first load reaches the load's progress in every domain.** Captioner, Detector, Embedder,
  ImageGenerator, OCR, Reranker, Segmenter (both loads), Synthesizer, Transcriber (including speaker diarization) and
  Translator created their ONNX sessions without the caller's progress, so a first-run runtime download (tens of
  seconds) showed nothing. `LocalReranker.LoadAsync` did not report its model download either.
- **Generator loads from a local path pass progress and cancellation on.** `LoadFromPathAsync` and `LoadAsync` with a
  file or directory path provisioned llama-server or the GenAI runtime with no progress and `CancellationToken.None`.
  The opening llama-server report is now a `Runtime` report in the `Preparing` phase.
- **Runtime download progress names what it fetches and the phase it is in.** The ONNX Runtime package download put
  status text in `FileName` ("Extracting native libraries...", "Using cached runtime (already downloaded)") and left
  every report in the `Downloading` phase. Reports now name the package, carry `Preparing`, `Extracting` and
  `Complete`, and a cached runtime is one `Complete` report naming it.

## [0.111.0] - 2026-10-07

### Removed
- **Breaking** — `ExecutionProvider.DirectML`, obsolete since 0.67.0 (ONNX Runtime 1.25+ has no DirectML provider). Its
  value, 2, is never reused: a setting that still holds it, by number or as the provider name `"directml"`, is refused
  with the same explanation as before. Migration: use `ExecutionProvider.Auto` (or `Cpu`).

### Changed
- **Breaking** — The GGUF types the documentation presents as API moved from `LMSupply.Generator.Internal.Llama` to
  `LMSupply.Generator.Gguf`: `GgufModelRegistry`, `GgufModelInfo`, `GgufModelKnownIssues`, `GgufModelDownloader`,
  `GgufFileInfo`, `GgufMetadata`, `GgufMetadataReader`, `ModelSelectionResult`, `ModelSelectionCandidate`,
  `ModelSelectionReason`. `Internal` now holds only internal types. Migration: replace
  `using LMSupply.Generator.Internal.Llama;` with `using LMSupply.Generator.Gguf;`.
- The ViT-GPT2 captioner reports `SupportsVqa = false` and answers a question with `NotSupportedException`, as the
  architecture has no question input; it no longer carries an unreachable `NotImplementedException` path.
- **Packages ship debug symbols.** The build produced symbol packages but the publish workflow discarded them, and the
  libraries do not embed their PDBs, so a consumer could not step into LMSupply or get line numbers in its stack
  traces. The `.snupkg` files now go to the nuget.org symbol server with each package; SourceLink resolves the sources.
- **CI runs for every change that can affect a build or a package.** The workflow listed the folders it watched, so a
  change only to `samples/`, `benchmark/`, `tools/`, `global.json` or `nuget.config` started no CI run, and the
  publish workflow, which follows a successful CI run, could ship code nothing had built. It now lists what cannot
  affect a build (Markdown, `docs/`, `images/`, licence and git files) instead. `tools/verify-auto` and
  `samples/RuntimeDownloadTest` are in the solution, so CI builds them.
- Memory-based `auto` selection is one implementation for every domain that uses it (embedder, generator, detector,
  segmenter, synthesizer, transcriber): `ModelRegistryBase.SelectLargestFitting`. A candidate without size metadata is
  now an error naming it, instead of an estimate of zero that wins on every host.

### Fixed
- **`LocalEmbedder.LoadAsync("auto")` follows the hardware.** The embedder catalog carried no model sizes, every
  candidate estimated to 0 bytes and fit any budget, so `auto` took BAAI/bge-m3 (about 2.2 GB) everywhere, including a
  CPU-only host. Every built-in embedder now carries its ONNX size and parameter count; on a host where nothing fits
  the VRAM budget `auto` takes the smallest candidate (multilingual-e5-small). **Breaking** for an application that loads `auto` and
  keeps the vectors: on such a host the model, and with it the vector size (1024 → 384), changes; re-embed the stored
  vectors, or load a named model (`default`) to keep the previous one.
- **A loaded embedder's `EstimatedMemoryBytes` counts the external weight file** (`model.onnx_data`) that large ONNX
  models keep beside a graph file of a few hundred kilobytes.

### Documentation
- **The package READMEs no longer tell readers to install ONNX Runtime packages.** Eight package pages on nuget.org
  said to add `Microsoft.ML.OnnxRuntime.DirectML` (removed in 0.67.0), five guides said to add
  `Microsoft.ML.OnnxRuntime.CoreML` (no such package), and all of them said to add `Microsoft.ML.OnnxRuntime.Gpu`,
  while LMSupply provisions the runtime itself and a second copy conflicts with it. They now say so and point to the
  GPU acceleration section. A test fails when a document tells a reader to add a package that is not one of
  LMSupply's own.
- The model lifecycle and troubleshooting guides name the cache LMSupply actually uses (the Hugging Face cache), and
  the troubleshooting fix deletes one model's entry instead of the whole folder, which other tools share. The
  `CacheManager` snippet in the README has its `using`.

## [0.110.1] - 2026-10-07

### Dependencies
- `ChildProcessGuard` 1.1.1 -> 1.4.0 (llama-server process management).
- The explicit `SQLitePCLRaw.bundle_e_sqlite3` 3.0.5 lift is removed: Entity Framework Core Sqlite 10.0.12 already
  requires 2.1.12, which carries the fix the lift existed for. The console host resolves 2.1.12.
- Test and host tooling: `xunit.v3` 4.0.1, `Microsoft.Testing.Extensions.HangDump` 2.5.0, `Swashbuckle.AspNetCore` 10.2.3.
- Held versions now say why and when they move: `Microsoft.ML.OnnxRuntimeGenAI` 0.13.2 (0.x API changes; moves after the
  integration and local-model suites pass), `NAudio` 3.0.1 (locked by `NLayer.NAudioSupport`), `Microsoft.OpenApi` 2.x.

## [0.110.0] - 2026-10-07

### Fixed
- **A model already in the cache loads without a network, with downloads enabled.** A load that names a file list
  (every catalog alias, which uses the default list) asked the hub for the optional files the repository does not
  have (`model.onnx_data`, `vocab.txt`, `merges.txt`, ...) on every load, and with no network that request failed the
  load with `HttpRequestException` although every file the model needs was cached. The download manifest now records
  the files the repository lacks (`DownloadManifest.AbsentFiles`), so a warm load makes no request at all; a manifest
  written by an earlier version gets the record on the next online load.
- **An unreachable hub falls back to the cache.** When a request fails without an HTTP status (connection refused,
  name resolution, proxy) or times out, an optional file is skipped while the model's required files are cached, and
  `ModelDiscoveryService.ListRepositoryFilesAsync` (so a load by repository id) uses the file list the cache already
  holds, at any age. A required file that is not cached still fails, now as `ModelDownloadException` naming the file
  and the next step (load once online, or load from the cache with downloads disabled), with the network error as
  `InnerException`; it was the bare `HttpRequestException`. Status answers (404, 403, 5xx) are handled as before.
  **Breaking** — the exception type of that case changed. Migration: catch `ModelDownloadException` (the network error
  is its `InnerException`) where `HttpRequestException` was caught around a load.
- **A load whose repository listing failed no longer marks the manifest as verified** against lengths it never
  received.

## [0.109.1] - 2026-10-07

### Fixed
- **llama-server is found on Linux with a CUDA backend.** The archive pattern was not anchored, so the CUDA runtime
  companion (`cudart-llama-b<n>-bin-ubuntu-cuda-12.8-x64.tar.gz`), which the release lists first, was taken as the server
  archive and the load failed with "llama-server executable not found". Every server pattern is now anchored, and the
  runtime companion is also found under its Linux name, which carries the build number.
- **AMD and Intel backends resolve against current llama.cpp releases.** `Hip` matches the `rocm-<version>` archives
  (and the older `hip-radeon` name); `Sycl` on Linux takes the `sycl-fp32` build, llama.cpp's default SYCL configuration.
  Before, both found nothing and fell back to CPU.

## [0.109.0] - 2026-10-07

### Changed
- **A GGUF model loaded without `MaxContextLength` gets a context it can use.** llama-server is started with the model's
  trained length (GGUF `context_length`) up to 32,768, bounded by the memory that holds the KV cache: VRAM on a GPU
  backend (as before for explicit lengths), half of system RAM after the weights on the CPU backend, never below 4,096.
  Before, every such load ran at 4,096 tokens whatever the model and the machine, so a tool loop reading a few web pages
  hit the server's limit after two to four rounds. An explicit `MaxContextLength` is unchanged. Model auto-selection and
  the quantization choice still size for 4,096 when the length is unset.
- **Breaking: `IGeneratorModel.MaxContextLength` is the context the loaded model accepts.** For a GGUF model it is the
  context llama-server was started with (it was the GGUF trained length, or the explicit option even when memory cut the
  server below it). Request fit checks, generation's own turn trimming and hosts that size their context from this value
  now work against what the server holds. Migration: read the trained length from
  `GetModelInfo().GgufMetadata?.ContextLength`. `AdjustedContextLength` is set when memory reduced the context below the
  requested length (explicit or default).

### Fixed
- **A request the server refuses as too long says by how much.** `ContextLengthExceededException.TokenCount` and
  `MaxContextLength` come from llama-server's refusal (`n_prompt_tokens`, `n_ctx`) when it states them; before, the token
  count was always null.

## [0.108.0] - 2026-10-06

### Added
- **The transcriber reads what a browser records.** WebM/Opus (Chromium, Edge and WebView2 `MediaRecorder`, including
  live recordings whose sizes are left unknown) and Ogg/Opus (Firefox) decode on the device from a file path, a `Stream`
  or a `byte[]`, recognised from their first bytes; stereo is mixed to mono. Decoding is managed code only (new
  dependency: Concentus, the C# port of the Opus reference decoder). MP4/AAC (Safari) throws `NotSupportedException`
  naming the formats that are supported, instead of failing inside the WAV reader.

## [0.107.0] - 2026-10-06

### Removed
- **Breaking: public types that nothing used are removed.** No code path in this library constructed, returned or accepted them, and no implementation existed where they were interfaces. Code that never named them is unaffected; code that did can delete the reference - there was no behaviour behind it.
  Removed: `IModelRuntimeInfo` (no model info type implemented it) and `GgufQuantization`.

## [0.106.1] - 2026-10-06

### Fixed
- **Breaking** (released as a patch) — **cancelling a call now cancels it.** 19 method(s) that take a `CancellationToken` caught every exception to
  return a fallback (`null`, an empty result, a failure value) or to log and continue, and treated the caller's own
  cancellation the same way. They now let the caller's `OperationCanceledException` through; other failures behave
  as before. Affected: model and runtime downloads, update checks, GGUF metadata reading, the llama-server client/process, OCR and image generation.
  Migration: code that relied on a cancelled call returning `null`, an empty result or a failure value now
  receives `OperationCanceledException` — catch it where a cancellation is expected.

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
