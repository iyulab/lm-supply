# LMSupply.Decider

Typed decisions from a local model. Give a state (the content to evaluate) and named questions — a choice among options you
supply, yes/no, or a score on your own scale — and get a typed answer with a probability for every possible answer.

```bash
dotnet add package LMSupply.Decider
```

## Basic usage

```csharp
using LMSupply.Decider;

await using var decider = await LocalDecider.LoadAsync("ggml-org/gemma-4-E4B-it-GGUF");

var result = await decider.DecideAsync(
    state: "Customer message: I was charged twice for my order last week.",
    questions: new Dictionary<string, DecisionQuestion>
    {
        ["route"] = DecisionQuestion.Choice("Which team should handle this?", ["billing", "shipping", "technical"], allowNone: true),
        ["angry"] = DecisionQuestion.YesNo("Is the customer angry?"),
        ["urgency"] = DecisionQuestion.Score("How urgent is this?", ["can wait", "this week", "today", "right now"]),
    });

var route = result["route"];
Console.WriteLine(route.Choice);                 // "billing", or null when "none of these" is the most probable answer
Console.WriteLine(route.Probabilities["billing"]);
Console.WriteLine(result["angry"].Probability);  // probability of yes
Console.WriteLine(result["urgency"].Score);      // expected level; 0 is the first level and the value can fall between levels
```

The questions are independent of each other and of their order. State, questions and options are best written in the same
language; any language the model reads works.

## Question types

| Question | Factory | Answer |
|---|---|---|
| Choice | `DecisionQuestion.Choice(instructions, options, descriptions?, allowNone)` | `Choice` (the most probable option; `null` when `allowNone` and none of them is the most probable answer), `Probabilities` per option, `NoneProbability` |
| Yes/no | `DecisionQuestion.YesNo(instructions, trueMeans?, falseMeans?)` | `IsTrue`, `Probability` (of yes) |
| Score | `DecisionQuestion.Score(instructions, levels)` — 2 to 10 levels, lowest first | `Score` (expected level, Σ i·p(i)), `Level` (most probable level), `Probabilities` per level |

Every `Decision` also carries `Confidence` (the largest probability, «none of these» included), `Path`, `LabelMass` and
`Calibrated` (below). Reading a member that belongs to another question type throws `InvalidOperationException`.

`IDeciderModel.Capabilities` says what the loaded model can answer (`MaxChoices`, `SupportsNone`, `SupportsYesNo`,
`SupportsScore`). A choice takes at most 25 options on an instruct model.

## Two ways a model answers

`LocalDecider.LoadAsync` starts llama-server on a GGUF file (a local path, or a repository id that is downloaded and cached
like every other LMSupply GGUF model) and asks the server what the model produces:

- **Native** (`DecisionPath.Native`) — the model is a decision model that llama-server serves through its decision
  endpoint. All questions of a call go in one request and the server returns the probabilities. `LabelMass` is 1.
- **Emulated** (`DecisionPath.Emulated`) — any other instruct model. Each question becomes one chat request that lists the
  answers as labelled lines (`1.`, `2.`, … with `0.` for «none of these»; `Yes`/`No` for yes/no questions; `0`…`9` for score
  levels), with thinking off and a single generated token. The probabilities come from that token's log probabilities: the
  candidates that are answer labels are summed per label and renormalized over the labels. Questions of one call are sent in
  order, one at a time, with the state first, so the server reuses the state's evaluation for the later questions.

`Decision.LabelMass` is the share of the model's next-token probability that fell on answer labels before renormalizing. Near
1 means the model wanted to answer with a label; a low value means it mostly wanted to say something else and the
renormalized numbers rest on little. A model that formats its answer (it would write `**Yes**`) leaves part of the mass on the
formatting token; the renormalized numbers are then still its relative preference between the labels, and a low `LabelMass`
only says how much of its next-token probability they cover. If none of the candidate tokens the request asks for (the number of labels plus 10) is an answer label,
`DecisionException` reports the token the model started with (for example reasoning markup of a model run with thinking on).

Each answer label must be a single token of the model's tokenizer. The loader checks this once; a question that needs a label
the tokenizer splits throws `NotSupportedException` naming the label and its token count.

## Calibration

The probabilities are the model's own and are not guaranteed to match how often the model is right. `DeciderOptions.Temperature`
applies a temperature per question kind, p ∝ p^(1/T) renormalized — T above 1 flattens, below 1 sharpens — and then
`Decision.Calibrated` is `true`. LMSupply only applies the value; fitting it to labelled data is up to you.

```csharp
var options = new DeciderOptions
{
    Temperature = new Dictionary<DecisionKind, double> { [DecisionKind.Choice] = 1.4 },
};
```

## Options

| `DeciderOptions` | Meaning |
|---|---|
| `Temperature` | Temperature per `DecisionKind` (above) |
| `ContextSize` | Longest state plus question, in tokens (default 8192, or the model's own when shorter) |
| `DisableAutoDownload` | Model file and llama-server come from the cache only; nothing is requested |
| `Provider`, `QuantizationHint`, `CacheDirectory` | As for the other LMSupply tasks (`QuantizationHint` defaults to `Q4_K_M` for a repository) |
| `ServerUpdateOptions` | llama-server version pinning, a pre-provisioned binary, update behavior |

A native decision model needs a llama-server build that serves the decision endpoint; the loader updates a cached older
build once (or reports it when downloads are disabled). For an instruct model, a build with log probabilities is enough, and
a recent one is recommended: it reuses the state's evaluation across the questions of a call for models with sliding-window
or recurrent layers too.

### Before a download

A host that asks before downloading reads what a load would fetch, without loading or downloading anything:

```csharp
var artifacts = new DeciderArtifacts(new DeciderOptions { Provider = ExecutionProvider.Cpu });
if (!await artifacts.IsAvailableOfflineAsync("org/model-GGUF"))
{
    var remaining = await artifacts.GetRemainingDownloadAsync("org/model-GGUF");   // ModelBytes, RuntimeBytes, TotalBytes
    // ... after the user agrees:
    await artifacts.DownloadAsync("org/model-GGUF", progress);                      // the GGUF file and llama-server
}
```

`LocalDecider.GetDownloadSizeBytesAsync`, `GetRemainingDownloadBytesAsync`, `IsModelDownloaded`, `DownloadModelAsync` and
`DeleteModel` answer for the GGUF file alone: the quantization a load picks, at the length the repository lists.
`RuntimeBytes` is the llama-server build a load would run.

## Choosing a model

Any instruct GGUF that llama-server runs can be used. Smaller models read clear-cut questions well and fall apart on
questions that need a calculation. As a measurement on 20 short choice questions (English and Korean; sarcasm, negation, many
options, no option applies), a 4B-parameter model chose the intended option on 18 and a 4B-effective-parameter model on
19–20, with at least 96 % and 89 % of the next-token probability on answer labels respectively. The misses were the two
arithmetic questions.

## Not for

- **Decisions that need arithmetic or several steps of reasoning.** A decision is read from one generated token, so the
  model cannot work anything out first. Use generation for those.
- Fitting or evaluating calibration against your data.
- Questions whose answers depend on each other: each question is answered on its own.
