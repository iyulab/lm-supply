# LMSupply.Decider

Typed decisions from a local model: give a state and named questions — a choice among your options, yes/no, or a
score on your scale — and get a typed answer with a probability for every possible answer.

```csharp
using LMSupply.Decider;

await using var decider = await LocalDecider.LoadAsync("org/model-GGUF");

var result = await decider.DecideAsync(
    state: "Customer message: I was charged twice for my order last week.",
    questions: new Dictionary<string, DecisionQuestion>
    {
        ["route"] = DecisionQuestion.Choice("Which team should handle this?", ["billing", "shipping", "technical"], allowNone: true),
        ["angry"] = DecisionQuestion.YesNo("Is the customer angry?"),
        ["urgency"] = DecisionQuestion.Score("How urgent is this?", ["can wait", "this week", "today", "right now"]),
    });

Console.WriteLine(result["route"].Choice);        // "billing", or null for none of these
Console.WriteLine(result["angry"].Probability);   // probability of yes
Console.WriteLine(result["urgency"].Score);       // expected level, 0 = first
```

Native decision models run on llama-server's decision endpoint; instruct models are read from their next-token
probabilities. Decisions that need arithmetic or several steps of reasoning are out of scope — use generation.
See the [Decider guide](https://github.com/iyulab/lm-supply/blob/main/docs/decider.md).
