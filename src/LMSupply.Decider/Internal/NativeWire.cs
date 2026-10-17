using System.Text.Json.Nodes;

namespace LMSupply.Decider.Internal;

/// <summary>
/// The request and response of llama-server's <c>/v1/systemone</c> (TypeSafe-compatible): <c>choice</c>,
/// <c>score</c> and <c>noul</c> questions. The wire has no «none of these»; a choice that allows it is sent with one
/// more option, which the response then reports apart from the caller's options.
/// </summary>
internal static class NativeWire
{
    /// <summary>The option sent for «none of these», unless the caller already has an option of that name.</summary>
    public const string NoneOption = "none of these";

    /// <summary>The «none of these» option for <paramref name="question"/>: a name none of its options has.</summary>
    public static string NoneOptionFor(ChoiceQuestion question)
    {
        var name = NoneOption;
        for (var n = 2; question.Options.Contains(name, StringComparer.Ordinal); n++)
            name = $"{NoneOption} ({n})";
        return name;
    }

    /// <summary>The request body for <paramref name="questions"/> about <paramref name="state"/>, in their order.</summary>
    public static JsonObject BuildRequest(string state, IEnumerable<KeyValuePair<string, DecisionQuestion>> questions)
    {
        var wireQuestions = new JsonObject();
        foreach (var (name, question) in questions)
            wireQuestions[name] = BuildQuestion(question);

        return new JsonObject { ["state"] = state, ["questions"] = wireQuestions };
    }

    private static JsonObject BuildQuestion(DecisionQuestion question)
    {
        switch (question)
        {
            case ChoiceQuestion c:
            {
                var criteria = new JsonObject();
                foreach (var option in c.Options)
                    criteria[option] = c.DescriptionOf(option) is { } d ? JsonValue.Create(d) : null;
                if (c.AllowNone)
                    criteria[NoneOptionFor(c)] = null;
                return new JsonObject { ["type"] = "choice", ["instructions"] = c.Instructions, ["criteria"] = criteria };
            }

            case ScoreQuestion s:
                return new JsonObject
                {
                    ["type"] = "score",
                    ["instructions"] = s.Instructions,
                    ["criteria"] = new JsonArray(s.Levels.Select(l => (JsonNode?)JsonValue.Create(l)).ToArray()),
                };

            case YesNoQuestion y:
            {
                var wire = new JsonObject { ["type"] = "noul", ["instructions"] = y.Instructions };
                if (y.TrueMeans is not null || y.FalseMeans is not null)
                {
                    var criteria = new JsonObject();
                    if (y.TrueMeans is not null)
                        criteria["true"] = y.TrueMeans;
                    if (y.FalseMeans is not null)
                        criteria["false"] = y.FalseMeans;
                    wire["criteria"] = criteria;
                }
                return wire;
            }

            default:
                throw new ArgumentOutOfRangeException(nameof(question));
        }
    }

    /// <summary>
    /// The probability of each answer slot of <paramref name="question"/> from the server's answer to it, normalized
    /// (the server's already sum to 1; normalizing absorbs its rounding).
    /// </summary>
    /// <exception cref="DecisionException">The answer is missing or lacks a probability the question needs.</exception>
    public static double[] ReadAnswer(string name, DecisionQuestion question, JsonObject response)
    {
        if (response["answers"]?[name] is not JsonObject answer)
            throw new DecisionException($"The server's response has no answer for question '{name}'.", name);

        switch (question)
        {
            case ChoiceQuestion c:
            {
                var probabilities = answer["probabilities"] as JsonObject;
                var slots = new double[DecisionMath.SlotCount(c)];
                for (var i = 0; i < c.Options.Count; i++)
                    slots[i] = Probability(probabilities, c.Options[i], name);
                if (c.AllowNone)
                    slots[c.Options.Count] = Probability(probabilities, NoneOptionFor(c), name);
                return DecisionMath.Normalize(slots);
            }

            case ScoreQuestion s:
            {
                var probabilities = answer["probabilities"] as JsonObject;
                var slots = new double[s.Levels.Count];
                for (var i = 0; i < slots.Length; i++)
                    slots[i] = Probability(probabilities, i.ToString(System.Globalization.CultureInfo.InvariantCulture), name);
                return DecisionMath.Normalize(slots);
            }

            case YesNoQuestion:
            {
                if (answer["noul"] is not JsonValue value || !value.TryGetValue<double>(out var yes))
                    throw new DecisionException($"The server's answer to question '{name}' has no 'noul' probability.", name);
                yes = Math.Clamp(yes, 0, 1);
                return [yes, 1 - yes];
            }

            default:
                throw new ArgumentOutOfRangeException(nameof(question));
        }
    }

    private static double Probability(JsonObject? probabilities, string key, string name)
    {
        if (probabilities?[key] is JsonValue value && value.TryGetValue<double>(out var p))
            return p;
        throw new DecisionException($"The server's answer to question '{name}' has no probability for '{key}'.", name);
    }
}
