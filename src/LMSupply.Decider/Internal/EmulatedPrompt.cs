using System.Text;
using LMSupply.Llama.Server;

namespace LMSupply.Decider.Internal;

/// <summary>The labels an emulated question's answers are written as; each must be one token of the model.</summary>
internal enum LabelSystem
{
    /// <summary><c>1</c>…<c>9</c> for options (<c>0</c> for «none of these»), <c>0</c>…<c>9</c> for score levels.</summary>
    Digits,

    /// <summary><c>B</c>…<c>Z</c> for 10 to 25 options, <c>A</c> for «none of these».</summary>
    Letters,

    /// <summary><c>Yes</c> and <c>No</c>.</summary>
    YesNo,
}

/// <summary>An emulated question: the user message to send and the label of each answer slot.</summary>
internal sealed record EmulatedQuestion(string Prompt, IReadOnlyList<string> SlotLabels, LabelSystem System)
{
    /// <summary>
    /// How many next-token candidates to ask for: the labels plus a margin for the tokens the model also considers,
    /// capped at <see cref="EmulatedPrompt.MaxTopLogprobs"/>. Fixed per question, so a request does not depend on the
    /// model's answer.
    /// </summary>
    public int TopLogprobs => Math.Min(SlotLabels.Count + EmulatedPrompt.TopLogprobsMargin, EmulatedPrompt.MaxTopLogprobs);
}

/// <summary>
/// How an instruct model is asked a typed question and how its answer is read: the state first and the question
/// after it (so consecutive questions about one state share a prompt prefix the server can reuse), the answers as
/// labelled lines, and one generated token whose candidates' probabilities are summed per label and renormalized.
/// The frame text is English; the state, question and options are the caller's, in their language.
/// </summary>
internal static class EmulatedPrompt
{
    public const int MaxDigitChoices = 9;
    public const int MaxLetterChoices = 25;
    public const int TopLogprobsMargin = 10;
    public const int MaxTopLogprobs = 40;

    public const string NoneOfThese = "none of these";
    public const string AnswerWithNumber = "Answer with the number only.";
    public const string AnswerWithLetter = "Answer with the letter only.";
    public const string AnswerYesOrNo = "Answer Yes or No.";

    public static readonly IReadOnlyList<string> DigitLabels = Enumerable.Range(0, 10).Select(i => i.ToString(System.Globalization.CultureInfo.InvariantCulture)).ToArray();
    public static readonly IReadOnlyList<string> LetterLabels = Enumerable.Range(0, 26).Select(i => ((char)('A' + i)).ToString()).ToArray();
    public static readonly IReadOnlyList<string> YesNoLabels = ["Yes", "No"];

    /// <summary>The labels of <paramref name="system"/>, each of which the model must spell as one token.</summary>
    public static IReadOnlyList<string> LabelsOf(LabelSystem system) => system switch
    {
        LabelSystem.Digits => DigitLabels,
        LabelSystem.Letters => LetterLabels,
        LabelSystem.YesNo => YesNoLabels,
        _ => throw new ArgumentOutOfRangeException(nameof(system)),
    };

    /// <summary>The label system <paramref name="question"/> needs.</summary>
    /// <exception cref="ArgumentException">A choice with more than <see cref="MaxLetterChoices"/> options.</exception>
    public static LabelSystem SystemFor(DecisionQuestion question) => question switch
    {
        ChoiceQuestion c when c.Options.Count > MaxLetterChoices => throw new ArgumentException(
            $"A choice takes at most {MaxLetterChoices} options on this model; this one has {c.Options.Count}.", nameof(question)),
        ChoiceQuestion c when c.Options.Count > MaxDigitChoices => LabelSystem.Letters,
        ChoiceQuestion or ScoreQuestion => LabelSystem.Digits,
        YesNoQuestion => LabelSystem.YesNo,
        _ => throw new ArgumentOutOfRangeException(nameof(question)),
    };

    /// <summary>The message and slot labels for asking <paramref name="question"/> about <paramref name="state"/>.</summary>
    public static EmulatedQuestion Build(string state, DecisionQuestion question)
    {
        var system = SystemFor(question);
        var text = new StringBuilder();
        text.Append(state).Append("\n\nQuestion: ").Append(question.Instructions).Append('\n');

        string[] slotLabels;
        string tail;
        switch (question)
        {
            case ChoiceQuestion c:
            {
                var labels = LabelsOf(system);
                // Label 0 (A) is «none of these» and comes first; the options take the labels after it.
                slotLabels = new string[DecisionMath.SlotCount(c)];
                if (c.AllowNone)
                {
                    text.Append(labels[0]).Append(". ").Append(NoneOfThese).Append('\n');
                    slotLabels[c.Options.Count] = labels[0];
                }
                for (var i = 0; i < c.Options.Count; i++)
                {
                    slotLabels[i] = labels[i + 1];
                    text.Append(labels[i + 1]).Append(". ").Append(c.Options[i]);
                    if (c.DescriptionOf(c.Options[i]) is { } description)
                        text.Append(": ").Append(description);
                    text.Append('\n');
                }
                tail = system == LabelSystem.Letters ? AnswerWithLetter : AnswerWithNumber;
                break;
            }

            case ScoreQuestion s:
            {
                slotLabels = DigitLabels.Take(s.Levels.Count).ToArray();
                for (var i = 0; i < s.Levels.Count; i++)
                    text.Append(slotLabels[i]).Append(". ").Append(s.Levels[i]).Append('\n');
                tail = AnswerWithNumber;
                break;
            }

            case YesNoQuestion y:
            {
                slotLabels = [.. YesNoLabels];
                if (y.TrueMeans is not null)
                    text.Append(YesNoLabels[0]).Append(": ").Append(y.TrueMeans).Append('\n');
                if (y.FalseMeans is not null)
                    text.Append(YesNoLabels[1]).Append(": ").Append(y.FalseMeans).Append('\n');
                tail = AnswerYesOrNo;
                break;
            }

            default:
                throw new ArgumentOutOfRangeException(nameof(question));
        }

        text.Append('\n').Append(tail);
        return new EmulatedQuestion(text.ToString(), slotLabels, system);
    }

    /// <summary>
    /// The probability of each answer slot from the first generated token's candidates: each candidate whose text,
    /// trimmed, is a slot's label adds its probability to that slot (a label can appear with and without a leading
    /// space); the sums are renormalized. <paramref name="labelMass"/> is the probability that fell on labels at all.
    /// </summary>
    /// <exception cref="DecisionException">No candidate is a label.</exception>
    public static double[] Read(
        string questionName,
        LlamaTokenLogprob? firstToken,
        IReadOnlyList<string> slotLabels,
        out double labelMass)
    {
        var mass = new double[slotLabels.Count];
        foreach (var candidate in firstToken?.TopLogprobs ?? [])
        {
            var text = candidate.Token?.Trim();
            if (string.IsNullOrEmpty(text))
                continue;
            for (var i = 0; i < slotLabels.Count; i++)
            {
                if (string.Equals(slotLabels[i], text, StringComparison.Ordinal))
                    mass[i] += Math.Exp(candidate.Logprob);
            }
        }

        labelMass = mass.Sum();
        if (labelMass <= 0)
        {
            var top = firstToken?.TopLogprobs is { Count: > 0 } list ? list[0].Token : firstToken?.Token;
            throw new DecisionException(
                $"The model's next token for question '{questionName}' was '{top}', and none of its " +
                $"{firstToken?.TopLogprobs?.Count ?? 0} most probable tokens is an answer label " +
                $"({string.Join(", ", slotLabels)}). A model that starts its answer with reasoning or markup cannot be read this way.",
                questionName,
                top);
        }

        return DecisionMath.Normalize(mass);
    }
}
