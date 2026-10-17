using AwesomeAssertions;
using LMSupply.Decider.Internal;
using LMSupply.Llama.Server;

namespace LMSupply.Decider.Tests;

public class EmulatedPromptTests
{
    private const string State = "Customer message: My card was charged twice.";

    [Fact]
    public void Choice_WithNone_NumbersTheOptionsFromOne_AndNoneIsZeroOnTheFirstLine()
    {
        var q = DecisionQuestion.Choice("What does the customer want?", ["refund", "cancel"], allowNone: true);

        var e = EmulatedPrompt.Build(State, q);

        e.System.Should().Be(LabelSystem.Digits);
        e.SlotLabels.Should().Equal("1", "2", "0");
        e.Prompt.Should().Be(
            "Customer message: My card was charged twice.\n\n" +
            "Question: What does the customer want?\n" +
            "0. none of these\n" +
            "1. refund\n" +
            "2. cancel\n\n" +
            "Answer with the number only.");
    }

    [Fact]
    public void Choice_WithoutNone_HasNoNoneLine_AndDescriptionsFollowTheOption()
    {
        var q = DecisionQuestion.Choice("Which team?", ["billing", "legal"], new Dictionary<string, string> { ["legal"] = "contracts and disputes" });

        var e = EmulatedPrompt.Build(State, q);

        e.SlotLabels.Should().Equal("1", "2");
        e.Prompt.Should().NotContain("none of these");
        e.Prompt.Should().Contain("1. billing\n2. legal: contracts and disputes\n");
    }

    [Fact]
    public void Choice_OfTenToTwentyFiveOptions_UsesLetters_NoneIsA()
    {
        var options = Enumerable.Range(1, 25).Select(i => $"option {i}").ToArray();

        var e = EmulatedPrompt.Build(State, DecisionQuestion.Choice("q", options, allowNone: true));

        e.System.Should().Be(LabelSystem.Letters);
        e.SlotLabels[0].Should().Be("B");
        e.SlotLabels[24].Should().Be("Z");
        e.SlotLabels[25].Should().Be("A");
        e.Prompt.Should().Contain("A. none of these\nB. option 1\n").And.EndWith("Answer with the letter only.");
        e.TopLogprobs.Should().Be(26 + EmulatedPrompt.TopLogprobsMargin, "the largest label set plus the margin stays under the cap");
    }

    [Fact]
    public void Choice_OfNineOptions_StaysOnDigits()
        => EmulatedPrompt.Build(State, DecisionQuestion.Choice("q", Enumerable.Range(1, 9).Select(i => $"o{i}").ToArray()))
            .System.Should().Be(LabelSystem.Digits);

    [Fact]
    public void Choice_OfMoreThanTwentyFiveOptions_IsRefused()
    {
        var q = DecisionQuestion.Choice("q", Enumerable.Range(1, 26).Select(i => $"o{i}").ToArray());

        var act = () => EmulatedPrompt.Build(State, q);

        act.Should().Throw<ArgumentException>().WithMessage("*25*26*");
    }

    [Fact]
    public void Score_LabelsTheLevelsFromZero()
    {
        var e = EmulatedPrompt.Build(State, DecisionQuestion.Score("How upset?", ["calm", "annoyed", "furious"]));

        e.SlotLabels.Should().Equal("0", "1", "2");
        e.Prompt.Should().Contain("Question: How upset?\n0. calm\n1. annoyed\n2. furious\n\nAnswer with the number only.");
        e.TopLogprobs.Should().Be(13);
    }

    [Fact]
    public void YesNo_LabelsAreYesAndNo()
    {
        var e = EmulatedPrompt.Build(State, DecisionQuestion.YesNo("Is the customer angry?", falseMeans: "calm or neutral"));

        e.SlotLabels.Should().Equal("Yes", "No");
        e.Prompt.Should().EndWith("Question: Is the customer angry?\nNo: calm or neutral\n\nAnswer Yes or No.");
    }

    [Fact]
    public void Prompts_OfOneState_ShareTheirPrefix()
    {
        var a = EmulatedPrompt.Build(State, DecisionQuestion.YesNo("Angry?")).Prompt;
        var b = EmulatedPrompt.Build(State, DecisionQuestion.Score("Urgent?", ["no", "yes"])).Prompt;

        a.Should().StartWith(State + "\n\nQuestion: ");
        b.Should().StartWith(State + "\n\nQuestion: ");
    }

    [Fact]
    public void Read_SumsALabelWithAndWithoutALeadingSpace_AndRenormalizesOverTheLabels()
    {
        var first = Candidates(("1", 0.6), (" 1", 0.1), ("2", 0.2), ("Sure", 0.05), ("0", 0.05));

        var slots = EmulatedPrompt.Read("q", first, ["1", "2", "0"], out var mass);

        mass.Should().BeApproximately(0.95, 1e-9);
        slots[0].Should().BeApproximately(0.7 / 0.95, 1e-9);
        slots[1].Should().BeApproximately(0.2 / 0.95, 1e-9);
        slots[2].Should().BeApproximately(0.05 / 0.95, 1e-9);
    }

    [Fact]
    public void Read_ALabelAbsentFromTheCandidates_HasZeroProbability()
    {
        var slots = EmulatedPrompt.Read("q", Candidates(("Yes", 0.9)), ["Yes", "No"], out var mass);

        slots.Should().Equal(1, 0);
        mass.Should().BeApproximately(0.9, 1e-9);
    }

    [Fact]
    public void Read_NoLabelAmongTheCandidates_ThrowsNamingTheFirstToken()
    {
        var first = Candidates(("<|channel>", 0.99), ("thought", 0.01));

        var act = () => EmulatedPrompt.Read("route", first, ["1", "2"], out _);

        var thrown = act.Should().Throw<DecisionException>().Which;
        thrown.FirstToken.Should().Be("<|channel>");
        thrown.QuestionName.Should().Be("route");
        thrown.Message.Should().Contain("<|channel>");
    }

    [Fact]
    public void Read_NoCandidatesAtAll_Throws()
    {
        var act = () => EmulatedPrompt.Read("q", null, ["Yes", "No"], out _);
        act.Should().Throw<DecisionException>();
    }

    internal static LlamaTokenLogprob Candidates(params (string Token, double P)[] candidates) => new()
    {
        Token = candidates[0].Token,
        Logprob = Math.Log(candidates[0].P),
        TopLogprobs = candidates.Select(c => new LlamaTokenLogprob { Token = c.Token, Logprob = Math.Log(c.P) }).ToList(),
    };
}
