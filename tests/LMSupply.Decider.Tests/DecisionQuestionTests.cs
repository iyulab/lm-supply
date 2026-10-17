using AwesomeAssertions;

namespace LMSupply.Decider.Tests;

public class DecisionQuestionTests
{
    [Fact]
    public void Choice_KeepsTheOptionsInOrder()
    {
        var q = DecisionQuestion.Choice("Which team?", ["billing", "shipping"], new Dictionary<string, string> { ["billing"] = "money" }, allowNone: true);

        q.Kind.Should().Be(DecisionKind.Choice);
        q.Options.Should().Equal("billing", "shipping");
        q.Descriptions!["billing"].Should().Be("money");
        q.AllowNone.Should().BeTrue();
    }

    [Fact]
    public void Choice_OneOption_NeedsAllowNone()
    {
        var act = () => DecisionQuestion.Choice("q", ["only"]);
        act.Should().Throw<ArgumentException>();

        DecisionQuestion.Choice("q", ["only"], allowNone: true).Options.Should().ContainSingle();
    }

    [Fact]
    public void Choice_DuplicateOrEmptyOptions_AreRefused()
    {
        ((Action)(() => DecisionQuestion.Choice("q", ["a", "a"]))).Should().Throw<ArgumentException>();
        ((Action)(() => DecisionQuestion.Choice("q", ["a", " "]))).Should().Throw<ArgumentException>();
    }

    [Fact]
    public void Choice_ADescriptionOfNoOption_IsRefused()
    {
        var act = () => DecisionQuestion.Choice("q", ["a", "b"], new Dictionary<string, string> { ["c"] = "?" });
        act.Should().Throw<ArgumentException>().WithMessage("*'c'*");
    }

    [Theory]
    [InlineData(1)]
    [InlineData(11)]
    public void Score_TakesTwoToTenLevels(int count)
    {
        var act = () => DecisionQuestion.Score("q", Enumerable.Range(0, count).Select(i => $"level {i}").ToArray());
        act.Should().Throw<ArgumentException>();
    }

    [Fact]
    public void Score_TenLevels_IsAccepted()
        => DecisionQuestion.Score("q", Enumerable.Range(0, 10).Select(i => $"level {i}").ToArray()).Levels.Should().HaveCount(10);

    [Fact]
    public void Instructions_MustNotBeEmpty()
    {
        var act = () => DecisionQuestion.YesNo(" ");
        act.Should().Throw<ArgumentException>();
    }

    [Fact]
    public void YesNo_BlankMeanings_AreNone()
    {
        var q = DecisionQuestion.YesNo("Urgent?", trueMeans: "", falseMeans: "it can wait");
        q.TrueMeans.Should().BeNull();
        q.FalseMeans.Should().Be("it can wait");
    }
}
