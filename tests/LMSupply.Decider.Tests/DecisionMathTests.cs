using AwesomeAssertions;
using LMSupply.Decider.Internal;

namespace LMSupply.Decider.Tests;

public class DecisionMathTests
{
    private static readonly ChoiceQuestion RouteWithNone = DecisionQuestion.Choice("Which team?", ["billing", "shipping", "technical"], allowNone: true);

    [Fact]
    public void Choice_TheMostProbableOption_IsTheChoice()
    {
        var d = DecisionMath.Build(RouteWithNone, [0.7, 0.1, 0.05, 0.15], DecisionPath.Emulated, 0.98, null);

        d.Choice.Should().Be("billing");
        d.Probabilities.Keys.Should().Equal("billing", "shipping", "technical");
        d.Probabilities["billing"].Should().Be(0.7);
        d.NoneProbability.Should().Be(0.15);
        d.Confidence.Should().Be(0.7);
        d.LabelMass.Should().Be(0.98);
        d.Calibrated.Should().BeFalse();
    }

    [Fact]
    public void Choice_NoneMostProbable_IsANullChoice_AndNoneIsNotAnOption()
    {
        var d = DecisionMath.Build(RouteWithNone, [0.1, 0.1, 0.1, 0.7], DecisionPath.Native, 1, null);

        d.Choice.Should().BeNull();
        d.NoneProbability.Should().Be(0.7);
        d.Confidence.Should().Be(0.7);
        d.Probabilities.Should().NotContainKey(NativeWire.NoneOption);
        (d.Probabilities.Values.Sum() + d.NoneProbability).Should().BeApproximately(1, 1e-12);
    }

    [Fact]
    public void Score_IsTheExpectedLevel_AndLevelTheMostProbable()
    {
        var q = DecisionQuestion.Score("How urgent?", ["can wait", "this week", "today", "right now"]);

        var d = DecisionMath.Build(q, [0.1, 0.2, 0.6, 0.1], DecisionPath.Emulated, 1, null);

        d.Score.Should().BeApproximately(0 * 0.1 + 1 * 0.2 + 2 * 0.6 + 3 * 0.1, 1e-12);
        d.Level.Should().Be(2);
        d.Probabilities["today"].Should().Be(0.6);
    }

    [Fact]
    public void YesNo_ProbabilityIsYes()
    {
        var d = DecisionMath.Build(DecisionQuestion.YesNo("Angry?"), [0.3, 0.7], DecisionPath.Emulated, 1, null);

        d.IsTrue.Should().BeFalse();
        d.Probability.Should().Be(0.3);
        d.Probabilities["yes"].Should().Be(0.3);
        d.Probabilities["no"].Should().Be(0.7);
        d.Confidence.Should().Be(0.7);
    }

    [Fact]
    public void TypedAccessors_OfAnotherKind_Throw()
    {
        var d = DecisionMath.Build(DecisionQuestion.YesNo("Angry?"), [0.9, 0.1], DecisionPath.Emulated, 1, null);

        ((Func<object?>)(() => d.Choice)).Should().Throw<InvalidOperationException>();
        ((Func<object?>)(() => d.Score)).Should().Throw<InvalidOperationException>();
        ((Func<object?>)(() => d.Level)).Should().Throw<InvalidOperationException>();
    }

    [Fact]
    public void Temperature_IsPToTheOneOverT_Renormalized()
    {
        double[] p = [0.8, 0.2];

        var flattened = DecisionMath.ApplyTemperature(p, 2);

        // p^(1/2): 0.8944, 0.4472 → 0.6667, 0.3333
        flattened[0].Should().BeApproximately(Math.Sqrt(0.8) / (Math.Sqrt(0.8) + Math.Sqrt(0.2)), 1e-12);
        flattened[1].Should().BeApproximately(1.0 / 3, 1e-12);
        DecisionMath.ApplyTemperature(p, 1).Should().Equal(0.8, 0.2);
        DecisionMath.ApplyTemperature(p, 0.5)[0].Should().BeApproximately(0.64 / 0.68, 1e-12);
    }

    [Fact]
    public void Temperature_KeepsZeros_AndSurvivesUnderflow()
    {
        DecisionMath.ApplyTemperature([1e-300, 0, 1e-310], 0.01).Should().Equal(1, 0, 0);
    }

    [Fact]
    public void Temperature_ForTheKind_IsApplied_AndMarksCalibrated()
    {
        var temperatures = new Dictionary<DecisionKind, double> { [DecisionKind.YesNo] = 2 };

        var yesNo = DecisionMath.Build(DecisionQuestion.YesNo("q"), [0.8, 0.2], DecisionPath.Emulated, 1, temperatures);
        var choice = DecisionMath.Build(RouteWithNone, [0.7, 0.1, 0.05, 0.15], DecisionPath.Emulated, 1, temperatures);

        yesNo.Calibrated.Should().BeTrue();
        yesNo.Probability.Should().BeApproximately(2.0 / 3, 1e-12);
        choice.Calibrated.Should().BeFalse("only the kinds with a temperature are scaled");
        choice.Probabilities["billing"].Should().Be(0.7);
    }

    [Fact]
    public void Normalize_ScalesToOne()
        => DecisionMath.Normalize([2, 6]).Should().Equal(0.25, 0.75);
}
