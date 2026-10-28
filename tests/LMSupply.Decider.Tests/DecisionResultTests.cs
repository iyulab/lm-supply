using AwesomeAssertions;
using LMSupply.Decider.Internal;

namespace LMSupply.Decider.Tests;

public class DecisionResultTests
{
    [Fact]
    public void ReadingAQuestionThatWasNotAsked_ListsTheOnesThatWere()
    {
        var angry = DecisionQuestion.YesNo("Angry?");
        var result = new DecisionResult(DecisionPath.Emulated, [
            KeyValuePair.Create("route", DecisionMath.Build(angry, [0.3, 0.7], DecisionPath.Emulated, 1, null)),
            KeyValuePair.Create("angry", DecisionMath.Build(angry, [0.9, 0.1], DecisionPath.Emulated, 1, null)),
        ]);

        var act = () => result["nope"];

        act.Should().Throw<KeyNotFoundException>().WithMessage("*'nope'*route, angry*");
        result["angry"].IsTrue.Should().BeTrue();
    }
}
