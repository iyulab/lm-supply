using AwesomeAssertions;
using LMSupply.Exceptions;

namespace LMSupply.Core.Tests.Exceptions;

public class ContextLengthExceededExceptionTests
{
    [Fact]
    public void Message_NamesTheLimitTheModelWasLoadedWith_NotAMaximumOfTheModel()
    {
        var ex = new ContextLengthExceededException(44051, 8192);

        ex.Message.Should().Contain("44051 tokens").And.Contain("8192 tokens").And.Contain("was loaded with");
        ex.Message.Should().NotContain("model's maximum");
        ex.TokenCount.Should().Be(44051);
        ex.MaxContextLength.Should().Be(8192);
    }

    [Fact]
    public void Message_AdvisesLoadingWithALargerContext_NotChangingTheModel()
    {
        new ContextLengthExceededException(100, 64).Message.Should().Contain("larger context size").And.NotContain("use a model");
    }

    [Fact]
    public void Message_WithoutATokenCount_StillNamesTheLimit()
    {
        new ContextLengthExceededException(null, 512).Message.Should().Contain("512 tokens").And.NotContain("null");
    }
}
