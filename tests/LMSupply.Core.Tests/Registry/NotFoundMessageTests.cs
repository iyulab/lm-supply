using AwesomeAssertions;
using LMSupply.Exceptions;

namespace LMSupply.Core.Tests.Registry;

/// <summary>
/// An alias the registry does not know is answered with the aliases it does, so the caller's next step is in the message.
/// </summary>
public class NotFoundMessageTests
{
    private static TestModelRegistry Registry(int aliases)
    {
        var models = Enumerable.Range(0, aliases)
            .Select(i => new TestModelInfo { Id = $"org/model-{i}", AliasName = i == 0 ? "default" : $"alias-{i:D2}" });
        return new TestModelRegistry(models);
    }

    [Theory]
    [InlineData("klingon")]
    [InlineData("ko-KR")]
    public void AnUnknownAlias_ListsTheKnownOnes(string unknown)
    {
        var act = () => Registry(3).Resolve(unknown);

        var message = act.Should().Throw<ModelNotFoundException>().Which.Message;
        message.Should().Contain($"'{unknown}'").And.Contain("alias-01, alias-02, default").And.Contain("GetAvailableModels()");
    }

    [Fact]
    public void AnEmptyAlias_SaysItIsEmpty_NotThatTheEmptyNameIsUnknown()
    {
        var act = () => Registry(3).Resolve("");

        var message = act.Should().Throw<ModelNotFoundException>().Which.Message;
        message.Should().Contain("empty").And.NotContain("''").And.Contain("Known aliases");
    }

    [Fact]
    public void ALongCatalog_ListsTheFirstAliasesAndSaysThereAreMore()
    {
        var act = () => Registry(40).Resolve("klingon");

        var message = act.Should().Throw<ModelNotFoundException>().Which.Message;
        message.Should().Contain("alias-01").And.Contain(", ...").And.NotContain("alias-39");
    }

    [Fact]
    public void ARegistryWithNoAliases_KeepsTheShortMessage()
    {
        var act = () => new TestModelRegistry([]).Resolve("klingon");

        act.Should().Throw<ModelNotFoundException>().Which.Message.Should().Be("Model 'klingon' not found in registry.");
    }
}
