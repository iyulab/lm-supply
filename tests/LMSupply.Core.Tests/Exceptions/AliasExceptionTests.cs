using AwesomeAssertions;
using LMSupply.Exceptions;

namespace LMSupply.Core.Tests.Exceptions;

public class AliasExceptionTests
{
    [Fact]
    public void AliasConflictException_ShouldContainAliasName()
    {
        var ex = new AliasConflictException("default");
        ex.AliasName.Should().Be("default");
        ex.Message.Should().Contain("default");
        ex.Should().BeAssignableTo<LMSupplyException>();
    }

    [Fact]
    public void AliasChainException_ShouldContainBothAliases()
    {
        var ex = new AliasChainException("b", "a");
        ex.AliasName.Should().Be("b");
        ex.TargetAlias.Should().Be("a");
        ex.Message.Should().Contain("b");
        ex.Message.Should().Contain("a");
        ex.Should().BeAssignableTo<LMSupplyException>();
    }
}
