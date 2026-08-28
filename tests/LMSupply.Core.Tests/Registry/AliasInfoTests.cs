using AwesomeAssertions;

namespace LMSupply.Core.Tests.Registry;

public class AliasInfoTests
{
    [Fact]
    public void AliasInfo_ShouldStoreProperties()
    {
        var info = new AliasInfo("default", "BAAI/bge-small-en-v1.5", AliasKind.System);
        info.Name.Should().Be("default");
        info.TargetModelId.Should().Be("BAAI/bge-small-en-v1.5");
        info.Kind.Should().Be(AliasKind.System);
    }

    [Fact]
    public void AliasKind_ShouldHaveSystemAndUser()
    {
        Enum.GetValues<AliasKind>().Should().Contain(AliasKind.System);
        Enum.GetValues<AliasKind>().Should().Contain(AliasKind.User);
    }
}
