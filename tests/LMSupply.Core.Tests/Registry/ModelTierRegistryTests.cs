using AwesomeAssertions;
using LMSupply.Exceptions;

namespace LMSupply.Core.Tests.Registry;

public class ModelTierRegistryTests
{
    private static TestModelInfo Model(string id, string alias, ModelTier? tier = null) =>
        new() { Id = id, AliasName = alias, Tier = tier };

    // A ladder with gaps: entry, basic (called 'default' by its own name), advanced.
    private static TestModelRegistry Ladder(IReadOnlyDictionary<string, string>? retired = null) => new(
        [
            Model("org/tiny", "entry", ModelTier.Entry),
            Model("org/base", "default", ModelTier.Basic),
            Model("org/big", "advanced", ModelTier.Advanced),
            Model("org/pinned", "pinned-by-name"),
        ],
        retired);

    [Theory]
    [InlineData("entry", "org/tiny")]
    [InlineData("basic", "org/base")]
    [InlineData("default", "org/base")]
    [InlineData("advanced", "org/big")]
    public void ARungIsAnAlias_OfItsModel_UnderItsOwnNameAndUnderTheModelsPrimaryAlias(string alias, string id)
        => Ladder().Resolve(alias).Id.Should().Be(id);

    [Fact]
    public void ARungNameIsCaseInsensitive_LikeEveryAlias()
        => Ladder().Resolve("ADVANCED").Id.Should().Be("org/big");

    [Fact]
    public void GetTiers_ListsOnlyTheOccupiedRungs_LowestFirst_AndNotOffLadderModels()
    {
        var tiers = Ladder().GetTiers();

        tiers.Select(t => (t.Tier, t.Alias, t.Model.Id)).Should().Equal(
            (ModelTier.Entry, "entry", "org/tiny"),
            (ModelTier.Basic, "basic", "org/base"),
            (ModelTier.Advanced, "advanced", "org/big"));
    }

    [Fact]
    public void GetAliases_ListsTheRungAliasOnce_WithTheModelsCatalogKey()
    {
        var aliases = Ladder().GetAliases().Where(a => a.Kind == AliasKind.System).ToList();

        aliases.Should().Contain(a => a.Name == "basic" && a.TargetModelId == "org/base");
        aliases.Should().Contain(a => a.Name == "default" && a.TargetModelId == "org/base");
        aliases.Count(a => a.Name == "entry").Should().Be(1, "a rung named like the model's own alias is not registered twice");
    }

    [Fact]
    public void AnAbsentRung_IsNotAnswered_WithANeighbour_Unless_TheCallerAsks()
    {
        var registry = Ladder();

        registry.TryResolveTier(ModelTier.Balanced, out var none).Should().BeFalse();
        none.Should().BeNull();
        registry.TryResolveTier(ModelTier.Balanced, out var lower, TierFallback.Lower).Should().BeTrue();
        lower!.Id.Should().Be("org/base");
        registry.TryResolveTier(ModelTier.Balanced, out var higher, TierFallback.Higher).Should().BeTrue();
        higher!.Id.Should().Be("org/big");
    }

    [Fact]
    public void AFallback_HasNothingToFallBackTo_PastTheEndOfTheLadder()
    {
        var registry = Ladder();

        registry.TryResolveTier(ModelTier.Frontier, out _, TierFallback.Higher).Should().BeFalse();
        registry.TryResolveTier(ModelTier.Frontier, out var lower, TierFallback.Lower).Should().BeTrue();
        lower!.Id.Should().Be("org/big");
    }

    [Fact]
    public void ResolveTier_OfAnAbsentRung_NamesTheRungsTheDomainHolds()
    {
        var act = () => Ladder().ResolveTier(ModelTier.Frontier);

        act.Should().Throw<ModelNotFoundException>()
            .WithMessage("*'frontier'*holds no model*entry, basic, advanced*");
    }

    [Fact]
    public void AnAbsentRungAlias_IsNotFound_NotMistakenForAHubRepository()
    {
        var registry = Ladder();

        registry.TryResolve("frontier", out _).Should().BeFalse();
        registry.Invoking(r => r.Resolve("frontier")).Should().Throw<ModelNotFoundException>();
    }

    [Fact]
    public void ADomainWithoutALadder_SaysSo()
    {
        var registry = new TestModelRegistry([Model("org/a", "default")]);

        registry.GetTiers().Should().BeEmpty();
        registry.Invoking(r => r.ResolveTier(ModelTier.Basic)).Should().Throw<ModelNotFoundException>()
            .WithMessage("*no ladder*");
    }

    [Fact]
    public void TwoDifferentModelsOnOneRung_AreACatalogDefect_ReportedWhenTheCatalogIsBuilt()
    {
        var act = () => new TestModelRegistry(
        [
            Model("org/a", "one", ModelTier.Basic),
            Model("org/b", "two", ModelTier.Basic),
        ]);

        act.Should().Throw<InvalidOperationException>().WithMessage("*'basic'*org/a*org/b*");
    }

    [Fact]
    public void OneModelUnderTwoAliases_MayHoldARung_OnBothEntries()
    {
        var registry = new TestModelRegistry(
        [
            Model("org/a", "default", ModelTier.Basic),
            Model("org/a", "quality", ModelTier.Basic),
        ]);

        registry.GetTiers().Should().ContainSingle().Which.Model.Id.Should().Be("org/a");
    }

    [Fact]
    public void ARungNameThatAnotherModelAlreadyUses_IsACatalogDefect()
    {
        var act = () => new TestModelRegistry(
        [
            Model("org/a", "basic"),
            Model("org/b", "default", ModelTier.Basic),
        ]);

        act.Should().Throw<InvalidOperationException>().WithMessage("*'basic'*org/a*org/b*");
    }

    [Fact]
    public void ARetiredName_IsNotResolved_ButTheErrorSaysWhatReplacedIt()
    {
        var registry = Ladder(new Dictionary<string, string> { ["fast"] = "it is now 'entry'" });

        registry.TryResolve("fast", out _).Should().BeFalse("a retired name loads nothing");
        registry.Invoking(r => r.Resolve("fast")).Should().Throw<ModelNotFoundException>()
            .WithMessage("*'fast' is no longer an alias; it is now 'entry'*");
        registry.DescribeRetiredAlias("FAST:fp16").Should().Contain("entry", "a variant qualifier does not hide the name");
        registry.DescribeRetiredAlias("entry").Should().BeNull();
    }
}
