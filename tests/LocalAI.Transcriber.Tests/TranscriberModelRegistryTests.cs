using FluentAssertions;
using LocalAI.Transcriber.Models;

namespace LocalAI.Transcriber.Tests;

public class TranscriberModelRegistryTests
{
    [Fact]
    public void Default_ShouldNotBeNull()
    {
        TranscriberModelRegistry.Default.Should().NotBeNull();
    }

    [Fact]
    public void GetAliases_ShouldReturnStandardAliases()
    {
        var aliases = TranscriberModelRegistry.Default.GetAliases().ToList();

        aliases.Should().Contain("default");
        aliases.Should().Contain("fast");
        aliases.Should().Contain("quality");
    }

    [Fact]
    public void GetAll_ShouldReturnAllModels()
    {
        var models = TranscriberModelRegistry.Default.GetAll().ToList();

        models.Should().NotBeEmpty();
        models.Count.Should().BeGreaterThanOrEqualTo(3);
    }

    [Theory]
    [InlineData("default")]
    [InlineData("fast")]
    [InlineData("quality")]
    [InlineData("large")]
    public void TryGet_ValidAlias_ShouldReturnModel(string alias)
    {
        var result = TranscriberModelRegistry.Default.TryGet(alias, out var model);

        result.Should().BeTrue();
        model.Should().NotBeNull();
        model!.Alias.Should().Be(alias);
    }

    [Fact]
    public void TryGet_InvalidAlias_ShouldReturnFalse()
    {
        var result = TranscriberModelRegistry.Default.TryGet("nonexistent-model", out var model);

        result.Should().BeFalse();
        model.Should().BeNull();
    }

    [Fact]
    public void TryGet_ByModelId_ShouldReturnModel()
    {
        var allModels = TranscriberModelRegistry.Default.GetAll().ToList();
        var firstModel = allModels.First();

        var result = TranscriberModelRegistry.Default.TryGet(firstModel.Id, out var model);

        result.Should().BeTrue();
        model.Should().NotBeNull();
        model!.Id.Should().Be(firstModel.Id);
    }

    [Fact]
    public void GetAliases_ShouldBeCaseInsensitive()
    {
        var resultLower = TranscriberModelRegistry.Default.TryGet("default", out var modelLower);
        var resultUpper = TranscriberModelRegistry.Default.TryGet("DEFAULT", out var modelUpper);
        var resultMixed = TranscriberModelRegistry.Default.TryGet("Default", out var modelMixed);

        resultLower.Should().BeTrue();
        resultUpper.Should().BeTrue();
        resultMixed.Should().BeTrue();
        modelLower.Should().BeEquivalentTo(modelUpper);
        modelUpper.Should().BeEquivalentTo(modelMixed);
    }

}
