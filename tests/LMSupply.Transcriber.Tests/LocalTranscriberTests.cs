using FluentAssertions;
using LMSupply.Transcriber.Models;

namespace LMSupply.Transcriber.Tests;

public class LocalTranscriberTests
{
    [Fact]
    public void GetAvailableModels_ShouldReturnAliases()
    {
        var models = LocalTranscriber.GetAvailableModels().ToList();

        models.Should().NotBeEmpty();
        models.Should().Contain("default");
        models.Should().Contain("fast");
        models.Should().Contain("quality");
    }

    [Fact]
    public void GetAllModels_ShouldReturnModelInfos()
    {
        var models = LocalTranscriber.GetAllModels().ToList();

        models.Should().NotBeEmpty();
        models.Should().AllSatisfy(m =>
        {
            m.Id.Should().NotBeNullOrEmpty();
            m.Alias.Should().NotBeNullOrEmpty();
            m.DisplayName.Should().NotBeNullOrEmpty();
            m.Architecture.Should().Be("Whisper");
        });
    }

    [Fact]
    public void GetAllModels_ShouldContainExpectedAliases()
    {
        var aliases = LocalTranscriber.GetAvailableModels().ToList();

        aliases.Should().Contain("fast");
        aliases.Should().Contain("default");
        aliases.Should().Contain("quality");
        aliases.Should().Contain("large");
    }

    [Fact]
    public void GetAllModels_DefaultModel_ShouldBeWhisperBase()
    {
        var models = LocalTranscriber.GetAllModels().ToList();
        var defaultModel = models.FirstOrDefault(m => m.Alias == "default");

        defaultModel.Should().NotBeNull();
        defaultModel!.Id.Should().Contain("base");
    }

    [Theory]
    [InlineData("default")]
    [InlineData("fast")]
    [InlineData("quality")]
    public void ResolveAlias_StandardAliases_ShouldResolve(string alias)
    {
        var result = TranscriberModelRegistry.Default.TryGet(alias, out var model);

        result.Should().BeTrue();
        model.Should().NotBeNull();
        model!.Id.Should().NotBeNullOrEmpty();
    }

    [Fact]
    public void GetAllModels_AllModels_ShouldHaveValidEncoderDecoderFiles()
    {
        var models = LocalTranscriber.GetAllModels();

        foreach (var model in models)
        {
            model.EncoderFile.Should().EndWith(".onnx");
            model.DecoderFile.Should().EndWith(".onnx");
        }
    }

    [Fact]
    public void GetAllModels_AllModels_ShouldHaveValidMelBins()
    {
        var models = LocalTranscriber.GetAllModels();

        foreach (var model in models)
        {
            model.NumMelBins.Should().BeOneOf(80, 128);
        }
    }

    [Fact]
    public void GetAvailableModels_Count_ShouldBeGreaterOrEqualToGetAllModels()
    {
        var aliases = LocalTranscriber.GetAvailableModels().ToList();
        var models = LocalTranscriber.GetAllModels().ToList();

        aliases.Count.Should().BeGreaterThanOrEqualTo(models.Count);
    }
}
