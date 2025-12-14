using FluentAssertions;
using LocalAI.Synthesizer;
using LocalAI.Synthesizer.Models;

namespace LocalAI.Synthesizer.Tests;

public class LocalSynthesizerTests
{
    [Fact]
    public void GetAvailableModels_ReturnsAllAliases()
    {
        // Act
        var models = LocalSynthesizer.GetAvailableModels().ToList();

        // Assert
        models.Should().NotBeEmpty();
        models.Should().Contain("default");
        models.Should().Contain("fast");
        models.Should().Contain("quality");
    }

    [Fact]
    public void GetAllModels_ReturnsModelInfoCollection()
    {
        // Act
        var models = LocalSynthesizer.GetAllModels().ToList();

        // Assert
        models.Should().NotBeEmpty();
        models.Should().AllSatisfy(m =>
        {
            m.Id.Should().NotBeNullOrEmpty();
            m.Alias.Should().NotBeNullOrEmpty();
            m.DisplayName.Should().NotBeNullOrEmpty();
        });
    }

    [Fact]
    public void GetAvailableModels_ContainsMultiLanguageModels()
    {
        // Act
        var models = LocalSynthesizer.GetAvailableModels().ToList();

        // Assert
        models.Should().Contain("korean");
        models.Should().Contain("japanese");
        models.Should().Contain("chinese");
        models.Should().Contain("british");
    }
}
