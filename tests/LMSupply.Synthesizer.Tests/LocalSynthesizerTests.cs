using AwesomeAssertions;
using LMSupply.Synthesizer;
using LMSupply.Synthesizer.Models;

namespace LMSupply.Synthesizer.Tests;

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
    public void GetAllModels_ReturnsEveryVoice()
    {
        // Act
        var models = LocalSynthesizer.GetAllModels().ToList();

        // Assert
        models.Select(m => m.AliasName).Should().Equal("default", "lessac", "fast", "quality", "british", "korean", "chinese");
        models.Should().AllSatisfy(m =>
        {
            m.Id.Should().NotBeNullOrEmpty();
            m.AliasName.Should().NotBeNullOrEmpty();
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
        models.Should().NotContain("japanese", "its voice file was never published");
        models.Should().Contain("chinese");
        models.Should().Contain("british");
    }
}
