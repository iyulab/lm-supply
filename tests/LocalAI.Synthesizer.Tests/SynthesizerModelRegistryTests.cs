using FluentAssertions;
using LocalAI.Synthesizer.Models;

namespace LocalAI.Synthesizer.Tests;

public class SynthesizerModelRegistryTests
{
    [Fact]
    public void Default_ReturnsSharedInstance()
    {
        // Act
        var instance1 = SynthesizerModelRegistry.Default;
        var instance2 = SynthesizerModelRegistry.Default;

        // Assert
        instance1.Should().BeSameAs(instance2);
    }

    [Theory]
    [InlineData("default")]
    [InlineData("fast")]
    [InlineData("quality")]
    [InlineData("british")]
    [InlineData("korean")]
    [InlineData("japanese")]
    [InlineData("chinese")]
    public void TryGet_ReturnsModelByAlias(string alias)
    {
        // Act
        var found = SynthesizerModelRegistry.Default.TryGet(alias, out var model);

        // Assert
        found.Should().BeTrue();
        model.Should().NotBeNull();
        model!.Alias.Should().Be(alias);
    }

    [Fact]
    public void TryGet_ReturnsModelById()
    {
        // Arrange
        var id = "rhasspy/piper-voices";

        // Act
        var found = SynthesizerModelRegistry.Default.TryGet(id, out var model);

        // Assert
        found.Should().BeTrue();
        model.Should().NotBeNull();
        model!.Id.Should().Be(id);
    }

    [Fact]
    public void TryGet_ReturnsFalseForUnknown()
    {
        // Act
        var found = SynthesizerModelRegistry.Default.TryGet("unknown-model", out var model);

        // Assert
        found.Should().BeFalse();
        model.Should().BeNull();
    }

    [Fact]
    public void TryGet_IsCaseInsensitive()
    {
        // Act
        var found1 = SynthesizerModelRegistry.Default.TryGet("DEFAULT", out var model1);
        var found2 = SynthesizerModelRegistry.Default.TryGet("Default", out var model2);
        var found3 = SynthesizerModelRegistry.Default.TryGet("default", out var model3);

        // Assert
        found1.Should().BeTrue();
        found2.Should().BeTrue();
        found3.Should().BeTrue();
        model1.Should().BeEquivalentTo(model2);
        model2.Should().BeEquivalentTo(model3);
    }

    [Fact]
    public void GetAliases_ReturnsAllRegisteredAliases()
    {
        // Act
        var aliases = SynthesizerModelRegistry.Default.GetAliases().ToList();

        // Assert
        aliases.Should().NotBeEmpty();
        aliases.Should().Contain("default");
        aliases.Should().Contain("fast");
        aliases.Should().Contain("quality");
        aliases.Should().Contain("british");
        aliases.Should().Contain("korean");
        aliases.Should().Contain("japanese");
        aliases.Should().Contain("chinese");
    }

    [Fact]
    public void GetAll_ReturnsAllModels()
    {
        // Act
        var models = SynthesizerModelRegistry.Default.GetAll().ToList();

        // Assert
        models.Should().NotBeEmpty();
        models.Should().HaveCount(7); // 7 default models
    }
}

public class DefaultModelsTests
{
    [Fact]
    public void EnUsLessac_IsDefaultModel()
    {
        // Assert
        DefaultModels.EnUsLessac.Alias.Should().Be("default");
        DefaultModels.EnUsLessac.Language.Should().Be("en-US");
        DefaultModels.EnUsLessac.VoiceName.Should().Be("en_US-lessac-medium");
    }

    [Fact]
    public void EnUsRyan_IsFastModel()
    {
        // Assert
        DefaultModels.EnUsRyan.Alias.Should().Be("fast");
        DefaultModels.EnUsRyan.Language.Should().Be("en-US");
        DefaultModels.EnUsRyan.SampleRate.Should().Be(16000);
    }

    [Fact]
    public void EnUsAmy_IsQualityModel()
    {
        // Assert
        DefaultModels.EnUsAmy.Alias.Should().Be("quality");
        DefaultModels.EnUsAmy.Language.Should().Be("en-US");
        DefaultModels.EnUsAmy.SampleRate.Should().Be(22050);
    }

    [Fact]
    public void EnGbSemaine_IsBritishModel()
    {
        // Assert
        DefaultModels.EnGbSemaine.Alias.Should().Be("british");
        DefaultModels.EnGbSemaine.Language.Should().Be("en-GB");
    }

    [Fact]
    public void KoKr_IsKoreanModel()
    {
        // Assert
        DefaultModels.KoKr.Alias.Should().Be("korean");
        DefaultModels.KoKr.Language.Should().Be("ko-KR");
    }

    [Fact]
    public void JaJp_IsJapaneseModel()
    {
        // Assert
        DefaultModels.JaJp.Alias.Should().Be("japanese");
        DefaultModels.JaJp.Language.Should().Be("ja-JP");
    }

    [Fact]
    public void ZhCn_IsChineseModel()
    {
        // Assert
        DefaultModels.ZhCn.Alias.Should().Be("chinese");
        DefaultModels.ZhCn.Language.Should().Be("zh-CN");
    }

    [Fact]
    public void All_ContainsAllDefaultModels()
    {
        // Assert
        DefaultModels.All.Should().HaveCount(7);
        DefaultModels.All.Should().Contain(DefaultModels.EnUsLessac);
        DefaultModels.All.Should().Contain(DefaultModels.EnUsRyan);
        DefaultModels.All.Should().Contain(DefaultModels.EnUsAmy);
        DefaultModels.All.Should().Contain(DefaultModels.EnGbSemaine);
        DefaultModels.All.Should().Contain(DefaultModels.KoKr);
        DefaultModels.All.Should().Contain(DefaultModels.JaJp);
        DefaultModels.All.Should().Contain(DefaultModels.ZhCn);
    }

    [Fact]
    public void AllModels_HaveRequiredFields()
    {
        // Assert
        DefaultModels.All.Should().AllSatisfy(m =>
        {
            m.Id.Should().NotBeNullOrEmpty();
            m.Alias.Should().NotBeNullOrEmpty();
            m.DisplayName.Should().NotBeNullOrEmpty();
            m.Architecture.Should().Be("VITS");
            m.Language.Should().NotBeNullOrEmpty();
            m.VoiceName.Should().NotBeNullOrEmpty();
            m.ModelFile.Should().NotBeNullOrEmpty();
            m.ConfigFile.Should().NotBeNullOrEmpty();
            m.SampleRate.Should().BeGreaterThan(0);
        });
    }

    [Fact]
    public void AllModels_UsePiperVoicesRepo()
    {
        // Assert
        DefaultModels.All.Should().AllSatisfy(m =>
        {
            m.Id.Should().Be("rhasspy/piper-voices");
        });
    }
}

public class SynthesizerModelInfoTests
{
    [Fact]
    public void SynthesizerModelInfo_CanBeCreated()
    {
        // Act
        var info = new SynthesizerModelInfo
        {
            Id = "test/model",
            Alias = "test",
            DisplayName = "Test Model",
            Architecture = "VITS",
            Language = "en-US",
            VoiceName = "test-voice",
            ModelFile = "custom.onnx",
            ConfigFile = "custom.json",
            NumSpeakers = 3,
            SampleRate = 44100
        };

        // Assert
        info.Id.Should().Be("test/model");
        info.Alias.Should().Be("test");
        info.DisplayName.Should().Be("Test Model");
        info.VoiceName.Should().Be("test-voice");
        info.ModelFile.Should().Be("custom.onnx");
        info.ConfigFile.Should().Be("custom.json");
        info.NumSpeakers.Should().Be(3);
        info.SampleRate.Should().Be(44100);
    }

    [Fact]
    public void SynthesizerModelInfo_HasCorrectDefaults()
    {
        // Act - only required properties
        var info = new SynthesizerModelInfo
        {
            Id = "test",
            Alias = "test",
            DisplayName = "Test"
        };

        // Assert
        info.Architecture.Should().Be("VITS");
        info.Language.Should().Be("en");
        info.VoiceName.Should().BeNull();
        info.NumSpeakers.Should().Be(1);
        info.SampleRate.Should().Be(22050);
        info.ModelFile.Should().Be("model.onnx");
        info.ConfigFile.Should().Be("config.json");
        info.SizeBytes.Should().Be(0);
        info.Description.Should().BeNull();
        info.License.Should().Be("MIT");
    }
}
