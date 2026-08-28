using AwesomeAssertions;
using LMSupply;
using LMSupply.Exceptions;
using LMSupply.Synthesizer.Models;

namespace LMSupply.Synthesizer.Tests;

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
    public void TryResolve_ReturnsModelByAlias(string alias)
    {
        // Act
        var found = SynthesizerModelRegistry.Default.TryResolve(alias, out var model);

        // Assert
        found.Should().BeTrue();
        model.Should().NotBeNull();
        model!.AliasName.Should().Be(alias);
    }

    [Fact]
    public void TryResolve_ReturnsModelById()
    {
        // Arrange
        var id = "rhasspy/piper-voices";

        // Act
        var found = SynthesizerModelRegistry.Default.TryResolve(id, out var model);

        // Assert
        found.Should().BeTrue();
        model.Should().NotBeNull();
        model!.Id.Should().Be(id);
    }

    [Fact]
    public void TryResolve_ReturnsFalseForUnknown()
    {
        // Act
        var found = SynthesizerModelRegistry.Default.TryResolve("unknown-model", out var model);

        // Assert
        found.Should().BeFalse();
        model.Should().BeNull();
    }

    [Fact]
    public void TryResolve_IsCaseInsensitive()
    {
        // Act
        var found1 = SynthesizerModelRegistry.Default.TryResolve("DEFAULT", out var model1);
        var found2 = SynthesizerModelRegistry.Default.TryResolve("Default", out var model2);
        var found3 = SynthesizerModelRegistry.Default.TryResolve("default", out var model3);

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
        var aliases = SynthesizerModelRegistry.Default.GetAliases();
        var aliasNames = aliases.Select(a => a.Name).ToList();

        // Assert
        aliasNames.Should().NotBeEmpty();
        aliasNames.Should().Contain("default");
        aliasNames.Should().Contain("fast");
        aliasNames.Should().Contain("quality");
        aliasNames.Should().Contain("british");
        aliasNames.Should().Contain("korean");
        aliasNames.Should().Contain("japanese");
        aliasNames.Should().Contain("chinese");
    }

    [Fact]
    public void GetAvailableModels_ReturnsAllModels()
    {
        // Act
        var models = SynthesizerModelRegistry.Default.GetAvailableModels();

        // Assert
        models.Should().NotBeEmpty();
        // All models share the same ID "rhasspy/piper-voices", so only 1 unique by ID
        models.Should().HaveCountGreaterThanOrEqualTo(1);
    }

    [Fact]
    public void Resolve_AutoAlias_ShouldReturnModel()
    {
        var model = SynthesizerModelRegistry.Default.Resolve("auto");

        model.Should().NotBeNull();
        model.AliasName.Should().Be("auto");
    }

    [Fact]
    public void Resolve_UnknownAlias_ShouldThrow()
    {
        var act = () => SynthesizerModelRegistry.Default.Resolve("nonexistent");

        act.Should().Throw<ModelNotFoundException>()
            .Where(e => e.ModelId == "nonexistent");
    }

    [Fact]
    public void RegisterAlias_ShouldBeResolvable()
    {
        var registry = new SynthesizerModelRegistry(DefaultModels.All);

        registry.RegisterAlias("my-voice", "rhasspy/piper-voices");

        var model = registry.Resolve("my-voice");
        model.Should().NotBeNull();
        model.Id.Should().Be("rhasspy/piper-voices");
    }

    [Fact]
    public void RegisterAlias_SystemAliasConflict_ShouldThrow()
    {
        var registry = new SynthesizerModelRegistry(DefaultModels.All);

        var act = () => registry.RegisterAlias("default", "rhasspy/piper-voices");

        act.Should().Throw<AliasConflictException>();
    }

    [Fact]
    public void LocalSynthesizer_Registry_ShouldExpose()
    {
        var registry = LocalSynthesizer.Registry;

        registry.Should().NotBeNull();
        registry.Should().BeAssignableTo<IModelRegistry<SynthesizerModelInfo>>();
    }
}

public class DefaultModelsTests
{
    [Fact]
    public void EnUsLessac_IsDefaultModel()
    {
        // Assert
        DefaultModels.EnUsLessac.AliasName.Should().Be("default");
        DefaultModels.EnUsLessac.Language.Should().Be("en-US");
        DefaultModels.EnUsLessac.VoiceName.Should().Be("en/en_US/lessac/medium");
    }

    [Fact]
    public void EnUsRyan_IsFastModel()
    {
        // Assert
        DefaultModels.EnUsRyan.AliasName.Should().Be("fast");
        DefaultModels.EnUsRyan.Language.Should().Be("en-US");
        DefaultModels.EnUsRyan.SampleRate.Should().Be(16000);
    }

    [Fact]
    public void EnUsAmy_IsQualityModel()
    {
        // Assert
        DefaultModels.EnUsAmy.AliasName.Should().Be("quality");
        DefaultModels.EnUsAmy.Language.Should().Be("en-US");
        DefaultModels.EnUsAmy.SampleRate.Should().Be(22050);
    }

    [Fact]
    public void EnGbSemaine_IsBritishModel()
    {
        // Assert
        DefaultModels.EnGbSemaine.AliasName.Should().Be("british");
        DefaultModels.EnGbSemaine.Language.Should().Be("en-GB");
    }

    [Fact]
    public void KoKr_IsKoreanModel()
    {
        // Assert
        DefaultModels.KoKr.AliasName.Should().Be("korean");
        DefaultModels.KoKr.Language.Should().Be("ko-KR");
    }

    [Fact]
    public void JaJp_IsJapaneseModel()
    {
        // Assert
        DefaultModels.JaJp.AliasName.Should().Be("japanese");
        DefaultModels.JaJp.Language.Should().Be("ja-JP");
    }

    [Fact]
    public void ZhCn_IsChineseModel()
    {
        // Assert
        DefaultModels.ZhCn.AliasName.Should().Be("chinese");
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
            m.AliasName.Should().NotBeNullOrEmpty();
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
            AliasName = "test",
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
        info.AliasName.Should().Be("test");
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
            AliasName = "test",
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
