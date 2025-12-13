using FluentAssertions;
using LocalAI.Generator.Abstractions;
using LocalAI.Generator.Models;
using NSubstitute;

namespace LocalAI.Generator.Tests;

public class SpeculativeDecoderTests
{
    [Fact]
    public void Constructor_NullDraftModel_ThrowsArgumentNullException()
    {
        // Arrange
        var targetModel = Substitute.For<IGeneratorModel>();

        // Act & Assert
        var action = () => new SpeculativeDecoder(null!, targetModel);
        action.Should().Throw<ArgumentNullException>()
            .WithParameterName("draftModel");
    }

    [Fact]
    public void Constructor_NullTargetModel_ThrowsArgumentNullException()
    {
        // Arrange
        var draftModel = Substitute.For<IGeneratorModel>();

        // Act & Assert
        var action = () => new SpeculativeDecoder(draftModel, null!);
        action.Should().Throw<ArgumentNullException>()
            .WithParameterName("targetModel");
    }

    [Fact]
    public void DraftModelId_ReturnsDraftModelId()
    {
        // Arrange
        var draftModel = Substitute.For<IGeneratorModel>();
        draftModel.ModelId.Returns("draft-model");
        var targetModel = Substitute.For<IGeneratorModel>();
        targetModel.ModelId.Returns("target-model");

        using var decoder = new SpeculativeDecoder(draftModel, targetModel);

        // Act
        var result = decoder.DraftModelId;

        // Assert
        result.Should().Be("draft-model");
    }

    [Fact]
    public void TargetModelId_ReturnsTargetModelId()
    {
        // Arrange
        var draftModel = Substitute.For<IGeneratorModel>();
        draftModel.ModelId.Returns("draft-model");
        var targetModel = Substitute.For<IGeneratorModel>();
        targetModel.ModelId.Returns("target-model");

        using var decoder = new SpeculativeDecoder(draftModel, targetModel);

        // Act
        var result = decoder.TargetModelId;

        // Assert
        result.Should().Be("target-model");
    }

    [Fact]
    public void SpeculationLength_DefaultsTo5()
    {
        // Arrange
        var draftModel = Substitute.For<IGeneratorModel>();
        var targetModel = Substitute.For<IGeneratorModel>();

        using var decoder = new SpeculativeDecoder(draftModel, targetModel);

        // Act & Assert
        decoder.SpeculationLength.Should().Be(5);
    }

    [Fact]
    public void SpeculationLength_CanBeSet()
    {
        // Arrange
        var draftModel = Substitute.For<IGeneratorModel>();
        var targetModel = Substitute.For<IGeneratorModel>();

        using var decoder = new SpeculativeDecoder(draftModel, targetModel);

        // Act
        decoder.SpeculationLength = 10;

        // Assert
        decoder.SpeculationLength.Should().Be(10);
    }

    [Fact]
    public void GetLastStats_InitiallyEmpty()
    {
        // Arrange
        var draftModel = Substitute.For<IGeneratorModel>();
        var targetModel = Substitute.For<IGeneratorModel>();

        using var decoder = new SpeculativeDecoder(draftModel, targetModel);

        // Act
        var stats = decoder.GetLastStats();

        // Assert
        stats.TotalTokens.Should().Be(0);
        stats.DraftTokens.Should().Be(0);
        stats.AcceptedTokens.Should().Be(0);
        stats.TargetTokens.Should().Be(0);
    }

    [Fact]
    public async Task GenerateCompleteAsync_ReturnsResult()
    {
        // Arrange
        var draftModel = Substitute.For<IGeneratorModel>();
        var targetModel = Substitute.For<IGeneratorModel>();

        // Setup draft model to generate tokens
        draftModel.GenerateAsync(Arg.Any<string>(), Arg.Any<GeneratorOptions>(), Arg.Any<CancellationToken>())
            .Returns(CreateAsyncEnumerable("Hello", " world", "!"));

        // Setup target model to verify tokens (same tokens = accepted)
        targetModel.GenerateAsync(Arg.Any<string>(), Arg.Any<GeneratorOptions>(), Arg.Any<CancellationToken>())
            .Returns(CreateAsyncEnumerable("Hello", " world", "!"));

        using var decoder = new SpeculativeDecoder(draftModel, targetModel);

        // Act
        var result = await decoder.GenerateCompleteAsync("Test", new GeneratorOptions { MaxTokens = 3 });

        // Assert
        result.Should().NotBeNull();
        result.Text.Should().NotBeEmpty();
        result.Stats.Should().NotBeNull();
    }

    [Fact]
    public async Task DisposeAsync_CanBeCalledMultipleTimes()
    {
        // Arrange
        var draftModel = Substitute.For<IGeneratorModel>();
        var targetModel = Substitute.For<IGeneratorModel>();

        var decoder = new SpeculativeDecoder(draftModel, targetModel);

        // Act
        await decoder.DisposeAsync();
        await decoder.DisposeAsync();

        // Assert - should only dispose models once
        await draftModel.Received(1).DisposeAsync();
        await targetModel.Received(1).DisposeAsync();
    }

    [Fact]
    public async Task GenerateCompleteAsync_WhenDisposed_ThrowsObjectDisposedException()
    {
        // Arrange
        var draftModel = Substitute.For<IGeneratorModel>();
        var targetModel = Substitute.For<IGeneratorModel>();

        var decoder = new SpeculativeDecoder(draftModel, targetModel);
        await decoder.DisposeAsync();

        // Act & Assert
        var action = () => decoder.GenerateCompleteAsync("prompt");
        await action.Should().ThrowAsync<ObjectDisposedException>();
    }

    private static async IAsyncEnumerable<string> CreateAsyncEnumerable(params string[] items)
    {
        foreach (var item in items)
        {
            yield return item;
            await Task.Yield();
        }
    }
}

public class SpeculativeStatsTests
{
    [Fact]
    public void AcceptanceRate_WithDraftTokens_ReturnsCorrectRate()
    {
        // Arrange
        var stats = new SpeculativeStats
        {
            TotalTokens = 10,
            DraftTokens = 8,
            AcceptedTokens = 6,
            TargetTokens = 4,
            ElapsedMilliseconds = 1000
        };

        // Act
        var rate = stats.AcceptanceRate;

        // Assert
        rate.Should().Be(0.75); // 6/8
    }

    [Fact]
    public void AcceptanceRate_WithZeroDraftTokens_ReturnsZero()
    {
        // Arrange
        var stats = new SpeculativeStats
        {
            TotalTokens = 5,
            DraftTokens = 0,
            AcceptedTokens = 0,
            TargetTokens = 5,
            ElapsedMilliseconds = 500
        };

        // Act
        var rate = stats.AcceptanceRate;

        // Assert
        rate.Should().Be(0);
    }

    [Fact]
    public void TokensPerSecond_WithElapsedTime_ReturnsCorrectThroughput()
    {
        // Arrange
        var stats = new SpeculativeStats
        {
            TotalTokens = 100,
            DraftTokens = 80,
            AcceptedTokens = 60,
            TargetTokens = 40,
            ElapsedMilliseconds = 2000
        };

        // Act
        var tps = stats.TokensPerSecond;

        // Assert
        tps.Should().Be(50); // 100 tokens / 2 seconds
    }

    [Fact]
    public void TokensPerSecond_WithZeroElapsed_ReturnsZero()
    {
        // Arrange
        var stats = new SpeculativeStats
        {
            TotalTokens = 10,
            DraftTokens = 8,
            AcceptedTokens = 6,
            TargetTokens = 4,
            ElapsedMilliseconds = 0
        };

        // Act
        var tps = stats.TokensPerSecond;

        // Assert
        tps.Should().Be(0);
    }

    [Fact]
    public void GetSummary_ReturnsFormattedString()
    {
        // Arrange
        var stats = new SpeculativeStats
        {
            TotalTokens = 100,
            DraftTokens = 80,
            AcceptedTokens = 60,
            TargetTokens = 40,
            ElapsedMilliseconds = 2000
        };

        // Act
        var summary = stats.GetSummary();

        // Assert
        summary.Should().Contain("Total Tokens: 100");
        summary.Should().Contain("Draft/Accepted: 80/60");
        summary.Should().Contain("Target Tokens: 40");
        summary.Should().Contain("Time: 2000ms");
    }
}

public class SpeculativeDecoderBuilderTests
{
    [Fact]
    public void Build_WithoutDraftModel_ThrowsInvalidOperationException()
    {
        // Arrange
        var builder = SpeculativeDecoderBuilder.Create()
            .WithTargetModel(Substitute.For<IGeneratorModel>());

        // Act & Assert
        var action = () => builder.Build();
        action.Should().Throw<InvalidOperationException>()
            .WithMessage("*Draft model*");
    }

    [Fact]
    public void Build_WithoutTargetModel_ThrowsInvalidOperationException()
    {
        // Arrange
        var builder = SpeculativeDecoderBuilder.Create()
            .WithDraftModel(Substitute.For<IGeneratorModel>());

        // Act & Assert
        var action = () => builder.Build();
        action.Should().Throw<InvalidOperationException>()
            .WithMessage("*Target model*");
    }

    [Fact]
    public void Build_WithBothModels_ReturnsDecoder()
    {
        // Arrange
        var draftModel = Substitute.For<IGeneratorModel>();
        var targetModel = Substitute.For<IGeneratorModel>();

        var builder = SpeculativeDecoderBuilder.Create()
            .WithDraftModel(draftModel)
            .WithTargetModel(targetModel);

        // Act
        using var decoder = builder.Build();

        // Assert
        decoder.Should().NotBeNull();
    }

    [Fact]
    public void WithSpeculationLength_SetsLength()
    {
        // Arrange
        var draftModel = Substitute.For<IGeneratorModel>();
        var targetModel = Substitute.For<IGeneratorModel>();

        // Act
        using var decoder = SpeculativeDecoderBuilder.Create()
            .WithDraftModel(draftModel)
            .WithTargetModel(targetModel)
            .WithSpeculationLength(8)
            .Build();

        // Assert
        decoder.SpeculationLength.Should().Be(8);
    }

    [Fact]
    public void FluentApi_AllMethodsChainable()
    {
        // Arrange
        var draftModel = Substitute.For<IGeneratorModel>();
        var targetModel = Substitute.For<IGeneratorModel>();

        // Act
        using var decoder = SpeculativeDecoderBuilder.Create()
            .WithDraftModel(draftModel)
            .WithTargetModel(targetModel)
            .WithSpeculationLength(6)
            .WithAdaptiveSpeculation(true)
            .WithMinAcceptanceRate(0.6)
            .WithDraftTemperature(0.3f)
            .Build();

        // Assert
        decoder.Should().NotBeNull();
        decoder.SpeculationLength.Should().Be(6);
    }
}

public class SpeculativeDecodingOptionsTests
{
    [Fact]
    public void Default_SpeculationLength_Is5()
    {
        // Act
        var options = new SpeculativeDecodingOptions();

        // Assert
        options.SpeculationLength.Should().Be(5);
    }

    [Fact]
    public void Default_MinAcceptanceRate_Is0Point5()
    {
        // Act
        var options = new SpeculativeDecodingOptions();

        // Assert
        options.MinAcceptanceRate.Should().Be(0.5);
    }

    [Fact]
    public void Default_AdaptiveSpeculation_IsTrue()
    {
        // Act
        var options = new SpeculativeDecodingOptions();

        // Assert
        options.AdaptiveSpeculation.Should().BeTrue();
    }

    [Fact]
    public void Default_DraftTemperature_IsNull()
    {
        // Act
        var options = new SpeculativeDecodingOptions();

        // Assert
        options.DraftTemperature.Should().BeNull();
    }
}
