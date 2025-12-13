using FluentAssertions;
using LocalAI.Generator.Abstractions;
using LocalAI.Generator.Models;
using NSubstitute;

namespace LocalAI.Generator.Tests;

public class MemoryAwareGeneratorTests
{
    [Fact]
    public void Constructor_NullInner_ThrowsArgumentNullException()
    {
        // Act & Assert
        var action = () => new MemoryAwareGenerator(null!);
        action.Should().Throw<ArgumentNullException>();
    }

    [Fact]
    public void ModelId_ReturnsInnerModelId()
    {
        // Arrange
        var inner = Substitute.For<IGeneratorModel>();
        inner.ModelId.Returns("test-model");
        var wrapper = new MemoryAwareGenerator(inner);

        // Act
        var result = wrapper.ModelId;

        // Assert
        result.Should().Be("test-model");
    }

    [Fact]
    public void MaxContextLength_ReturnsInnerMaxContextLength()
    {
        // Arrange
        var inner = Substitute.For<IGeneratorModel>();
        inner.MaxContextLength.Returns(8192);
        var wrapper = new MemoryAwareGenerator(inner);

        // Act
        var result = wrapper.MaxContextLength;

        // Assert
        result.Should().Be(8192);
    }

    [Fact]
    public void GetCurrentMemoryUsage_ReturnsPositiveValue()
    {
        // Arrange
        var inner = Substitute.For<IGeneratorModel>();
        var wrapper = new MemoryAwareGenerator(inner);

        // Act
        var result = wrapper.GetCurrentMemoryUsage();

        // Assert
        result.Should().BeGreaterThan(0);
    }

    [Fact]
    public void GetMemoryUsagePercent_WithLargeLimit_ReturnsSmallPercentage()
    {
        // Arrange
        var inner = Substitute.For<IGeneratorModel>();
        var options = MemoryAwareOptions.WithLimitGB(100); // 100GB - should be much more than used
        var wrapper = new MemoryAwareGenerator(inner, options);

        // Act
        var result = wrapper.GetMemoryUsagePercent();

        // Assert
        result.Should().BeLessThan(0.5); // Less than 50%
    }

    [Fact]
    public void TryReduceMemory_DoesNotThrow()
    {
        // Arrange
        var inner = Substitute.For<IGeneratorModel>();
        var wrapper = new MemoryAwareGenerator(inner);

        // Act & Assert
        var action = () => wrapper.TryReduceMemory();
        action.Should().NotThrow();
    }

    [Fact]
    public async Task GenerateCompleteAsync_CallsInner()
    {
        // Arrange
        var inner = Substitute.For<IGeneratorModel>();
        inner.GenerateCompleteAsync(Arg.Any<string>(), Arg.Any<GeneratorOptions>(), Arg.Any<CancellationToken>())
            .Returns("test response");

        var options = MemoryAwareOptions.WithLimitGB(100);
        var wrapper = new MemoryAwareGenerator(inner, options);

        // Act
        var result = await wrapper.GenerateCompleteAsync("prompt");

        // Assert
        result.Should().Be("test response");
        await inner.Received(1).GenerateCompleteAsync("prompt", Arg.Any<GeneratorOptions>(), Arg.Any<CancellationToken>());
    }

    [Fact]
    public async Task WarmupAsync_CallsInner()
    {
        // Arrange
        var inner = Substitute.For<IGeneratorModel>();
        var options = MemoryAwareOptions.WithLimitGB(100);
        var wrapper = new MemoryAwareGenerator(inner, options);

        // Act
        await wrapper.WarmupAsync();

        // Assert
        await inner.Received(1).WarmupAsync(Arg.Any<CancellationToken>());
    }

    [Fact]
    public void GetModelInfo_ReturnsInnerModelInfo()
    {
        // Arrange
        var inner = Substitute.For<IGeneratorModel>();
        var expectedInfo = new GeneratorModelInfo("id", "path", 4096, "phi3", "cpu");
        inner.GetModelInfo().Returns(expectedInfo);
        var wrapper = new MemoryAwareGenerator(inner);

        // Act
        var result = wrapper.GetModelInfo();

        // Assert
        result.Should().Be(expectedInfo);
    }

    [Fact]
    public async Task DisposeAsync_DisposesInner()
    {
        // Arrange
        var inner = Substitute.For<IGeneratorModel>();
        var wrapper = new MemoryAwareGenerator(inner);

        // Act
        await wrapper.DisposeAsync();

        // Assert
        await inner.Received(1).DisposeAsync();
    }

    [Fact]
    public async Task DisposeAsync_CanBeCalledMultipleTimes()
    {
        // Arrange
        var inner = Substitute.For<IGeneratorModel>();
        var wrapper = new MemoryAwareGenerator(inner);

        // Act
        await wrapper.DisposeAsync();
        await wrapper.DisposeAsync();

        // Assert - should only dispose inner once
        await inner.Received(1).DisposeAsync();
    }

    [Fact]
    public async Task GenerateCompleteAsync_WhenDisposed_ThrowsObjectDisposedException()
    {
        // Arrange
        var inner = Substitute.For<IGeneratorModel>();
        var wrapper = new MemoryAwareGenerator(inner);
        await wrapper.DisposeAsync();

        // Act & Assert
        var action = () => wrapper.GenerateCompleteAsync("prompt");
        await action.Should().ThrowAsync<ObjectDisposedException>();
    }
}

public class MemoryAwareOptionsTests
{
    [Fact]
    public void WithLimitMB_SetsCorrectBytes()
    {
        // Act
        var options = MemoryAwareOptions.WithLimitMB(1024);

        // Assert
        options.MaxMemoryBytes.Should().Be(1024L * 1024 * 1024);
    }

    [Fact]
    public void WithLimitGB_SetsCorrectBytes()
    {
        // Act
        var options = MemoryAwareOptions.WithLimitGB(8);

        // Assert
        options.MaxMemoryBytes.Should().Be(8L * 1024 * 1024 * 1024);
    }

    [Fact]
    public void Default_Has4GBLimit()
    {
        // Act
        var options = new MemoryAwareOptions();

        // Assert
        options.MaxMemoryBytes.Should().Be(4L * 1024 * 1024 * 1024);
    }

    [Fact]
    public void Default_Has80PercentWarningThreshold()
    {
        // Act
        var options = new MemoryAwareOptions();

        // Assert
        options.WarningThreshold.Should().Be(0.80);
    }

    [Fact]
    public void Default_Has95PercentCriticalThreshold()
    {
        // Act
        var options = new MemoryAwareOptions();

        // Assert
        options.CriticalThreshold.Should().Be(0.95);
    }
}
