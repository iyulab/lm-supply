using FluentAssertions;
using LMSupply.Generator.Abstractions;
using LMSupply.Generator.Models;
using NSubstitute;

namespace LMSupply.Generator.Tests;

public class GeneratorPoolTests : IAsyncDisposable
{
    private readonly IGeneratorModelFactory _mockFactory;
    private readonly GeneratorPool _pool;

    public GeneratorPoolTests()
    {
        _mockFactory = Substitute.For<IGeneratorModelFactory>();
        _pool = new GeneratorPool(_mockFactory, new GeneratorPoolOptions
        {
            MaxMemoryBytes = 8L * 1024 * 1024 * 1024, // 8GB
            MemorySafetyMargin = 0.1
        });
    }

    [Fact]
    public void Constructor_WithValidFactory_CreatesPool()
    {
        // Assert
        _pool.LoadedModelCount.Should().Be(0);
        _pool.AllocatedMemoryBytes.Should().Be(0);
    }

    [Fact]
    public void Constructor_WithNullFactory_ThrowsArgumentNullException()
    {
        // Act & Assert
        var action = () => new GeneratorPool(null!);
        action.Should().Throw<ArgumentNullException>();
    }

    [Fact]
    public async Task GetOrLoadAsync_LoadsModelOnFirstCall()
    {
        // Arrange
        var modelId = "microsoft/Phi-3.5-mini-instruct-onnx";
        var mockModel = CreateMockModel(modelId);

        _mockFactory.LoadAsync(modelId, Arg.Any<GeneratorOptions?>(), Arg.Any<CancellationToken>())
            .Returns(Task.FromResult(mockModel));

        // Act
        var model = await _pool.GetOrLoadAsync(modelId);

        // Assert
        model.Should().NotBeNull();
        _pool.LoadedModelCount.Should().Be(1);
        _pool.IsLoaded(modelId).Should().BeTrue();
    }

    [Fact]
    public async Task GetOrLoadAsync_ReturnsCachedModel()
    {
        // Arrange
        var modelId = "microsoft/Phi-3.5-mini-instruct-onnx";
        var mockModel = CreateMockModel(modelId);

        _mockFactory.LoadAsync(modelId, Arg.Any<GeneratorOptions?>(), Arg.Any<CancellationToken>())
            .Returns(Task.FromResult(mockModel));

        // Act
        var model1 = await _pool.GetOrLoadAsync(modelId);
        var model2 = await _pool.GetOrLoadAsync(modelId);

        // Assert
        model1.Should().BeSameAs(model2);
        await _mockFactory.Received(1).LoadAsync(modelId, Arg.Any<GeneratorOptions?>(), Arg.Any<CancellationToken>());
    }

    [Fact]
    public async Task UnloadAsync_RemovesModel()
    {
        // Arrange
        var modelId = "microsoft/Phi-3.5-mini-instruct-onnx";
        var mockModel = CreateMockModel(modelId);

        _mockFactory.LoadAsync(modelId, Arg.Any<GeneratorOptions?>(), Arg.Any<CancellationToken>())
            .Returns(Task.FromResult(mockModel));

        await _pool.GetOrLoadAsync(modelId);

        // Act
        await _pool.UnloadAsync(modelId);

        // Assert
        _pool.IsLoaded(modelId).Should().BeFalse();
        _pool.LoadedModelCount.Should().Be(0);
    }

    [Fact]
    public async Task UnloadAllAsync_RemovesAllModels()
    {
        // Arrange
        var model1Id = "microsoft/Phi-3.5-mini-instruct-onnx";
        var model2Id = "onnx-community/Llama-3.2-1B-Instruct-ONNX";

        var mockModel1 = CreateMockModel(model1Id);
        var mockModel2 = CreateMockModel(model2Id);

        _mockFactory.LoadAsync(
                Arg.Is<string>(s => s == model1Id),
                Arg.Any<GeneratorOptions?>(),
                Arg.Any<CancellationToken>())
            .Returns(Task.FromResult(mockModel1));

        _mockFactory.LoadAsync(
                Arg.Is<string>(s => s == model2Id),
                Arg.Any<GeneratorOptions?>(),
                Arg.Any<CancellationToken>())
            .Returns(Task.FromResult(mockModel2));

        await _pool.GetOrLoadAsync(model1Id);
        await _pool.GetOrLoadAsync(model2Id);

        // Act
        await _pool.UnloadAllAsync();

        // Assert
        _pool.LoadedModelCount.Should().Be(0);
    }

    [Fact]
    public async Task GetLoadedModels_ReturnsCorrectInfo()
    {
        // Arrange
        var modelId = "microsoft/Phi-3.5-mini-instruct-onnx";
        var mockModel = CreateMockModel(modelId);

        _mockFactory.LoadAsync(modelId, Arg.Any<GeneratorOptions?>(), Arg.Any<CancellationToken>())
            .Returns(Task.FromResult(mockModel));

        await _pool.GetOrLoadAsync(modelId);

        // Act
        var loaded = _pool.GetLoadedModels();

        // Assert
        loaded.Should().HaveCount(1);
        loaded[0].ModelId.Should().Be(modelId);
        loaded[0].AllocatedMemoryBytes.Should().BeGreaterThan(0);
    }

    [Fact]
    public async Task GetOrLoadAsync_WhenDisposed_ThrowsObjectDisposedException()
    {
        // Arrange - use separate pool to avoid affecting other tests
        var factory = Substitute.For<IGeneratorModelFactory>();
        var disposedPool = new GeneratorPool(factory, new GeneratorPoolOptions
        {
            MaxMemoryBytes = 8L * 1024 * 1024 * 1024
        });
        await disposedPool.DisposeAsync();

        // Act & Assert
        var action = () => disposedPool.GetOrLoadAsync("model");
        await action.Should().ThrowAsync<ObjectDisposedException>();
    }

    [Fact]
    public void AvailableMemoryBytes_ReturnsRemainingMemory()
    {
        // Assert
        _pool.AvailableMemoryBytes.Should().BeGreaterThan(0);
        _pool.AvailableMemoryBytes.Should().Be(8L * 1024 * 1024 * 1024 - _pool.AllocatedMemoryBytes);
    }

    private static IGeneratorModel CreateMockModel(string modelId)
    {
        var mock = Substitute.For<IGeneratorModel>();
        mock.ModelId.Returns(modelId);
        mock.MaxContextLength.Returns(4096);
        mock.DisposeAsync().Returns(ValueTask.CompletedTask);
        return mock;
    }

    public async ValueTask DisposeAsync()
    {
        await _pool.DisposeAsync();
    }
}
