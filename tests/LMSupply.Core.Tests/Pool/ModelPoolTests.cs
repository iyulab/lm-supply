using AwesomeAssertions;
using LMSupply.Pool;

namespace LMSupply.Core.Tests.Pool;

/// <summary>
/// <see cref="ModelPoolOptions.MaxLoadedModels"/> was documented as the number of models the pool keeps
/// loaded, but the pool evicted by memory alone: with memory to spare, every model stayed resident however
/// many were loaded. These tests give the pool ample memory so that only the count can evict.
/// </summary>
public class ModelPoolTests
{
    [Fact]
    public async Task LoadingPastTheLimit_UnloadsTheLeastRecentlyUsedModel()
    {
        var loader = new FakeLoader();
        await using var pool = new ModelPool<FakeModel, object>(loader, Options(maxLoadedModels: 1));

        await pool.GetOrLoadAsync("a", cancellationToken: TestContext.Current.CancellationToken);
        await pool.GetOrLoadAsync("b", cancellationToken: TestContext.Current.CancellationToken);

        pool.LoadedModelCount.Should().Be(1);
        pool.IsLoaded("a").Should().BeFalse();
        pool.IsLoaded("b").Should().BeTrue();
        loader.Loaded["a"].Disposed.Should().BeTrue("an evicted model releases its resources");
    }

    [Fact]
    public async Task AModelUsedAgain_IsKeptOverOneLoadedLater()
    {
        var loader = new FakeLoader();
        await using var pool = new ModelPool<FakeModel, object>(loader, Options(maxLoadedModels: 2));
        var ct = TestContext.Current.CancellationToken;

        await pool.GetOrLoadAsync("a", cancellationToken: ct);
        await pool.GetOrLoadAsync("b", cancellationToken: ct);
        await pool.GetOrLoadAsync("a", cancellationToken: ct);   // a is now the most recently used
        await pool.GetOrLoadAsync("c", cancellationToken: ct);

        pool.IsLoaded("a").Should().BeTrue();
        pool.IsLoaded("b").Should().BeFalse("b is the least recently used");
        pool.IsLoaded("c").Should().BeTrue();
        loader.LoadCount("a").Should().Be(1, "a was served from the pool, not loaded twice");
    }

    [Fact]
    public async Task WithinTheLimit_NothingIsUnloaded()
    {
        var loader = new FakeLoader();
        await using var pool = new ModelPool<FakeModel, object>(loader, Options(maxLoadedModels: 2));

        await pool.GetOrLoadAsync("a", cancellationToken: TestContext.Current.CancellationToken);
        await pool.GetOrLoadAsync("b", cancellationToken: TestContext.Current.CancellationToken);

        pool.LoadedModelCount.Should().Be(2);
        loader.Loaded.Values.Should().OnlyContain(m => !m.Disposed);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(-1)]
    public void ALimitBelowOne_IsRejected(int maxLoadedModels)
    {
        var act = () => new ModelPool<FakeModel, object>(new FakeLoader(), Options(maxLoadedModels));

        act.Should().Throw<ArgumentOutOfRangeException>();
    }

    // Memory is fixed and ample, so the hardware probe is never consulted and never the reason to evict.
    private static ModelPoolOptions Options(int maxLoadedModels) => new()
    {
        MaxMemoryBytes = 1L << 40,
        MaxLoadedModels = maxLoadedModels,
    };

    private sealed class FakeModel : IAsyncDisposable
    {
        public bool Disposed { get; private set; }

        public ValueTask DisposeAsync()
        {
            Disposed = true;
            return ValueTask.CompletedTask;
        }
    }

    private sealed class FakeLoader : IModelLoader<FakeModel, object>
    {
        private readonly Dictionary<string, int> _loads = [];

        public Dictionary<string, FakeModel> Loaded { get; } = [];

        public int LoadCount(string modelId) => _loads.GetValueOrDefault(modelId);

        public Task<FakeModel> LoadAsync(string modelId, object? options, CancellationToken cancellationToken)
        {
            _loads[modelId] = LoadCount(modelId) + 1;
            var model = new FakeModel();
            Loaded[modelId] = model;
            return Task.FromResult(model);
        }

        public long EstimateMemoryBytes(string modelId, object? options) => 1;
    }
}
