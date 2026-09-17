using LMSupply.Download;
using LMSupply.Exceptions;
using LMSupply.Reranker.Infrastructure;
using LMSupply.Reranker.Models;
using Xunit;

namespace LMSupply.Reranker.Tests;

/// <summary>
/// With auto-download disabled the model manager no longer checks the cache itself: it hands the
/// downloader the local-files-only mode, the same one every other module now uses for
/// <c>DisableAutoDownload</c>. The messages are asserted because they tell the two paths apart — a
/// network attempt for this made-up repository would fail differently (a download error, or no network).
/// </summary>
public sealed class ModelManagerOfflineTests : IDisposable
{
    private readonly string _cacheDir = Path.Combine(Path.GetTempPath(), "lmsupply-reranker-offline-" + Guid.NewGuid().ToString("N"));

    private static readonly ModelInfo Model = new()
    {
        Id = "acme/reranker-offline",
        AliasName = "offline-test",
        DisplayName = "Offline test",
        Parameters = 1,
        MaxSequenceLength = 16,
        SizeBytes = 1,
        OnnxFile = "model.onnx",
        TokenizerFile = "tokenizer.json",
        Description = "fixture",
    };

    public void Dispose()
    {
        if (Directory.Exists(_cacheDir))
            Directory.Delete(_cacheDir, recursive: true);
    }

    [Fact]
    public async Task NothingCached_ThrowsModelNotFound_WithoutTryingToDownload()
    {
        using var manager = new ModelManager(_cacheDir, autoDownload: false);

        var ex = await Assert.ThrowsAsync<ModelNotFoundException>(() =>
            manager.EnsureModelAsync(Model, cancellationToken: TestContext.Current.CancellationToken));

        Assert.Contains("downloads are disabled", ex.Message, StringComparison.Ordinal);
    }

    [Fact]
    public async Task GraphCachedButNoTokenizer_ThrowsModelNotFound()
    {
        Cache(Model.OnnxFile, "graph");
        using var manager = new ModelManager(_cacheDir, autoDownload: false);

        var ex = await Assert.ThrowsAsync<ModelNotFoundException>(() =>
            manager.EnsureModelAsync(Model, cancellationToken: TestContext.Current.CancellationToken));

        Assert.Contains(Model.TokenizerFile, ex.Message, StringComparison.Ordinal);
    }

    [Fact]
    public async Task EverythingCached_ReturnsThePaths()
    {
        Cache(Model.OnnxFile, "graph");
        Cache(Model.TokenizerFile, "{}");
        using var manager = new ModelManager(_cacheDir, autoDownload: false);

        var paths = await manager.EnsureModelAsync(Model, cancellationToken: TestContext.Current.CancellationToken);

        Assert.True(File.Exists(paths.ModelPath));
        Assert.True(File.Exists(paths.TokenizerPath));
    }

    private static readonly ModelInfo ExternalWeightsModel = Model with { OnnxFile = "onnx/model.onnx", OnnxDataFile = "onnx/model.onnx_data" };

    [Fact]
    public async Task GraphAndTokenizerCached_ButDeclaredExternalWeightsMissing_IsNotACachedModel()
    {
        // A cache a release that did not fetch external weights left behind: the graph shell loads nothing on its own.
        Cache(ExternalWeightsModel.OnnxFile, "graph");
        Cache(ExternalWeightsModel.TokenizerFile, "{}");
        using var manager = new ModelManager(_cacheDir, autoDownload: false);

        Assert.Null(manager.GetCachedModel(ExternalWeightsModel));
        var ex = await Assert.ThrowsAsync<ModelNotFoundException>(() =>
            manager.EnsureModelAsync(ExternalWeightsModel, cancellationToken: TestContext.Current.CancellationToken));
        Assert.Contains("model.onnx_data", ex.Message, StringComparison.Ordinal);
    }

    [Fact]
    public async Task GraphTokenizerAndExternalWeightsCached_ReturnsThePaths()
    {
        Cache(ExternalWeightsModel.OnnxFile, "graph");
        Cache(ExternalWeightsModel.OnnxDataFile!, "weights");
        Cache(ExternalWeightsModel.TokenizerFile, "{}");
        using var manager = new ModelManager(_cacheDir, autoDownload: false);

        Assert.NotNull(manager.GetCachedModel(ExternalWeightsModel));
        var paths = await manager.EnsureModelAsync(ExternalWeightsModel, cancellationToken: TestContext.Current.CancellationToken);
        Assert.True(File.Exists(paths.ModelPath));
    }

    [Fact]
    public async Task TokenizerFromAnotherRepository_IsReadFromThatRepositorysCache()
    {
        var split = Model with { TokenizerRepoId = "acme/reranker-tokenizer" };
        Cache(split.OnnxFile, "graph");
        CacheIn("acme/reranker-tokenizer", split.TokenizerFile, "{}");
        using var manager = new ModelManager(_cacheDir, autoDownload: false);

        var paths = await manager.EnsureModelAsync(split, cancellationToken: TestContext.Current.CancellationToken);

        Assert.Equal(CacheManager.GetModelFilePath(_cacheDir, "acme/reranker-tokenizer", split.TokenizerFile), paths.TokenizerPath);
        Assert.NotNull(manager.GetCachedModel(split));
    }

    [Fact]
    public async Task TokenizerFromAnotherRepository_NotCachedThere_IsNotFound_EvenWhenTheModelRepositoryHasOne()
    {
        var split = Model with { TokenizerRepoId = "acme/reranker-tokenizer" };
        Cache(split.OnnxFile, "graph");
        Cache(split.TokenizerFile, "{}");
        using var manager = new ModelManager(_cacheDir, autoDownload: false);

        Assert.Null(manager.GetCachedModel(split));
        var ex = await Assert.ThrowsAsync<ModelNotFoundException>(() =>
            manager.EnsureModelAsync(split, cancellationToken: TestContext.Current.CancellationToken));
        Assert.Contains("acme/reranker-tokenizer", ex.Message, StringComparison.Ordinal);
    }

    private void CacheIn(string repo, string file, string content)
    {
        var path = CacheManager.GetModelFilePath(_cacheDir, repo, file);
        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        File.WriteAllText(path, content);
    }

    [Fact]
    public void EveryBuiltInModelOver2GB_DeclaresItsExternalWeights()
    {
        // ONNX protobuf cannot hold more than 2 GB inline, so a built-in model that size ships its weights separately.
        Assert.All(DefaultModels.All.Where(m => m.SizeBytes >= 2_000_000_000), m => Assert.NotNull(m.OnnxDataFile));
    }

    private void Cache(string file, string content)
    {
        var path = CacheManager.GetModelFilePath(_cacheDir, Model.Id, file);
        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        File.WriteAllText(path, content);
    }
}
