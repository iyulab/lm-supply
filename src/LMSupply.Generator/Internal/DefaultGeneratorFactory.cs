using LMSupply.Generator.Abstractions;

namespace LMSupply.Generator.Internal;

/// <summary>
/// Default factory that delegates to LocalGenerator.LoadAsync for use with LocalGenerator.Pool.
/// </summary>
internal sealed class DefaultGeneratorFactory : IGeneratorModelFactory
{
    public static readonly DefaultGeneratorFactory Instance = new();

    public Task<IGeneratorModel> LoadAsync(
        string modelId,
        GeneratorOptions? options,
        CancellationToken cancellationToken)
        => LocalGenerator.LoadAsync(modelId, options, null, cancellationToken);

    public bool IsModelAvailable(string modelId) => false;

    public Task DownloadModelAsync(
        string modelId,
        IProgress<DownloadProgress>? progress = null,
        CancellationToken cancellationToken = default)
        => Task.CompletedTask;
}
