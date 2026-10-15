using LMSupply.Generator.Abstractions;
using LMSupply.Pool;
using LMSupply.Core.Download;

namespace LMSupply.Generator.Internal;

internal sealed class GeneratorLoader : IModelLoader<IGeneratorModel, GeneratorOptions>
{
    private readonly IGeneratorModelFactory _factory;

    public GeneratorLoader(IGeneratorModelFactory factory)
    {
        _factory = factory;
    }

    public Task<IGeneratorModel> LoadAsync(string modelId, GeneratorOptions? options, CancellationToken ct)
        => _factory.LoadAsync(modelId, options, ct);

    public long EstimateMemoryBytes(string modelId, GeneratorOptions? options)
    {
        GeneratorModelRegistry.Default.TryResolve(modelId, out var modelInfo);
        if (modelInfo != null)
        {
            var config = modelInfo.GetMemoryConfig(options?.MaxContextLength);
            return MemoryEstimator.Calculate(config).TotalBytes;
        }

        return MemoryEstimator.Calculate(new ModelMemoryConfig
        {
            ParameterCount = 3_000_000_000,
            NumLayers = 32,
            HiddenSize = 2560,
            ContextLength = options?.MaxContextLength ?? 4096,
            Quantization = Quantization.Quant4
        }).TotalBytes;
    }
}
