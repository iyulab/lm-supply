using LMSupply.Pool;
using LMSupply.Synthesizer.Models;

namespace LMSupply.Synthesizer.Pool;

internal sealed class SynthesizerLoader : IModelLoader<ISynthesizerModel, SynthesizerOptions>
{
    public Task<ISynthesizerModel> LoadAsync(string modelId, SynthesizerOptions? options, CancellationToken ct)
        => LocalSynthesizer.LoadAsync(modelId, options, null, ct);

    public long EstimateMemoryBytes(string modelId, SynthesizerOptions? options)
    {
        if (SynthesizerModelRegistry.Default.TryResolve(modelId, out var info) && info is not null && info.SizeBytes > 0)
            return info.SizeBytes;

        return 200_000_000; // 200 MB default
    }
}
