using LMSupply.Download;
using LMSupply.Exceptions;
using LMSupply.Inference;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace LMSupply.Embedder.Inference;

/// <summary>
/// ONNX Runtime inference engine for embedding models. Session ownership, gating, cancellation
/// translation and provider fallback (crash and timeout) live in
/// <see cref="RecoverableOnnxSession"/>; this class only knows the embedding model's tensor shape.
/// </summary>
internal sealed class OnnxInferenceEngine : IDisposable
{
    private readonly RecoverableOnnxSession _session;
    private readonly bool _hasTokenTypeIds;
    private readonly string _outputName;

    public int HiddenSize { get; }

    /// <summary>
    /// Gets whether GPU acceleration is being used for inference.
    /// </summary>
    public bool IsGpuActive => _session.IsGpuActive;

    /// <summary>
    /// Gets the list of active execution providers.
    /// </summary>
    public IReadOnlyList<string> ActiveProviders => _session.ActiveProviders;

    /// <summary>
    /// Gets the execution provider that was requested.
    /// </summary>
    public ExecutionProvider RequestedProvider => _session.RequestedProvider;

    private OnnxInferenceEngine(
        InferenceSession session,
        int hiddenSize,
        bool hasTokenTypeIds,
        string outputName,
        bool isGpuProvider,
        bool isGpuActive,
        IReadOnlyList<string> activeProviders,
        ExecutionProvider requestedProvider,
        string modelPath)
    {
        _ = isGpuProvider; // kept for constructor-shape compatibility; derived from the provider by the session
        _session = new RecoverableOnnxSession(
            session, activeProviders, isGpuActive, requestedProvider, modelPath, ConfigureOptions,
            logPrefix: "[OnnxInferenceEngine]");
        HiddenSize = hiddenSize;
        _hasTokenTypeIds = hasTokenTypeIds;
        _outputName = outputName;
    }

    /// <summary>
    /// Creates an inference engine from an ONNX model file asynchronously.
    /// This method ensures runtime binaries are available before creating the session.
    /// </summary>
    /// <param name="modelPath">Path to the ONNX model file.</param>
    /// <param name="provider">The execution provider to use.</param>
    /// <param name="progress">Optional progress reporter for binary downloads.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>A configured inference engine.</returns>
    public static async Task<OnnxInferenceEngine> CreateAsync(
        string modelPath,
        ExecutionProvider provider,
        IProgress<DownloadProgress>? progress = null,
        CancellationToken cancellationToken = default)
    {
        if (!File.Exists(modelPath))
            throw new ModelNotFoundException("Model file not found", modelPath);

        var result = await OnnxSessionFactory.CreateWithInfoAsync(
            modelPath,
            provider,
            ConfigureOptions,
            progress,
            cancellationToken);

        return CreateFromSessionResult(result, modelPath);
    }

    /// <summary>
    /// Creates an inference engine from an ONNX model file.
    /// Note: This assumes runtime binaries are already available. For lazy loading, use CreateAsync.
    /// </summary>
    public static OnnxInferenceEngine Create(string modelPath, ExecutionProvider provider)
    {
        if (!File.Exists(modelPath))
            throw new ModelNotFoundException("Model file not found", modelPath);

        var session = OnnxSessionFactory.Create(modelPath, provider, ConfigureOptions, out var gpuEpAppended);
        var activeProviders = OnnxSessionFactory.ResolveActiveProviders(provider, gpuEpAppended);
        var isGpuActive = activeProviders.Any(p => p != "CPUExecutionProvider");

        return CreateFromSession(session, IsGpuProvider(provider), isGpuActive, activeProviders, provider, modelPath);
    }

    private static bool IsGpuProvider(ExecutionProvider provider)
    {
        return provider is ExecutionProvider.Cuda
            or ExecutionProvider.DirectML
            or ExecutionProvider.CoreML
            or ExecutionProvider.Auto; // Auto may select GPU, so treat as GPU for safety
    }

    private static void ConfigureOptions(SessionOptions options)
    {
        options.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
        options.ExecutionMode = ExecutionMode.ORT_SEQUENTIAL;
        options.EnableCpuMemArena = true;
        options.EnableMemoryPattern = true;
        options.IntraOpNumThreads = Environment.ProcessorCount;
        options.InterOpNumThreads = 1;
    }

    private static OnnxInferenceEngine CreateFromSessionResult(SessionCreationResult result, string modelPath)
    {
        return CreateFromSession(
            result.Session,
            IsGpuProvider(result.RequestedProvider),
            result.IsGpuActive,
            result.ActiveProviders,
            result.RequestedProvider,
            modelPath);
    }

    private static OnnxInferenceEngine CreateFromSession(
        InferenceSession session,
        bool isGpuProvider,
        bool isGpuActive,
        IReadOnlyList<string> activeProviders,
        ExecutionProvider requestedProvider,
        string modelPath)
    {
        // Detect model configuration from metadata
        var inputNames = session.InputMetadata.Keys.ToHashSet();
        bool hasTokenTypeIds = inputNames.Contains("token_type_ids");

        // Get output name and hidden size
        var outputMeta = session.OutputMetadata.First();
        string outputName = outputMeta.Key;
        int hiddenSize = (int)outputMeta.Value.Dimensions[^1]; // Last dimension is hidden size

        return new OnnxInferenceEngine(
            session,
            hiddenSize,
            hasTokenTypeIds,
            outputName,
            isGpuProvider,
            isGpuActive,
            activeProviders,
            requestedProvider,
            modelPath);
    }

    /// <summary>
    /// Runs inference for a single sequence.
    /// </summary>
    /// <param name="inputIds">Token ids for the sequence.</param>
    /// <param name="attentionMask">Attention mask for the sequence.</param>
    /// <param name="cancellationToken">
    /// Cancellation token. When cancelled, the native ONNX run is asked to terminate
    /// cooperatively via <see cref="RunOptions.Terminate"/> (best-effort — honored only between
    /// operators, so an intra-kernel hang such as a cold DirectML init may not be preempted).
    /// </param>
    public float[] RunInference(long[] inputIds, long[] attentionMask, CancellationToken cancellationToken = default)
    {
        int seqLength = inputIds.Length;

        var inputIdsTensor = new DenseTensor<long>(inputIds, [1, seqLength]);
        var attentionMaskTensor = new DenseTensor<long>(attentionMask, [1, seqLength]);

        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("input_ids", inputIdsTensor),
            NamedOnnxValue.CreateFromTensor("attention_mask", attentionMaskTensor)
        };

        if (_hasTokenTypeIds)
        {
            var tokenTypeIds = new long[seqLength]; // All zeros
            var tokenTypeIdsTensor = new DenseTensor<long>(tokenTypeIds, [1, seqLength]);
            inputs.Add(NamedOnnxValue.CreateFromTensor("token_type_ids", tokenTypeIdsTensor));
        }

        return _session.Run((session, runOptions) =>
        {
            using var results = session.Run(inputs, [_outputName], runOptions);
            var output = results[0].AsTensor<float>();

            // Output shape: [1, seqLength, hiddenSize] — copy to a flat array
            var outputArray = new float[seqLength * HiddenSize];
            int idx = 0;
            for (int seq = 0; seq < seqLength; seq++)
            {
                for (int dim = 0; dim < HiddenSize; dim++)
                {
                    outputArray[idx++] = output[0, seq, dim];
                }
            }

            return outputArray;
        }, cancellationToken);
    }

    /// <summary>
    /// After a run exceeded the caller-side timeout, moves inference to the next provider in the
    /// fallback chain. See <see cref="RecoverableOnnxSession.TryRecoverAfterTimeout"/>.
    /// </summary>
    public bool TryRecoverAfterTimeout(InferenceTimeoutException ex) => _session.TryRecoverAfterTimeout(ex);

    // Retained under this name because the fallback regression test drives it by reflection.
    private bool TryFallback(OnnxRuntimeException ex) => _session.TryFallback(ex);

    public void Dispose()
    {
        _session.Dispose();
    }
}
