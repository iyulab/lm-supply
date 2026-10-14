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
        string modelPath,
        Action<SessionOptions> configure)
    {
        _ = isGpuProvider; // kept for constructor-shape compatibility; derived from the provider by the session
        _session = new RecoverableOnnxSession(
            session, activeProviders, isGpuActive, requestedProvider, modelPath, configure,
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
    /// <param name="options">Log level and thread count for the session; null keeps the ONNX Runtime defaults.</param>
    /// <param name="progress">Receives the runtime download session creation may make; <see langword="null"/> when nobody is listening.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>A configured inference engine.</returns>
    public static async Task<OnnxInferenceEngine> CreateAsync(
        string modelPath,
        ExecutionProvider provider,
        LMSupplyOptionsBase? options,
        IProgress<DownloadProgress>? progress,
        CancellationToken cancellationToken = default)
    {
        if (!File.Exists(modelPath))
            throw new ModelNotFoundException("Model file not found", modelPath);

        var configure = ConfigureFor(options);
        var result = await OnnxSessionFactory.CreateWithInfoAsync(
            modelPath,
            provider,
            configure,
            progress,
            cancellationToken);

        return CreateFromSessionResult(result, modelPath, configure);
    }

    /// <summary>
    /// Creates an inference engine from an ONNX model file.
    /// Note: This assumes runtime binaries are already available. For lazy loading, use CreateAsync.
    /// </summary>
    public static OnnxInferenceEngine Create(string modelPath, ExecutionProvider provider, LMSupplyOptionsBase? options = null)
    {
        if (!File.Exists(modelPath))
            throw new ModelNotFoundException("Model file not found", modelPath);

        var configure = ConfigureFor(options);
        var session = OnnxSessionFactory.Create(modelPath, provider, configure, out var gpuEpAppended);
        var activeProviders = OnnxSessionFactory.ResolveActiveProviders(provider, gpuEpAppended);
        var isGpuActive = activeProviders.Any(p => p != "CPUExecutionProvider");

        return CreateFromSession(session, IsGpuProvider(provider), isGpuActive, activeProviders, provider, modelPath, configure);
    }

    private static bool IsGpuProvider(ExecutionProvider provider)
    {
        return provider is ExecutionProvider.Cuda
            or ExecutionProvider.CoreML
            or ExecutionProvider.OpenVino
            or ExecutionProvider.Auto; // Auto may select GPU, so treat as GPU for safety
    }

    /// <summary>
    /// The session settings: the embedding graph's own (sequential, arenas) plus the caller's log level and threads. The
    /// thread count used to be fixed at one per logical core, whatever <see cref="LMSupplyOptionsBase.ThreadCount"/> said.
    /// </summary>
    private static Action<SessionOptions> ConfigureFor(LMSupplyOptionsBase? options) => sessionOptions =>
    {
        sessionOptions.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
        sessionOptions.ExecutionMode = ExecutionMode.ORT_SEQUENTIAL;
        sessionOptions.EnableCpuMemArena = true;
        sessionOptions.EnableMemoryPattern = true;
        if (options != null)
            sessionOptions.ApplyCommonOptions(options);
    };

    private static OnnxInferenceEngine CreateFromSessionResult(SessionCreationResult result, string modelPath, Action<SessionOptions> configure)
    {
        return CreateFromSession(
            result.Session,
            IsGpuProvider(result.RequestedProvider),
            result.IsGpuActive,
            result.ActiveProviders,
            result.RequestedProvider,
            modelPath,
            configure);
    }

    private static OnnxInferenceEngine CreateFromSession(
        InferenceSession session,
        bool isGpuProvider,
        bool isGpuActive,
        IReadOnlyList<string> activeProviders,
        ExecutionProvider requestedProvider,
        string modelPath,
        Action<SessionOptions> configure)
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
            modelPath,
            configure);
    }

    /// <summary>
    /// Runs inference for a single sequence.
    /// </summary>
    /// <param name="inputIds">Token ids for the sequence.</param>
    /// <param name="attentionMask">Attention mask for the sequence.</param>
    /// <param name="cancellationToken">
    /// Cancellation token. When cancelled, the native ONNX run is asked to terminate
    /// cooperatively via <see cref="RunOptions.Terminate"/> (best-effort — honored only between
    /// operators, so an intra-kernel hang such as a cold GPU kernel init may not be preempted).
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
            // Output shape: [1, seqLength, hiddenSize], row-major.
            return Flatten(results[0].AsTensor<float>(), seqLength * HiddenSize);
        }, cancellationToken);
    }

    /// <summary>
    /// Runs inference for a batch of sequences in one call: <paramref name="inputIds"/> and
    /// <paramref name="attentionMask"/> are <c>[batch, sequenceLength]</c>, padded to a common length (masked
    /// positions are 0 in the mask). Returns the token embeddings as one row-major
    /// <c>[batch, sequenceLength, hiddenSize]</c> array.
    /// </summary>
    public float[] RunBatchInference(long[,] inputIds, long[,] attentionMask, CancellationToken cancellationToken = default)
    {
        int batch = inputIds.GetLength(0);
        int seqLength = inputIds.GetLength(1);
        int[] shape = [batch, seqLength];

        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("input_ids", new DenseTensor<long>(Copy(inputIds), shape)),
            NamedOnnxValue.CreateFromTensor("attention_mask", new DenseTensor<long>(Copy(attentionMask), shape))
        };

        if (_hasTokenTypeIds)
            inputs.Add(NamedOnnxValue.CreateFromTensor("token_type_ids", new DenseTensor<long>(new long[batch * seqLength], shape)));

        return _session.Run((session, runOptions) =>
        {
            using var results = session.Run(inputs, [_outputName], runOptions);
            return Flatten(results[0].AsTensor<float>(), batch * seqLength * HiddenSize);
        }, cancellationToken);
    }

    private static long[] Copy(long[,] values)
    {
        var flat = new long[values.Length];
        Buffer.BlockCopy(values, 0, flat, 0, values.Length * sizeof(long));
        return flat;
    }

    /// <summary>
    /// The tensor's values in row-major order. A dense tensor is copied as one block; the per-element indexer
    /// (the previous copy) costs a multi-dimensional index computation for each of millions of values.
    /// </summary>
    private static float[] Flatten(Tensor<float> output, int expectedLength)
    {
        var values = output is DenseTensor<float> dense ? dense.Buffer.ToArray() : output.ToArray();
        if (values.Length != expectedLength)
            throw new InvalidOperationException(
                $"Embedding model output has {values.Length} values; expected {expectedLength} (batch x sequence x hidden size).");
        return values;
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
