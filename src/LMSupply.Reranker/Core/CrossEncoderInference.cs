using LMSupply.Download;
using LMSupply.Inference;
using LMSupply.Reranker.Models;
using LMSupply.Text;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace LMSupply.Reranker.Core;

/// <summary>
/// Handles ONNX Runtime inference for cross-encoder models. Session ownership, gating,
/// cancellation translation and provider fallback (crash and timeout) live in
/// <see cref="RecoverableOnnxSession"/>; this class only knows the cross-encoder's tensor shape.
/// </summary>
internal sealed class CrossEncoderInference : IDisposable
{
    private readonly RecoverableOnnxSession _session;
    private readonly string _outputName;
    private readonly OutputShape _outputShape;
    private readonly bool _hasTokenTypeIds;

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

    private CrossEncoderInference(
        RecoverableOnnxSession session,
        string outputName,
        OutputShape outputShape,
        bool hasTokenTypeIds)
    {
        _session = session;
        _outputName = outputName;
        _outputShape = outputShape;
        _hasTokenTypeIds = hasTokenTypeIds;
    }

    /// <summary>
    /// Creates an inference engine from an ONNX model file asynchronously.
    /// This method ensures runtime binaries are available before creating the session.
    /// </summary>
    /// <param name="modelPath">Path to the ONNX model.</param>
    /// <param name="modelInfo">Model information.</param>
    /// <param name="provider">Execution provider for inference.</param>
    /// <param name="threadCount">Number of inference threads.</param>
    /// <param name="progress">Optional progress reporter for binary downloads.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>Configured inference engine.</returns>
    public static async Task<CrossEncoderInference> CreateAsync(
        string modelPath,
        ModelInfo modelInfo,
        ExecutionProvider provider = ExecutionProvider.Auto,
        int? threadCount = null,
        IProgress<DownloadProgress>? progress = null,
        CancellationToken cancellationToken = default)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(modelPath);

        if (!File.Exists(modelPath))
        {
            throw new FileNotFoundException($"Model file not found: {modelPath}", modelPath);
        }

        Action<SessionOptions> configure = options => ConfigureSessionOptions(options, threadCount);

        try
        {
            var result = await OnnxSessionFactory.CreateWithInfoAsync(
                modelPath,
                provider,
                configure,
                progress,
                cancellationToken);

            var session = RecoverableOnnxSession.FromResult(result, modelPath, configure, logPrefix: "[CrossEncoderInference]");
            return CreateFromSession(session, modelInfo);
        }
        catch (Exception ex) when (ex is not FileNotFoundException)
        {
            throw new InferenceException($"Failed to load ONNX model from {modelPath}", ex);
        }
    }

    /// <summary>
    /// Creates an inference engine from an ONNX model file.
    /// Note: This assumes runtime binaries are already available. For lazy loading, use CreateAsync.
    /// </summary>
    /// <param name="modelPath">Path to the ONNX model.</param>
    /// <param name="modelInfo">Model information.</param>
    /// <param name="provider">Execution provider for inference.</param>
    /// <param name="threadCount">Number of inference threads.</param>
    /// <returns>Configured inference engine.</returns>
    public static CrossEncoderInference Create(
        string modelPath,
        ModelInfo modelInfo,
        ExecutionProvider provider = ExecutionProvider.Auto,
        int? threadCount = null)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(modelPath);

        if (!File.Exists(modelPath))
        {
            throw new FileNotFoundException($"Model file not found: {modelPath}", modelPath);
        }

        Action<SessionOptions> configure = options => ConfigureSessionOptions(options, threadCount);

        try
        {
            // Resolve the providers actually active on the session the same way the async path does,
            // instead of reporting CPU regardless of what was requested — the diagnostics were wrong
            // and, worse, a CPU report would disable provider recovery on a GPU session.
            var inferenceSession = OnnxSessionFactory.Create(modelPath, provider, configure, out var gpuEpAppended);
            var activeProviders = OnnxSessionFactory.ResolveActiveProviders(provider, gpuEpAppended);
            var isGpuActive = activeProviders.Any(p => p != "CPUExecutionProvider");

            var session = new RecoverableOnnxSession(
                inferenceSession, activeProviders, isGpuActive, provider, modelPath, configure,
                logPrefix: "[CrossEncoderInference]");
            return CreateFromSession(session, modelInfo);
        }
        catch (Exception ex)
        {
            throw new InferenceException($"Failed to load ONNX model from {modelPath}", ex);
        }
    }

    private static CrossEncoderInference CreateFromSession(RecoverableOnnxSession session, ModelInfo modelInfo)
    {
        var hasTokenTypeIds = session.Session.InputMetadata.ContainsKey("token_type_ids");
        var outputName = session.Session.OutputMetadata.Keys.First();

        return new CrossEncoderInference(session, outputName, modelInfo.OutputShape, hasTokenTypeIds);
    }

    private static void ConfigureSessionOptions(SessionOptions options, int? threadCount)
    {
        options.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
        options.ExecutionMode = ExecutionMode.ORT_PARALLEL;

        // Set thread count
        var threads = threadCount ?? Environment.ProcessorCount;
        options.IntraOpNumThreads = threads;
        options.InterOpNumThreads = Math.Max(1, threads / 2);
    }

    /// <summary>
    /// Runs inference on a batch of encoded inputs under <see cref="CancellableInference"/>'s
    /// bound, moving to the next execution provider and retrying once if the provider crashes or
    /// hangs (see <see cref="RecoverableOnnxSession.RunWithRecoveryAsync{T}"/>).
    /// </summary>
    /// <param name="batch">Encoded batch of query-document pairs.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>Array of relevance scores (0-1).</returns>
    public async Task<float[]> InferAsync(EncodedPairBatch batch, CancellationToken cancellationToken = default)
    {
        var inputs = BuildInputs(batch);
        try
        {
            return await _session.RunWithRecoveryAsync(
                (session, runOptions) => RunAndExtract(session, runOptions, inputs, batch.BatchSize),
                cancellationToken: cancellationToken);
        }
        catch (Exception ex) when (ex is not OperationCanceledException and not InferenceTimeoutException)
        {
            throw new InferenceException("Model inference failed", ex);
        }
    }

    /// <summary>
    /// Runs inference on a batch of encoded inputs synchronously on the calling thread. Provider
    /// crash fallback applies; the hang bound does not — prefer <see cref="InferAsync"/>.
    /// </summary>
    /// <param name="batch">Encoded batch of query-document pairs.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>Array of relevance scores (0-1).</returns>
    public float[] Infer(EncodedPairBatch batch, CancellationToken cancellationToken = default)
    {
        var inputs = BuildInputs(batch);
        try
        {
            return _session.Run(
                (session, runOptions) => RunAndExtract(session, runOptions, inputs, batch.BatchSize),
                cancellationToken);
        }
        catch (Exception ex) when (ex is not OperationCanceledException)
        {
            throw new InferenceException("Model inference failed", ex);
        }
    }

    private List<NamedOnnxValue> BuildInputs(EncodedPairBatch batch)
    {
        var inputIds = CreateTensor(batch.GetFlatInputIds(), batch.BatchSize, batch.SequenceLength);
        var attentionMask = CreateTensor(batch.GetFlatAttentionMask(), batch.BatchSize, batch.SequenceLength);

        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("input_ids", inputIds),
            NamedOnnxValue.CreateFromTensor("attention_mask", attentionMask)
        };

        if (_hasTokenTypeIds)
        {
            var tokenTypeIds = CreateTensor(batch.GetFlatTokenTypeIds(), batch.BatchSize, batch.SequenceLength);
            inputs.Add(NamedOnnxValue.CreateFromTensor("token_type_ids", tokenTypeIds));
        }

        return inputs;
    }

    private float[] RunAndExtract(InferenceSession session, RunOptions runOptions, List<NamedOnnxValue> inputs, int batchSize)
    {
        using var results = session.Run(inputs, [_outputName], runOptions);
        return ExtractScores(results[0].AsTensor<float>(), batchSize);
    }

    private float[] ExtractScores(Tensor<float> outputTensor, int batchSize)
    {
        var scores = new float[batchSize];
        var dimensions = outputTensor.Dimensions.ToArray();

        switch (_outputShape)
        {
            case OutputShape.SingleLogit:
                // Shape: [batch_size, 1] or [batch_size]
                for (var i = 0; i < batchSize; i++)
                {
                    var logit = dimensions.Length == 1
                        ? outputTensor[i]
                        : outputTensor[i, 0];
                    scores[i] = ScoreNormalizer.Sigmoid(logit);
                }
                break;

            case OutputShape.BinaryClassification:
                // Shape: [batch_size, 2] - use softmax on positive class
                for (var i = 0; i < batchSize; i++)
                {
                    var logit0 = outputTensor[i, 0];
                    var logit1 = outputTensor[i, 1];
                    scores[i] = ScoreNormalizer.SoftmaxPositive(logit0, logit1);
                }
                break;

            case OutputShape.FlatLogit:
                // Shape: [batch_size]
                for (var i = 0; i < batchSize; i++)
                {
                    scores[i] = ScoreNormalizer.Sigmoid(outputTensor[i]);
                }
                break;
        }

        return scores;
    }

    private static DenseTensor<long> CreateTensor(long[] data, int batchSize, int sequenceLength)
    {
        return new DenseTensor<long>(data, [batchSize, sequenceLength]);
    }

    public void Dispose()
    {
        _session.Dispose();
    }
}
