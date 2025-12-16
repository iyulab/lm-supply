using LMSupply.Download;
using LMSupply.Inference;
using LMSupply.Reranker.Models;
using LMSupply.Text;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace LMSupply.Reranker.Core;

/// <summary>
/// Handles ONNX Runtime inference for cross-encoder models.
/// </summary>
internal sealed class CrossEncoderInference : IDisposable
{
    private readonly InferenceSession _session;
    private readonly string[] _inputNames;
    private readonly string _outputName;
    private readonly OutputShape _outputShape;
    private readonly bool _hasTokenTypeIds;
    private bool _disposed;

    /// <summary>
    /// Gets the ONNX Runtime session for diagnostics.
    /// </summary>
    public InferenceSession Session => _session;

    private CrossEncoderInference(
        InferenceSession session,
        string[] inputNames,
        string outputName,
        OutputShape outputShape,
        bool hasTokenTypeIds)
    {
        _session = session;
        _inputNames = inputNames;
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

        try
        {
            var session = await OnnxSessionFactory.CreateAsync(
                modelPath,
                provider,
                options => ConfigureSessionOptions(options, threadCount),
                progress,
                cancellationToken);

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

        var sessionOptions = CreateSessionOptions(provider, threadCount);

        try
        {
            var session = new InferenceSession(modelPath, sessionOptions);
            return CreateFromSession(session, modelInfo);
        }
        catch (Exception ex)
        {
            throw new InferenceException($"Failed to load ONNX model from {modelPath}", ex);
        }
    }

    private static CrossEncoderInference CreateFromSession(InferenceSession session, ModelInfo modelInfo)
    {
        // Detect input names
        var inputNames = session.InputMetadata.Keys.ToArray();
        var hasTokenTypeIds = inputNames.Contains("token_type_ids");

        // Detect output name
        var outputName = session.OutputMetadata.Keys.First();

        return new CrossEncoderInference(
            session,
            inputNames,
            outputName,
            modelInfo.OutputShape,
            hasTokenTypeIds);
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
    /// Runs inference on a batch of encoded inputs.
    /// </summary>
    /// <param name="batch">Encoded batch of query-document pairs.</param>
    /// <returns>Array of relevance scores (0-1).</returns>
    public float[] Infer(EncodedPairBatch batch)
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

        try
        {
            using var results = _session.Run(inputs);
            var outputTensor = results.First().AsTensor<float>();

            return ExtractScores(outputTensor, batch.BatchSize);
        }
        catch (Exception ex)
        {
            throw new InferenceException("Model inference failed", ex);
        }
    }

    /// <summary>
    /// Runs inference on a single query-document pair.
    /// </summary>
    /// <param name="encoded">Encoded input.</param>
    /// <returns>Relevance score (0-1).</returns>
    public float InferSingle(EncodedPair encoded)
    {
        var inputIds = CreateTensor(encoded.InputIds, 1, encoded.InputIds.Length);
        var attentionMask = CreateTensor(encoded.AttentionMask, 1, encoded.AttentionMask.Length);

        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("input_ids", inputIds),
            NamedOnnxValue.CreateFromTensor("attention_mask", attentionMask)
        };

        if (_hasTokenTypeIds)
        {
            var tokenTypeIds = CreateTensor(encoded.TokenTypeIds, 1, encoded.TokenTypeIds.Length);
            inputs.Add(NamedOnnxValue.CreateFromTensor("token_type_ids", tokenTypeIds));
        }

        try
        {
            using var results = _session.Run(inputs);
            var outputTensor = results.First().AsTensor<float>();

            return ExtractScores(outputTensor, 1)[0];
        }
        catch (Exception ex)
        {
            throw new InferenceException("Model inference failed", ex);
        }
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

    private static SessionOptions CreateSessionOptions(ExecutionProvider provider, int? threadCount)
    {
        var options = new SessionOptions();

        // Apply thread count and optimization settings
        ConfigureSessionOptions(options, threadCount);

        // Configure execution provider using shared factory
        OnnxSessionFactory.ConfigureExecutionProvider(options, provider);

        return options;
    }

    public void Dispose()
    {
        if (_disposed) return;
        _session.Dispose();
        _disposed = true;
    }
}
