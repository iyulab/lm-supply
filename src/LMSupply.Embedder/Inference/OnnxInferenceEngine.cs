using LMSupply.Download;
using LMSupply.Inference;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace LMSupply.Embedder.Inference;

/// <summary>
/// ONNX Runtime inference engine for embedding models.
/// </summary>
internal sealed class OnnxInferenceEngine : IDisposable
{
    private readonly InferenceSession _session;
    private readonly bool _hasTokenTypeIds;
    private readonly string _outputName;

    public int HiddenSize { get; }

    private OnnxInferenceEngine(InferenceSession session, int hiddenSize, bool hasTokenTypeIds, string outputName)
    {
        _session = session;
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

        var session = await OnnxSessionFactory.CreateAsync(
            modelPath,
            provider,
            ConfigureOptions,
            progress,
            cancellationToken);

        return CreateFromSession(session);
    }

    /// <summary>
    /// Creates an inference engine from an ONNX model file.
    /// Note: This assumes runtime binaries are already available. For lazy loading, use CreateAsync.
    /// </summary>
    public static OnnxInferenceEngine Create(string modelPath, ExecutionProvider provider)
    {
        if (!File.Exists(modelPath))
            throw new ModelNotFoundException("Model file not found", modelPath);

        var session = OnnxSessionFactory.Create(modelPath, provider, ConfigureOptions);
        return CreateFromSession(session);
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

    private static OnnxInferenceEngine CreateFromSession(InferenceSession session)
    {
        // Detect model configuration from metadata
        var inputNames = session.InputMetadata.Keys.ToHashSet();
        bool hasTokenTypeIds = inputNames.Contains("token_type_ids");

        // Get output name and hidden size
        var outputMeta = session.OutputMetadata.First();
        string outputName = outputMeta.Key;
        int hiddenSize = (int)outputMeta.Value.Dimensions[^1]; // Last dimension is hidden size

        return new OnnxInferenceEngine(session, hiddenSize, hasTokenTypeIds, outputName);
    }

    /// <summary>
    /// Runs inference for a single sequence.
    /// </summary>
    public float[] RunInference(long[] inputIds, long[] attentionMask)
    {
        int seqLength = inputIds.Length;

        // Create input tensors
        var inputIdsTensor = new DenseTensor<long>(inputIds, [1, seqLength]);
        var attentionMaskTensor = new DenseTensor<long>(attentionMask, [1, seqLength]);

        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("input_ids", inputIdsTensor),
            NamedOnnxValue.CreateFromTensor("attention_mask", attentionMaskTensor)
        };

        // Add token_type_ids if model expects it
        if (_hasTokenTypeIds)
        {
            var tokenTypeIds = new long[seqLength]; // All zeros
            var tokenTypeIdsTensor = new DenseTensor<long>(tokenTypeIds, [1, seqLength]);
            inputs.Add(NamedOnnxValue.CreateFromTensor("token_type_ids", tokenTypeIdsTensor));
        }

        // Run inference
        using var results = _session.Run(inputs);
        var output = results.First().AsTensor<float>();

        // Output shape: [1, seqLength, hiddenSize]
        // Copy to flat array
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
    }

    /// <summary>
    /// Runs batch inference for multiple sequences (sequential).
    /// </summary>
    public float[][] RunBatchInference(long[][] inputIds, long[][] attentionMasks)
    {
        int batchSize = inputIds.Length;
        var results = new float[batchSize][];

        for (int i = 0; i < batchSize; i++)
        {
            results[i] = RunInference(inputIds[i], attentionMasks[i]);
        }

        return results;
    }

    /// <summary>
    /// Runs batch inference with parallel processing for CPU-bound workloads.
    /// </summary>
    public float[][] RunBatchInferenceParallel(long[][] inputIds, long[][] attentionMasks)
    {
        int batchSize = inputIds.Length;
        var results = new float[batchSize][];

        // Use parallel processing for large batches
        if (batchSize > 4)
        {
            var parallelOptions = new ParallelOptions
            {
                MaxDegreeOfParallelism = Math.Min(Environment.ProcessorCount, batchSize)
            };

            Parallel.For(0, batchSize, parallelOptions, i =>
            {
                results[i] = RunInference(inputIds[i], attentionMasks[i]);
            });
        }
        else
        {
            // Sequential for small batches (avoid parallel overhead)
            for (int i = 0; i < batchSize; i++)
            {
                results[i] = RunInference(inputIds[i], attentionMasks[i]);
            }
        }

        return results;
    }

    public void Dispose()
    {
        _session?.Dispose();
    }
}
