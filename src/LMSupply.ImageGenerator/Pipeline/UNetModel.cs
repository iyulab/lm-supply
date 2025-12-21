using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace LMSupply.ImageGenerator.Pipeline;

/// <summary>
/// UNet model for LCM/Stable Diffusion latent denoising.
/// </summary>
internal sealed class UNetModel : IAsyncDisposable
{
    private readonly InferenceSession _session;
    private readonly string _sampleInput;
    private readonly string _timestepInput;
    private readonly string _encoderHiddenStatesInput;
    private readonly string? _timestepCondInput;
    private readonly int _timestepCondDim;
    private readonly string _output;
    private bool _disposed;

    /// <summary>
    /// Latent channels (typically 4 for SD/LCM).
    /// </summary>
    public int LatentChannels { get; }

    private UNetModel(
        InferenceSession session,
        string sampleInput,
        string timestepInput,
        string encoderHiddenStatesInput,
        string? timestepCondInput,
        int timestepCondDim,
        string output,
        int latentChannels)
    {
        _session = session;
        _sampleInput = sampleInput;
        _timestepInput = timestepInput;
        _encoderHiddenStatesInput = encoderHiddenStatesInput;
        _timestepCondInput = timestepCondInput;
        _timestepCondDim = timestepCondDim;
        _output = output;
        LatentChannels = latentChannels;
    }

    /// <summary>
    /// Loads the UNet model from the model directory.
    /// </summary>
    public static async Task<UNetModel> LoadAsync(
        string modelDir,
        SessionOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        var modelPath = FindUNetPath(modelDir);

        options ??= new SessionOptions();
        var session = await Task.Run(
            () => new InferenceSession(modelPath, options),
            cancellationToken);

        // Detect input/output names
        var inputs = session.InputMetadata;
        var outputs = session.OutputMetadata;

        // Common input patterns
        var sampleInput = FindInput(inputs, ["sample", "latent_model_input", "x"]);
        var timestepInput = FindInput(inputs, ["timestep", "t", "timesteps"]);
        var encoderInput = FindInput(inputs, ["encoder_hidden_states", "context", "text_embeds"]);
        var outputName = outputs.Keys.First();

        // Check for optional timestep_cond input (used by LCM models)
        string? timestepCondInput = null;
        var timestepCondDim = 256; // Default dimension for LCM guidance embedding
        if (inputs.TryGetValue("timestep_cond", out var condMeta))
        {
            timestepCondInput = "timestep_cond";
            // Get the dimension from metadata if available
            if (condMeta.Dimensions.Length > 1 && condMeta.Dimensions[1] > 0)
            {
                timestepCondDim = condMeta.Dimensions[1];
            }
        }

        // Determine latent channels from sample input shape
        // Note: Dynamic dimensions are represented as -1, default to 4 (standard for SD/LCM)
        var sampleShape = inputs[sampleInput].Dimensions;
        var latentChannels = sampleShape.Length > 1 && sampleShape[1] > 0 ? sampleShape[1] : 4;

        return new UNetModel(session, sampleInput, timestepInput, encoderInput,
            timestepCondInput, timestepCondDim, outputName, latentChannels);
    }

    /// <summary>
    /// Runs a single UNet forward pass.
    /// </summary>
    /// <param name="latents">Latent tensor [batch, channels, height, width].</param>
    /// <param name="timestep">Current timestep.</param>
    /// <param name="textEmbeddings">Text encoder output [batch, seqLen, hiddenSize].</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>Predicted noise tensor.</returns>
    public async Task<DenseTensor<float>> ForwardAsync(
        DenseTensor<float> latents,
        long timestep,
        DenseTensor<float> textEmbeddings,
        CancellationToken cancellationToken = default)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        var batchSize = latents.Dimensions[0];

        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor(_sampleInput, latents),
            NamedOnnxValue.CreateFromTensor(_timestepInput, new DenseTensor<long>(new[] { timestep }, [1])),
            NamedOnnxValue.CreateFromTensor(_encoderHiddenStatesInput, textEmbeddings)
        };

        // Add timestep_cond if required by this model (LCM guidance embedding)
        if (_timestepCondInput != null)
        {
            // Create a conditioning tensor filled with guidance scale embedding
            // For LCM, this is typically computed from the guidance scale
            // Using zeros as a neutral embedding for basic operation
            var condData = new float[batchSize * _timestepCondDim];
            var condTensor = new DenseTensor<float>(condData, [batchSize, _timestepCondDim]);
            inputs.Add(NamedOnnxValue.CreateFromTensor(_timestepCondInput, condTensor));
        }

        var result = await Task.Run(() =>
        {
            using var outputs = _session.Run(inputs);
            var output = outputs.First();

            var outputTensor = output.AsTensor<float>();
            var dims = outputTensor.Dimensions.ToArray();
            var data = outputTensor.ToArray();

            return new DenseTensor<float>(data, dims);
        }, cancellationToken);

        return result;
    }

    private static string FindUNetPath(string modelDir)
    {
        var candidates = new[]
        {
            Path.Combine(modelDir, "unet", "model.onnx"),
            Path.Combine(modelDir, "unet.onnx"),
            Path.Combine(modelDir, "lcm_unet.onnx")
        };

        foreach (var path in candidates)
        {
            if (File.Exists(path))
                return path;
        }

        var files = Directory.GetFiles(modelDir, "*unet*.onnx", SearchOption.AllDirectories);
        if (files.Length > 0)
            return files[0];

        throw new FileNotFoundException($"Could not find UNet ONNX file in: {modelDir}");
    }

    private static string FindInput(IReadOnlyDictionary<string, NodeMetadata> inputs, string[] candidates)
    {
        foreach (var candidate in candidates)
        {
            var match = inputs.Keys.FirstOrDefault(k =>
                k.Equals(candidate, StringComparison.OrdinalIgnoreCase));
            if (match != null)
                return match;
        }

        // Fallback: try partial match
        foreach (var candidate in candidates)
        {
            var match = inputs.Keys.FirstOrDefault(k =>
                k.Contains(candidate, StringComparison.OrdinalIgnoreCase));
            if (match != null)
                return match;
        }

        throw new InvalidOperationException($"Could not find input matching: {string.Join(", ", candidates)}");
    }

    public ValueTask DisposeAsync()
    {
        if (_disposed) return ValueTask.CompletedTask;
        _disposed = true;

        _session.Dispose();
        return ValueTask.CompletedTask;
    }
}
