using LMSupply.Inference;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace LMSupply.ImageGenerator.Pipeline;

/// <summary>
/// UNet model for LCM/Stable Diffusion latent denoising.
/// </summary>
internal sealed class UNetModel : IAsyncDisposable
{
    // Owns the ONNX session and recovers from a crashing or hanging execution provider by moving
    // to the next one in the fallback chain (see RecoverableOnnxSession). Each denoising step is one
    // bounded, recoverable run — the pipeline's latents live on the managed side, so a provider
    // switch between steps is transparent to the scheduler.
    private readonly RecoverableOnnxSession _session;
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
        RecoverableOnnxSession session,
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
    /// <param name="modelDir">Path to model directory.</param>
    /// <param name="provider">Execution provider to create the session with.</param>
    /// <param name="deviceId">GPU device index for CUDA/DirectML.</param>
    /// <param name="configureOptions">Session options to apply (log level, threads).</param>
    /// <param name="blacklist">Provider blacklist shared with the pipeline's other sessions.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    public static async Task<UNetModel> LoadAsync(
        string modelDir,
        ExecutionProvider provider,
        int deviceId,
        Action<SessionOptions>? configureOptions,
        ProviderBlacklist blacklist,
        CancellationToken cancellationToken = default)
    {
        var modelPath = FindUNetPath(modelDir);

        var result = await OnnxSessionFactory.CreateWithInfoAsync(
            modelPath, provider, skipProviders: null, configureOptions,
            cancellationToken: cancellationToken, deviceId: deviceId);

        // Detect input/output names (identical on every provider)
        var inputs = result.Session.InputMetadata;
        var outputs = result.Session.OutputMetadata;

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

        var session = RecoverableOnnxSession.FromResult(
            result, modelPath, configureOptions, logPrefix: "[UNetModel]", blacklist: blacklist);

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
    public Task<DenseTensor<float>> ForwardAsync(
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

        // Bounded run: if the native call hangs (e.g. a cold DirectML kernel init) or the provider
        // crashes, the session moves to the next provider and the run is retried once.
        return _session.RunWithRecoveryAsync((session, runOptions) =>
        {
            using var outputs = session.Run(inputs, [_output], runOptions);
            var outputTensor = outputs[0].AsTensor<float>();

            // Copy out of the native buffer since the outputs are disposed with this delegate
            return new DenseTensor<float>(outputTensor.ToArray(), outputTensor.Dimensions);
        }, cancellationToken: cancellationToken);
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
