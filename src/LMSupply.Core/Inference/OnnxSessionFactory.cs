using LMSupply.Download;
using LMSupply.Runtime;
using Microsoft.ML.OnnxRuntime;

namespace LMSupply.Inference;

/// <summary>
/// Factory for creating ONNX Runtime inference sessions with proper execution provider configuration.
/// Supports lazy loading of native runtime binaries via RuntimeManager.
/// </summary>
public static class OnnxSessionFactory
{
    /// <summary>
    /// Creates an ONNX Runtime inference session asynchronously, ensuring runtime binaries are available.
    /// This is the recommended method as it downloads required binaries on first use.
    /// </summary>
    /// <param name="modelPath">Path to the ONNX model file.</param>
    /// <param name="provider">The execution provider to use.</param>
    /// <param name="configureOptions">Optional callback to configure additional session options.</param>
    /// <param name="progress">Optional progress reporter for binary downloads.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>A configured inference session.</returns>
    public static async Task<InferenceSession> CreateAsync(
        string modelPath,
        ExecutionProvider provider = ExecutionProvider.Auto,
        Action<SessionOptions>? configureOptions = null,
        IProgress<DownloadProgress>? progress = null,
        CancellationToken cancellationToken = default)
    {
        // Ensure runtime binaries are available
        await RuntimeManager.Instance.InitializeAsync(cancellationToken);

        // Determine the actual provider to use
        var actualProvider = provider == ExecutionProvider.Auto
            ? RuntimeManager.Instance.RecommendedProvider
            : provider;

        // Get the provider string for binary download
        var providerString = actualProvider switch
        {
            ExecutionProvider.Cuda => RuntimeManager.Instance.GetDefaultProvider(), // cuda11 or cuda12
            ExecutionProvider.DirectML => "directml",
            ExecutionProvider.CoreML => "cpu", // CoreML uses CPU binaries with framework
            _ => "cpu"
        };

        // Download runtime binaries if needed
        await RuntimeManager.Instance.EnsureRuntimeAsync(
            "onnxruntime",
            provider: providerString,
            progress: progress,
            cancellationToken: cancellationToken);

        // Create session using sync method (binaries now available)
        return Create(modelPath, actualProvider, configureOptions);
    }

    /// <summary>
    /// Creates an ONNX Runtime inference session with the specified execution provider.
    /// Note: This assumes runtime binaries are already available. For lazy loading, use CreateAsync.
    /// </summary>
    /// <param name="modelPath">Path to the ONNX model file.</param>
    /// <param name="provider">The execution provider to use.</param>
    /// <param name="configureOptions">Optional callback to configure additional session options.</param>
    /// <returns>A configured inference session.</returns>
    public static InferenceSession Create(
        string modelPath,
        ExecutionProvider provider = ExecutionProvider.Auto,
        Action<SessionOptions>? configureOptions = null)
    {
        var options = new SessionOptions();

        // Apply user configuration first
        configureOptions?.Invoke(options);

        // Configure execution provider
        ConfigureExecutionProvider(options, provider);

        return new InferenceSession(modelPath, options);
    }

    /// <summary>
    /// Configures the execution provider for the session options.
    /// </summary>
    public static void ConfigureExecutionProvider(SessionOptions options, ExecutionProvider provider)
    {
        switch (provider)
        {
            case ExecutionProvider.Auto:
                TryAddBestAvailableProvider(options);
                break;

            case ExecutionProvider.Cuda:
                TryAddCuda(options);
                break;

            case ExecutionProvider.DirectML:
                TryAddDirectML(options);
                break;

            case ExecutionProvider.CoreML:
                TryAddCoreML(options);
                break;

            case ExecutionProvider.Cpu:
                // CPU is always available as fallback
                break;

            default:
                throw new ArgumentOutOfRangeException(nameof(provider), provider, "Unknown execution provider");
        }
    }

    /// <summary>
    /// Tries to add the best available GPU provider, falls back to CPU.
    /// </summary>
    private static void TryAddBestAvailableProvider(SessionOptions options)
    {
        // Try providers in order of preference
        if (TryAddCuda(options)) return;
        if (TryAddDirectML(options)) return;
        if (TryAddCoreML(options)) return;
        // CPU fallback is automatic
    }

    private static bool TryAddCuda(SessionOptions options)
    {
        try
        {
            options.AppendExecutionProvider_CUDA();
            return true;
        }
        catch
        {
            return false;
        }
    }

    private static bool TryAddDirectML(SessionOptions options)
    {
        try
        {
            options.AppendExecutionProvider_DML();
            return true;
        }
        catch
        {
            return false;
        }
    }

    private static bool TryAddCoreML(SessionOptions options)
    {
        try
        {
            options.AppendExecutionProvider_CoreML();
            return true;
        }
        catch
        {
            return false;
        }
    }

    /// <summary>
    /// Gets the list of available execution providers on the current system.
    /// </summary>
    public static IEnumerable<ExecutionProvider> GetAvailableProviders()
    {
        // CPU is always available
        yield return ExecutionProvider.Cpu;

        // Check GPU providers
        var testOptions = new SessionOptions();

        if (TryAddCuda(testOptions))
            yield return ExecutionProvider.Cuda;

        testOptions = new SessionOptions();
        if (TryAddDirectML(testOptions))
            yield return ExecutionProvider.DirectML;

        testOptions = new SessionOptions();
        if (TryAddCoreML(testOptions))
            yield return ExecutionProvider.CoreML;
    }
}
