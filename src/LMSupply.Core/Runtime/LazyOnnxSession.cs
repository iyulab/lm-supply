using LMSupply.Download;
using LMSupply.Inference;
using Microsoft.ML.OnnxRuntime;

namespace LMSupply.Runtime;

/// <summary>
/// A lazy-loading ONNX inference session that automatically downloads runtime binaries on first use.
/// This demonstrates the complete lazy binary distribution workflow.
/// </summary>
public sealed class LazyOnnxSession : IDisposable, IAsyncDisposable
{
    private readonly string _modelPath;
    private readonly ExecutionProvider _requestedProvider;
    private readonly Action<SessionOptions>? _configureOptions;
    private readonly SemaphoreSlim _initLock = new(1, 1);

    private InferenceSession? _session;
    private ExecutionProvider _actualProvider;
    private bool _initialized;
    private bool _disposed;

    /// <summary>
    /// Creates a lazy ONNX session.
    /// </summary>
    /// <param name="modelPath">Path to the ONNX model file.</param>
    /// <param name="provider">Requested execution provider. Auto will select best available.</param>
    /// <param name="configureOptions">Optional callback to configure session options.</param>
    public LazyOnnxSession(
        string modelPath,
        ExecutionProvider provider = ExecutionProvider.Auto,
        Action<SessionOptions>? configureOptions = null)
    {
        _modelPath = modelPath;
        _requestedProvider = provider;
        _configureOptions = configureOptions;
    }

    /// <summary>
    /// Gets the actual execution provider being used.
    /// Only valid after initialization.
    /// </summary>
    public ExecutionProvider ActualProvider => _actualProvider;

    /// <summary>
    /// Gets whether the session has been initialized.
    /// </summary>
    public bool IsInitialized => _initialized;

    /// <summary>
    /// Gets the underlying inference session.
    /// Initializes lazily on first access.
    /// </summary>
    public InferenceSession Session => GetSessionAsync().GetAwaiter().GetResult();

    /// <summary>
    /// Initializes the session, ensuring runtime binaries are available.
    /// </summary>
    /// <param name="progress">Optional progress reporter for binary downloads.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    public async Task InitializeAsync(
        IProgress<DownloadProgress>? progress = null,
        CancellationToken cancellationToken = default)
    {
        if (_initialized)
            return;

        await _initLock.WaitAsync(cancellationToken);
        try
        {
            if (_initialized)
                return;

            // Initialize runtime manager
            await RuntimeManager.Instance.InitializeAsync(cancellationToken);

            // Determine the provider to use
            _actualProvider = _requestedProvider == ExecutionProvider.Auto
                ? RuntimeManager.Instance.RecommendedProvider
                : _requestedProvider;

            // Get the provider string for binary download
            var providerString = _actualProvider switch
            {
                ExecutionProvider.Cuda => RuntimeManager.Instance.GetDefaultProvider(), // cuda11 or cuda12
                ExecutionProvider.DirectML => "directml",
                ExecutionProvider.CoreML => "cpu", // CoreML uses CPU binaries with framework
                _ => "cpu"
            };

            // Ensure runtime binaries are available (downloads from GitHub Releases if needed)
            await RuntimeManager.Instance.EnsureRuntimeAsync(
                "onnxruntime",
                provider: providerString,
                progress: progress,
                cancellationToken: cancellationToken);

            // Create session options
            var options = new SessionOptions();
            _configureOptions?.Invoke(options);

            // Configure execution provider
            ConfigureProvider(options, _actualProvider);

            // Create the session
            _session = new InferenceSession(_modelPath, options);
            _initialized = true;
        }
        finally
        {
            _initLock.Release();
        }
    }

    /// <summary>
    /// Gets the session asynchronously, initializing if needed.
    /// </summary>
    public async Task<InferenceSession> GetSessionAsync(CancellationToken cancellationToken = default)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        if (!_initialized)
        {
            await InitializeAsync(cancellationToken: cancellationToken);
        }

        return _session!;
    }

    /// <summary>
    /// Runs inference on the model.
    /// </summary>
    public async Task<IDisposableReadOnlyCollection<DisposableNamedOnnxValue>> RunAsync(
        IReadOnlyCollection<NamedOnnxValue> inputs,
        CancellationToken cancellationToken = default)
    {
        var session = await GetSessionAsync(cancellationToken);
        return session.Run(inputs);
    }

    /// <summary>
    /// Runs inference on the model with specified output names.
    /// </summary>
    public async Task<IDisposableReadOnlyCollection<DisposableNamedOnnxValue>> RunAsync(
        IReadOnlyCollection<NamedOnnxValue> inputs,
        IReadOnlyCollection<string> outputNames,
        CancellationToken cancellationToken = default)
    {
        var session = await GetSessionAsync(cancellationToken);
        return session.Run(inputs, outputNames);
    }

    /// <summary>
    /// Gets the input metadata for the model.
    /// </summary>
    public async Task<IReadOnlyDictionary<string, NodeMetadata>> GetInputMetadataAsync(
        CancellationToken cancellationToken = default)
    {
        var session = await GetSessionAsync(cancellationToken);
        return session.InputMetadata;
    }

    /// <summary>
    /// Gets the output metadata for the model.
    /// </summary>
    public async Task<IReadOnlyDictionary<string, NodeMetadata>> GetOutputMetadataAsync(
        CancellationToken cancellationToken = default)
    {
        var session = await GetSessionAsync(cancellationToken);
        return session.OutputMetadata;
    }

    private void ConfigureProvider(SessionOptions options, ExecutionProvider provider)
    {
        switch (provider)
        {
            case ExecutionProvider.Auto:
                // Try best available
                if (!TryAddCuda(options) && !TryAddDirectML(options) && !TryAddCoreML(options))
                {
                    _actualProvider = ExecutionProvider.Cpu;
                }
                break;

            case ExecutionProvider.Cuda:
                if (!TryAddCuda(options))
                {
                    throw new InvalidOperationException("CUDA execution provider is not available");
                }
                break;

            case ExecutionProvider.DirectML:
                if (!TryAddDirectML(options))
                {
                    throw new InvalidOperationException("DirectML execution provider is not available");
                }
                // DirectML specific settings
                options.EnableMemoryPattern = false;
                options.ExecutionMode = ExecutionMode.ORT_SEQUENTIAL;
                break;

            case ExecutionProvider.CoreML:
                if (!TryAddCoreML(options))
                {
                    throw new InvalidOperationException("CoreML execution provider is not available");
                }
                break;

            case ExecutionProvider.Cpu:
                // CPU is always available
                break;
        }
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

    public void Dispose()
    {
        if (_disposed)
            return;

        _disposed = true;
        _session?.Dispose();
        _initLock.Dispose();
    }

    public async ValueTask DisposeAsync()
    {
        if (_disposed)
            return;

        _disposed = true;
        _session?.Dispose();
        _initLock.Dispose();
        await Task.CompletedTask;
    }
}
