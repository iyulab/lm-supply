using System.Runtime.CompilerServices;
using LocalAI.Generator.Abstractions;
using LocalAI.Generator.Models;

namespace LocalAI.Generator;

/// <summary>
/// Configuration options for memory-aware generation.
/// </summary>
public sealed class MemoryAwareOptions
{
    /// <summary>
    /// Maximum memory usage in bytes. Generation will be blocked if exceeded.
    /// </summary>
    public long MaxMemoryBytes { get; set; } = 4L * 1024 * 1024 * 1024; // 4GB default

    /// <summary>
    /// Warning threshold as a percentage of max memory (0.0 - 1.0).
    /// When exceeded, a GC will be triggered.
    /// </summary>
    public double WarningThreshold { get; set; } = 0.80;

    /// <summary>
    /// Critical threshold as a percentage of max memory (0.0 - 1.0).
    /// When exceeded after GC, generation will throw.
    /// </summary>
    public double CriticalThreshold { get; set; } = 0.95;

    /// <summary>
    /// Interval between memory checks during generation (in tokens).
    /// </summary>
    public int CheckIntervalTokens { get; set; } = 100;

    /// <summary>
    /// Whether to trigger GC when warning threshold is exceeded.
    /// </summary>
    public bool AutoGcOnWarning { get; set; } = true;

    /// <summary>
    /// Creates options with a specific memory limit in megabytes.
    /// </summary>
    public static MemoryAwareOptions WithLimitMB(long limitMB) => new()
    {
        MaxMemoryBytes = limitMB * 1024 * 1024
    };

    /// <summary>
    /// Creates options with a specific memory limit in gigabytes.
    /// </summary>
    public static MemoryAwareOptions WithLimitGB(double limitGB) => new()
    {
        MaxMemoryBytes = (long)(limitGB * 1024 * 1024 * 1024)
    };
}

/// <summary>
/// Decorator that wraps a generator model with memory monitoring and management.
/// </summary>
public sealed class MemoryAwareGenerator : IGeneratorModel
{
    private readonly IGeneratorModel _inner;
    private readonly MemoryAwareOptions _options;
    private bool _disposed;

    /// <summary>
    /// Creates a new memory-aware generator wrapper.
    /// </summary>
    /// <param name="inner">The underlying generator model.</param>
    /// <param name="options">Memory management options.</param>
    public MemoryAwareGenerator(IGeneratorModel inner, MemoryAwareOptions? options = null)
    {
        _inner = inner ?? throw new ArgumentNullException(nameof(inner));
        _options = options ?? new MemoryAwareOptions();
    }

    /// <inheritdoc />
    public string ModelId => _inner.ModelId;

    /// <inheritdoc />
    public int MaxContextLength => _inner.MaxContextLength;

    /// <inheritdoc />
    public IChatFormatter ChatFormatter => _inner.ChatFormatter;

    /// <inheritdoc />
    public async IAsyncEnumerable<string> GenerateAsync(
        string prompt,
        GeneratorOptions? options = null,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        ThrowIfDisposed();
        CheckMemoryBudget();

        var tokenCount = 0;
        await foreach (var token in _inner.GenerateAsync(prompt, options, cancellationToken))
        {
            yield return token;

            tokenCount++;
            if (tokenCount % _options.CheckIntervalTokens == 0)
            {
                CheckMemoryDuringGeneration();
            }
        }
    }

    /// <inheritdoc />
    public async IAsyncEnumerable<string> GenerateChatAsync(
        IEnumerable<ChatMessage> messages,
        GeneratorOptions? options = null,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        ThrowIfDisposed();
        CheckMemoryBudget();

        var tokenCount = 0;
        await foreach (var token in _inner.GenerateChatAsync(messages, options, cancellationToken))
        {
            yield return token;

            tokenCount++;
            if (tokenCount % _options.CheckIntervalTokens == 0)
            {
                CheckMemoryDuringGeneration();
            }
        }
    }

    /// <inheritdoc />
    public async Task<string> GenerateCompleteAsync(
        string prompt,
        GeneratorOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        ThrowIfDisposed();
        CheckMemoryBudget();

        var result = await _inner.GenerateCompleteAsync(prompt, options, cancellationToken);
        CheckMemoryDuringGeneration();
        return result;
    }

    /// <inheritdoc />
    public async Task<string> GenerateChatCompleteAsync(
        IEnumerable<ChatMessage> messages,
        GeneratorOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        ThrowIfDisposed();
        CheckMemoryBudget();

        var result = await _inner.GenerateChatCompleteAsync(messages, options, cancellationToken);
        CheckMemoryDuringGeneration();
        return result;
    }

    /// <inheritdoc />
    public Task WarmupAsync(CancellationToken cancellationToken = default)
    {
        ThrowIfDisposed();
        CheckMemoryBudget();
        return _inner.WarmupAsync(cancellationToken);
    }

    /// <inheritdoc />
    public GeneratorModelInfo GetModelInfo() => _inner.GetModelInfo();

    /// <summary>
    /// Gets the current memory usage in bytes.
    /// </summary>
    public long GetCurrentMemoryUsage() => GC.GetTotalMemory(forceFullCollection: false);

    /// <summary>
    /// Gets the memory usage as a percentage of the configured limit.
    /// </summary>
    public double GetMemoryUsagePercent()
    {
        var current = GetCurrentMemoryUsage();
        return (double)current / _options.MaxMemoryBytes;
    }

    /// <summary>
    /// Forces a garbage collection if memory usage exceeds the warning threshold.
    /// </summary>
    public void TryReduceMemory()
    {
        if (GetMemoryUsagePercent() > _options.WarningThreshold)
        {
            GC.Collect(2, GCCollectionMode.Optimized, blocking: false);
        }
    }

    private void CheckMemoryBudget()
    {
        var usagePercent = GetMemoryUsagePercent();

        if (usagePercent > _options.WarningThreshold && _options.AutoGcOnWarning)
        {
            GC.Collect(2, GCCollectionMode.Optimized, blocking: true);
            GC.WaitForPendingFinalizers();

            // Re-check after GC
            usagePercent = GetMemoryUsagePercent();
        }

        if (usagePercent > _options.CriticalThreshold)
        {
            var currentMB = GetCurrentMemoryUsage() / (1024 * 1024);
            var maxMB = _options.MaxMemoryBytes / (1024 * 1024);
            throw new OutOfMemoryException(
                $"Insufficient memory for generation. Current: {currentMB}MB, Limit: {maxMB}MB ({usagePercent:P1})");
        }
    }

    private void CheckMemoryDuringGeneration()
    {
        var usagePercent = GetMemoryUsagePercent();

        if (usagePercent > _options.WarningThreshold && _options.AutoGcOnWarning)
        {
            // Non-blocking GC during generation
            GC.Collect(2, GCCollectionMode.Optimized, blocking: false);
        }
    }

    private void ThrowIfDisposed()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
    }

    /// <inheritdoc />
    public async ValueTask DisposeAsync()
    {
        if (_disposed) return;

        _disposed = true;
        await _inner.DisposeAsync();
    }
}
