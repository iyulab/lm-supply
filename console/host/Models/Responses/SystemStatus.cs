namespace LMSupply.Console.Host.Models.Responses;

/// <summary>
/// System status
/// </summary>
public sealed record SystemStatus
{
    /// <summary>
    /// Whether ONNX Runtime is ready
    /// </summary>
    public bool EngineReady { get; init; }

    /// <summary>
    /// Whether a GPU is available
    /// </summary>
    public bool GpuAvailable { get; init; }

    /// <summary>
    /// Execution provider (CUDA, DirectML, CPU, etc.)
    /// </summary>
    public string? GpuProvider { get; init; }

    /// <summary>
    /// GPU name
    /// </summary>
    public string? GpuName { get; init; }

    /// <summary>
    /// CPU usage (0-100)
    /// </summary>
    public float CpuUsage { get; init; }

    /// <summary>
    /// RAM used (MB)
    /// </summary>
    public double RamUsageMB { get; init; }

    /// <summary>
    /// Total RAM (MB)
    /// </summary>
    public double RamTotalMB { get; init; }

    /// <summary>
    /// RAM usage (0-100)
    /// </summary>
    public double RamUsagePercent { get; init; }

    /// <summary>
    /// VRAM used (MB)
    /// </summary>
    public double? VramUsageMB { get; init; }

    /// <summary>
    /// Total VRAM (MB)
    /// </summary>
    public double? VramTotalMB { get; init; }

    /// <summary>
    /// VRAM usage (0-100)
    /// </summary>
    public double? VramUsagePercent { get; init; }

    /// <summary>
    /// Current process memory (MB)
    /// </summary>
    public double ProcessMemoryMB { get; init; }

    /// <summary>
    /// Timestamp
    /// </summary>
    public DateTime Timestamp { get; init; }
}

/// <summary>
/// Memory metrics
/// </summary>
public sealed record MemoryMetrics
{
    public double TotalMB { get; init; }
    public double UsedMB { get; init; }
    public double UsagePercent { get; init; }
}

/// <summary>
/// GPU info
/// </summary>
public sealed record GpuInfo
{
    public bool IsAvailable { get; init; }
    public string? Name { get; init; }
    public string? Provider { get; init; }
    public double? TotalVramMB { get; init; }
    public double? UsedVramMB { get; init; }
    public double? VramUsagePercent => TotalVramMB > 0 ? (UsedVramMB / TotalVramMB) * 100 : null;
}

/// <summary>
/// Real-time metrics (for streaming)
/// </summary>
public sealed record SystemMetrics
{
    public float CpuUsage { get; init; }
    public double RamUsageMB { get; init; }
    public DateTime Timestamp { get; init; }
}

/// <summary>
/// Loaded model info
/// </summary>
public sealed record LoadedModelInfo
{
    public required string ModelId { get; init; }
    public required string ModelType { get; init; }
    public DateTime LoadedAt { get; init; }
    public DateTime LastUsedAt { get; init; }
}
