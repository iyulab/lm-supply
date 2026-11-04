using System.Diagnostics;

namespace LMSupply;

/// <summary>
/// Tracks download progress and calculates speed/ETA.
/// </summary>
public sealed class DownloadProgressTracker
{
    private readonly Stopwatch _stopwatch = new();
    private readonly Queue<(long bytes, TimeSpan time)> _samples = new();
    private readonly int _sampleWindowSize;
    private readonly TimeSpan _sampleInterval;

    private long _lastReportedBytes;
    private TimeSpan _lastSampleTime;
    private double _smoothedSpeed;

    /// <summary>
    /// Creates a new progress tracker.
    /// </summary>
    /// <param name="sampleWindowSize">Number of samples for moving average (default: 10).</param>
    /// <param name="sampleIntervalMs">Minimum interval between samples in ms (default: 200).</param>
    public DownloadProgressTracker(int sampleWindowSize = 10, int sampleIntervalMs = 200)
    {
        _sampleWindowSize = sampleWindowSize;
        _sampleInterval = TimeSpan.FromMilliseconds(sampleIntervalMs);
    }

    /// <summary>
    /// Starts tracking progress.
    /// </summary>
    public void Start() => Start(0);

    /// <summary>
    /// Starts tracking progress of a transfer that resumes with bytes already on disk: those bytes are the
    /// baseline, not part of the first speed sample, so a resumed download does not report a speed spike.
    /// </summary>
    /// <param name="alreadyDownloadedBytes">Bytes held before this transfer began.</param>
    public void Start(long alreadyDownloadedBytes)
    {
        _stopwatch.Restart();
        _samples.Clear();
        _lastReportedBytes = Math.Max(0, alreadyDownloadedBytes);
        _lastSampleTime = TimeSpan.Zero;
        _smoothedSpeed = 0;
    }

    /// <summary>
    /// Creates a progress report with calculated speed and ETA.
    /// </summary>
    /// <param name="fileName">Name of the file being downloaded.</param>
    /// <param name="bytesDownloaded">Total bytes downloaded so far.</param>
    /// <param name="totalBytes">Total file size in bytes.</param>
    /// <param name="phase">Current download phase.</param>
    /// <returns>Progress report with speed and ETA.</returns>
    public DownloadProgress CreateProgress(
        string fileName,
        long bytesDownloaded,
        long totalBytes,
        DownloadPhase phase = DownloadPhase.Downloading)
    {
        var elapsed = _stopwatch.Elapsed;
        var timeSinceLast = elapsed - _lastSampleTime;

        // Update samples if enough time has passed
        if (timeSinceLast >= _sampleInterval && bytesDownloaded > _lastReportedBytes)
        {
            var bytesDelta = bytesDownloaded - _lastReportedBytes;
            _samples.Enqueue((bytesDelta, timeSinceLast));

            // Keep window size limited
            while (_samples.Count > _sampleWindowSize)
            {
                _samples.Dequeue();
            }

            _lastReportedBytes = bytesDownloaded;
            _lastSampleTime = elapsed;

            // Calculate moving average speed
            if (_samples.Count > 0)
            {
                var totalSampleBytes = _samples.Sum(s => s.bytes);
                var totalSampleTime = _samples.Sum(s => s.time.TotalSeconds);

                if (totalSampleTime > 0)
                {
                    var instantSpeed = totalSampleBytes / totalSampleTime;

                    // Exponential smoothing
                    const double alpha = 0.3;
                    _smoothedSpeed = _smoothedSpeed == 0
                        ? instantSpeed
                        : alpha * instantSpeed + (1 - alpha) * _smoothedSpeed;
                }
            }
        }

        // Calculate ETA
        TimeSpan? eta = null;
        if (_smoothedSpeed > 0 && totalBytes > bytesDownloaded)
        {
            var remainingBytes = totalBytes - bytesDownloaded;
            var secondsRemaining = remainingBytes / _smoothedSpeed;
            eta = TimeSpan.FromSeconds(secondsRemaining);
        }

        return new DownloadProgress
        {
            FileName = fileName,
            BytesDownloaded = bytesDownloaded,
            TotalBytes = totalBytes,
            BytesPerSecond = _smoothedSpeed > 0 ? _smoothedSpeed : null,
            EstimatedRemaining = eta,
            Phase = phase
        };
    }

    /// <summary>
    /// Creates a progress report for a specific phase (no speed calculation).
    /// </summary>
    public static DownloadProgress CreatePhaseProgress(
        string fileName,
        DownloadPhase phase,
        long bytesDownloaded = 0,
        long totalBytes = 0)
    {
        return new DownloadProgress
        {
            FileName = fileName,
            BytesDownloaded = bytesDownloaded,
            TotalBytes = totalBytes,
            Phase = phase
        };
    }

    /// <summary>
    /// Gets the current smoothed speed in bytes per second.
    /// </summary>
    public double CurrentSpeed => _smoothedSpeed;

    /// <summary>
    /// Gets the elapsed time since tracking started.
    /// </summary>
    public TimeSpan Elapsed => _stopwatch.Elapsed;
}
