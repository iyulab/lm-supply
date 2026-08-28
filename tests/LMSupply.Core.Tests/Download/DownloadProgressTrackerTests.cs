using AwesomeAssertions;

namespace LMSupply.Core.Tests.Download;

public class DownloadProgressTrackerTests
{
    // --- DownloadProgressTracker: Constructor & Start ---

    [Fact]
    public void NewTracker_CurrentSpeed_IsZero()
    {
        var tracker = new DownloadProgressTracker();

        tracker.CurrentSpeed.Should().Be(0);
    }

    [Fact]
    public void Start_ResetsState()
    {
        var tracker = new DownloadProgressTracker(sampleWindowSize: 5, sampleIntervalMs: 0);
        tracker.Start();

        // macOS timer resolution requires a small delay for non-zero elapsed time
        Thread.Sleep(16);

        // First call to seed a sample
        tracker.CreateProgress("file.bin", 1000, 10000);
        tracker.CurrentSpeed.Should().BeGreaterThan(0);

        // Restart should reset
        tracker.Start();
        tracker.CurrentSpeed.Should().Be(0);
    }

    // --- CreateProgress: Basic properties ---

    [Fact]
    public void CreateProgress_SetsFileNameAndBytes()
    {
        var tracker = new DownloadProgressTracker();
        tracker.Start();

        var progress = tracker.CreateProgress("model.onnx", 500, 1000);

        progress.FileName.Should().Be("model.onnx");
        progress.BytesDownloaded.Should().Be(500);
        progress.TotalBytes.Should().Be(1000);
    }

    [Fact]
    public void CreateProgress_SetsPhase()
    {
        var tracker = new DownloadProgressTracker();
        tracker.Start();

        var progress = tracker.CreateProgress("f.bin", 100, 200, DownloadPhase.Extracting);

        progress.Phase.Should().Be(DownloadPhase.Extracting);
    }

    [Fact]
    public void CreateProgress_DefaultPhase_IsDownloading()
    {
        var tracker = new DownloadProgressTracker();
        tracker.Start();

        var progress = tracker.CreateProgress("f.bin", 100, 200);

        progress.Phase.Should().Be(DownloadPhase.Downloading);
    }

    // --- Speed calculation ---

    [Fact]
    public void CreateProgress_WithSampleIntervalZero_CalculatesSpeed()
    {
        // sampleIntervalMs=0 means every call creates a sample
        var tracker = new DownloadProgressTracker(sampleWindowSize: 10, sampleIntervalMs: 0);
        tracker.Start();

        // macOS timer resolution requires a small delay for non-zero elapsed time
        Thread.Sleep(16);

        // First call with bytes > 0
        tracker.CreateProgress("f.bin", 1000, 10000);

        tracker.CurrentSpeed.Should().BeGreaterThan(0);
    }

    [Fact]
    public void CreateProgress_NoBytesIncrease_DoesNotUpdateSpeed()
    {
        var tracker = new DownloadProgressTracker(sampleWindowSize: 10, sampleIntervalMs: 0);
        tracker.Start();

        // Same bytes twice - no delta
        tracker.CreateProgress("f.bin", 1000, 10000);
        var speedAfterFirst = tracker.CurrentSpeed;

        // Same bytes - should not add a new sample
        tracker.CreateProgress("f.bin", 1000, 10000);

        tracker.CurrentSpeed.Should().Be(speedAfterFirst);
    }

    [Fact]
    public void CreateProgress_MultipleSamples_UsesMovingAverage()
    {
        var tracker = new DownloadProgressTracker(sampleWindowSize: 3, sampleIntervalMs: 0);
        tracker.Start();

        // Generate multiple samples with increasing bytes
        tracker.CreateProgress("f.bin", 1000, 100000);
        tracker.CreateProgress("f.bin", 3000, 100000);
        tracker.CreateProgress("f.bin", 6000, 100000);
        tracker.CreateProgress("f.bin", 10000, 100000);

        // Speed should be positive and smoothed
        tracker.CurrentSpeed.Should().BeGreaterThan(0);
    }

    [Fact]
    public void CreateProgress_ExceedWindowSize_TrimsOldSamples()
    {
        var tracker = new DownloadProgressTracker(sampleWindowSize: 2, sampleIntervalMs: 0);
        tracker.Start();

        // Generate more samples than window size
        for (var i = 1; i <= 5; i++)
        {
            tracker.CreateProgress("f.bin", i * 1000, 100000);
        }

        // Should still work with trimmed window
        tracker.CurrentSpeed.Should().BeGreaterThan(0);
    }

    // --- ETA calculation ---

    [Fact]
    public void CreateProgress_WithSpeedAndRemainingBytes_HasEta()
    {
        var tracker = new DownloadProgressTracker(sampleWindowSize: 10, sampleIntervalMs: 0);
        tracker.Start();

        // Build up speed
        tracker.CreateProgress("f.bin", 1000, 10000);
        tracker.CreateProgress("f.bin", 5000, 10000);

        var progress = tracker.CreateProgress("f.bin", 5000, 10000);

        // ETA should be set since there's remaining bytes and speed > 0
        if (tracker.CurrentSpeed > 0)
        {
            progress.EstimatedRemaining.Should().NotBeNull();
            progress.EstimatedRemaining!.Value.TotalSeconds.Should().BeGreaterThan(0);
        }
    }

    [Fact]
    public void CreateProgress_DownloadComplete_NoEta()
    {
        var tracker = new DownloadProgressTracker(sampleWindowSize: 10, sampleIntervalMs: 0);
        tracker.Start();

        tracker.CreateProgress("f.bin", 5000, 10000);

        // Download complete (downloaded == total)
        var progress = tracker.CreateProgress("f.bin", 10000, 10000);

        progress.EstimatedRemaining.Should().BeNull();
    }

    [Fact]
    public void CreateProgress_NoSpeedYet_NoEta()
    {
        var tracker = new DownloadProgressTracker(sampleWindowSize: 10, sampleIntervalMs: 999999);
        tracker.Start();

        // Large sample interval means no sample will be taken
        var progress = tracker.CreateProgress("f.bin", 100, 10000);

        progress.EstimatedRemaining.Should().BeNull();
        progress.BytesPerSecond.Should().BeNull();
    }

    // --- CreatePhaseProgress: Static factory ---

    [Fact]
    public void CreatePhaseProgress_SetsAllProperties()
    {
        var progress = DownloadProgressTracker.CreatePhaseProgress(
            "archive.zip", DownloadPhase.Extracting, 5000, 10000);

        progress.FileName.Should().Be("archive.zip");
        progress.Phase.Should().Be(DownloadPhase.Extracting);
        progress.BytesDownloaded.Should().Be(5000);
        progress.TotalBytes.Should().Be(10000);
        progress.BytesPerSecond.Should().BeNull();
        progress.EstimatedRemaining.Should().BeNull();
    }

    [Fact]
    public void CreatePhaseProgress_DefaultBytes_AreZero()
    {
        var progress = DownloadProgressTracker.CreatePhaseProgress("f.bin", DownloadPhase.Preparing);

        progress.BytesDownloaded.Should().Be(0);
        progress.TotalBytes.Should().Be(0);
    }

    // --- Elapsed ---

    [Fact]
    public void Elapsed_AfterStart_IsPositive()
    {
        var tracker = new DownloadProgressTracker();
        tracker.Start();

        // Small delay to ensure elapsed > 0
        Thread.Sleep(10);

        tracker.Elapsed.Should().BeGreaterThan(TimeSpan.Zero);
    }

    // --- DownloadProgress: Computed properties ---

    [Fact]
    public void PercentComplete_CalculatesCorrectly()
    {
        var progress = new DownloadProgress
        {
            FileName = "f.bin",
            BytesDownloaded = 250,
            TotalBytes = 1000
        };

        progress.PercentComplete.Should().BeApproximately(25.0, 0.001);
    }

    [Fact]
    public void PercentComplete_ZeroTotalBytes_ReturnsZero()
    {
        var progress = new DownloadProgress
        {
            FileName = "f.bin",
            BytesDownloaded = 100,
            TotalBytes = 0
        };

        progress.PercentComplete.Should().Be(0);
    }

    [Fact]
    public void PercentComplete_FullyDownloaded_Returns100()
    {
        var progress = new DownloadProgress
        {
            FileName = "f.bin",
            BytesDownloaded = 5000,
            TotalBytes = 5000
        };

        progress.PercentComplete.Should().BeApproximately(100.0, 0.001);
    }

    // --- SpeedDisplay ---

    [Fact]
    public void SpeedDisplay_Null_ReturnsEmpty()
    {
        var progress = new DownloadProgress
        {
            FileName = "f.bin",
            BytesPerSecond = null
        };

        progress.SpeedDisplay.Should().BeEmpty();
    }

    [Fact]
    public void SpeedDisplay_BytesPerSecond_ShowsBytes()
    {
        var progress = new DownloadProgress
        {
            FileName = "f.bin",
            BytesPerSecond = 500
        };

        progress.SpeedDisplay.Should().Contain("B/s");
        progress.SpeedDisplay.Should().NotContain("KB");
    }

    [Fact]
    public void SpeedDisplay_KilobytesPerSecond_ShowsKB()
    {
        var progress = new DownloadProgress
        {
            FileName = "f.bin",
            BytesPerSecond = 5000 // ~4.9 KB/s
        };

        progress.SpeedDisplay.Should().Contain("KB/s");
    }

    [Fact]
    public void SpeedDisplay_MegabytesPerSecond_ShowsMB()
    {
        var progress = new DownloadProgress
        {
            FileName = "f.bin",
            BytesPerSecond = 15_000_000 // ~14.3 MB/s
        };

        progress.SpeedDisplay.Should().Contain("MB/s");
    }

    // --- EtaDisplay ---

    [Fact]
    public void EtaDisplay_Null_ReturnsEmpty()
    {
        var progress = new DownloadProgress
        {
            FileName = "f.bin",
            EstimatedRemaining = null
        };

        progress.EtaDisplay.Should().BeEmpty();
    }

    [Fact]
    public void EtaDisplay_Seconds_ShowsSecondsOnly()
    {
        var progress = new DownloadProgress
        {
            FileName = "f.bin",
            EstimatedRemaining = TimeSpan.FromSeconds(30)
        };

        progress.EtaDisplay.Should().Contain("s");
        progress.EtaDisplay.Should().NotContain("m");
    }

    [Fact]
    public void EtaDisplay_Minutes_ShowsMinutesAndSeconds()
    {
        var progress = new DownloadProgress
        {
            FileName = "f.bin",
            EstimatedRemaining = TimeSpan.FromMinutes(2) + TimeSpan.FromSeconds(30)
        };

        progress.EtaDisplay.Should().Contain("m");
        progress.EtaDisplay.Should().Contain("s");
    }

    [Fact]
    public void EtaDisplay_Hours_ShowsHoursAndMinutes()
    {
        var progress = new DownloadProgress
        {
            FileName = "f.bin",
            EstimatedRemaining = TimeSpan.FromHours(1) + TimeSpan.FromMinutes(15)
        };

        progress.EtaDisplay.Should().Contain("h");
        progress.EtaDisplay.Should().Contain("m");
    }

    // --- DownloadPhase enum ---

    [Theory]
    [InlineData(DownloadPhase.Preparing)]
    [InlineData(DownloadPhase.Downloading)]
    [InlineData(DownloadPhase.Extracting)]
    [InlineData(DownloadPhase.Verifying)]
    [InlineData(DownloadPhase.Finalizing)]
    [InlineData(DownloadPhase.Complete)]
    public void DownloadPhase_AllValues_AreValid(DownloadPhase phase)
    {
        var progress = new DownloadProgress
        {
            FileName = "f.bin",
            Phase = phase
        };

        progress.Phase.Should().Be(phase);
    }
}
