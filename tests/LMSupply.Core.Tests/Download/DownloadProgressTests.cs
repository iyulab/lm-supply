using AwesomeAssertions;

namespace LMSupply.Core.Tests.Download;

public class DownloadProgressTests
{
    // --- PercentComplete (per-file) ---

    [Fact]
    public void PercentComplete_HalfDownloaded_Returns50()
    {
        var progress = new DownloadProgress
        {
            FileName = "model.onnx",
            BytesDownloaded = 500,
            TotalBytes = 1000
        };

        progress.PercentComplete.Should().BeApproximately(50.0, 0.01);
    }

    [Fact]
    public void PercentComplete_ZeroTotalBytes_ReturnsZero()
    {
        var progress = new DownloadProgress
        {
            FileName = "model.onnx",
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
            FileName = "model.onnx",
            BytesDownloaded = 2048,
            TotalBytes = 2048
        };

        progress.PercentComplete.Should().BeApproximately(100.0, 0.01);
    }

    // --- OverallPercentComplete (multi-file) ---

    [Fact]
    public void OverallPercentComplete_SingleFile_FallsBackToPercentComplete()
    {
        var progress = new DownloadProgress
        {
            FileName = "model.onnx",
            BytesDownloaded = 500,
            TotalBytes = 1000
        };

        progress.OverallPercentComplete.Should().BeApproximately(50.0, 0.01);
    }

    [Fact]
    public void OverallPercentComplete_FirstFileHalfDone_CalculatesCorrectly()
    {
        // File 1 of 4, 50% done => (0 * 100 + 50) / 4 = 12.5%
        var progress = new DownloadProgress
        {
            FileName = "file1.bin",
            BytesDownloaded = 500,
            TotalBytes = 1000,
            CurrentFileIndex = 1,
            TotalFileCount = 4
        };

        progress.OverallPercentComplete.Should().BeApproximately(12.5, 0.01);
    }

    [Fact]
    public void OverallPercentComplete_ThirdFileFullyDone_CalculatesCorrectly()
    {
        // File 3 of 4, 100% done => (2 * 100 + 100) / 4 = 75%
        var progress = new DownloadProgress
        {
            FileName = "file3.bin",
            BytesDownloaded = 1000,
            TotalBytes = 1000,
            CurrentFileIndex = 3,
            TotalFileCount = 4
        };

        progress.OverallPercentComplete.Should().BeApproximately(75.0, 0.01);
    }

    [Fact]
    public void OverallPercentComplete_LastFileFullyDone_Returns100()
    {
        // File 4 of 4, 100% done => (3 * 100 + 100) / 4 = 100%
        var progress = new DownloadProgress
        {
            FileName = "file4.bin",
            BytesDownloaded = 2048,
            TotalBytes = 2048,
            CurrentFileIndex = 4,
            TotalFileCount = 4
        };

        progress.OverallPercentComplete.Should().BeApproximately(100.0, 0.01);
    }

    [Fact]
    public void OverallPercentComplete_NeverExceeds100()
    {
        // Even with 77 files (the original bug scenario), overall should never exceed 100
        for (int i = 1; i <= 77; i++)
        {
            var progress = new DownloadProgress
            {
                FileName = $"file{i}.bin",
                BytesDownloaded = 1000,
                TotalBytes = 1000,
                CurrentFileIndex = i,
                TotalFileCount = 77
            };

            progress.OverallPercentComplete.Should().BeLessThanOrEqualTo(100.0,
                $"file {i} of 77 should not exceed 100%");
        }
    }

    [Fact]
    public void OverallPercentComplete_MonotonicallyIncreases_AcrossFiles()
    {
        var totalFiles = 5;
        double lastOverall = -1;

        for (int fileIdx = 1; fileIdx <= totalFiles; fileIdx++)
        {
            // Simulate 0%, 50%, 100% progress within each file
            foreach (var pct in new[] { 0.0, 0.5, 1.0 })
            {
                var totalBytes = 1000L;
                var progress = new DownloadProgress
                {
                    FileName = $"file{fileIdx}.bin",
                    BytesDownloaded = (long)(totalBytes * pct),
                    TotalBytes = totalBytes,
                    CurrentFileIndex = fileIdx,
                    TotalFileCount = totalFiles
                };

                progress.OverallPercentComplete.Should().BeGreaterThanOrEqualTo(lastOverall,
                    $"progress should never decrease (file {fileIdx}, {pct:P0})");
                lastOverall = progress.OverallPercentComplete;
            }
        }

        lastOverall.Should().BeApproximately(100.0, 0.01);
    }

    // --- Default values ---

    [Fact]
    public void DefaultValues_FileIndexAndCount_AreZero()
    {
        var progress = new DownloadProgress
        {
            FileName = "model.onnx",
            BytesDownloaded = 0,
            TotalBytes = 100
        };

        progress.CurrentFileIndex.Should().Be(0);
        progress.TotalFileCount.Should().Be(0);
    }
}
