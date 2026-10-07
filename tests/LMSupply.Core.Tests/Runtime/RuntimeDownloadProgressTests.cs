using LMSupply.Runtime;

namespace LMSupply.Core.Tests.Runtime;

/// <summary>
/// A first load reports the model download and the runtime download through one callback. Runtime reports carry
/// <see cref="DownloadKind.Runtime"/> and a real phase and file name, so a progress view can say "preparing the runtime"
/// instead of showing a silent or mislabelled step.
/// </summary>
public class RuntimeDownloadProgressTests
{
    private sealed class Recorder : IProgress<DownloadProgress>
    {
        public List<DownloadProgress> Reports { get; } = [];

        public void Report(DownloadProgress value) => Reports.Add(value);
    }

    [Fact]
    public void Wrapped_Reports_AreTaggedAsRuntime_AndKeepEverythingElse()
    {
        var recorder = new Recorder();
        var progress = RuntimeDownloadProgress.Wrap(recorder)!;

        progress.Report(new DownloadProgress { FileName = "runtime.nupkg", BytesDownloaded = 10, TotalBytes = 40, BytesPerSecond = 5 });

        var report = Assert.Single(recorder.Reports);
        Assert.Equal(DownloadKind.Runtime, report.Kind);
        Assert.Equal("runtime.nupkg", report.FileName);
        Assert.Equal(10, report.BytesDownloaded);
        Assert.Equal(40, report.TotalBytes);
        Assert.Equal(5, report.BytesPerSecond);
    }

    [Fact]
    public void Wrap_PassesNullThrough_AndDoesNotWrapTwice()
    {
        Assert.Null(RuntimeDownloadProgress.Wrap(null));

        var once = RuntimeDownloadProgress.Wrap(new Recorder());
        Assert.Same(once, RuntimeDownloadProgress.Wrap(once));
    }

    [Fact]
    public void ModelReports_StayModelReports()
    {
        Assert.Equal(DownloadKind.Model, new DownloadProgress { FileName = "model.onnx" }.Kind);
    }

    [Fact]
    public void ACachedRuntime_IsOneCompleteReport_NamingTheRuntime()
    {
        var recorder = new Recorder();

        OnnxNuGetDownloader.ReportCacheHit(recorder, "onnxruntime");

        var report = Assert.Single(recorder.Reports);
        Assert.Equal(DownloadKind.Runtime, report.Kind);
        Assert.Equal(DownloadPhase.Complete, report.Phase);
        Assert.Equal("onnxruntime", report.FileName);
    }
}
