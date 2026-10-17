using LMSupply.Download;

namespace LMSupply.Core.Tests.Download;

/// <summary>
/// A load completes each file and each runtime once, however many of its pieces report it — a load that creates three
/// runtime sessions used to complete the ONNX Runtime three times.
/// </summary>
public sealed class LoadProgressTests
{
    [Fact]
    public void ASecondComplete_OfTheSameFile_IsDropped_AndEverythingElsePasses()
    {
        var seen = new List<DownloadProgress>();
        var progress = LoadProgress.Begin(new Sink(seen.Add))!;

        progress.Report(Complete("onnxruntime", DownloadKind.Runtime));
        progress.Report(new DownloadProgress { FileName = "unet/model.onnx", Phase = DownloadPhase.Loading });
        progress.Report(Complete("onnxruntime", DownloadKind.Runtime));
        progress.Report(new DownloadProgress { FileName = "vae_decoder/model.onnx", Phase = DownloadPhase.Loading });
        progress.Report(Complete("onnxruntime", DownloadKind.Runtime));

        Assert.Equal(
            ["Runtime:onnxruntime:Complete", "Model:unet/model.onnx:Loading", "Model:vae_decoder/model.onnx:Loading"],
            seen.Select(r => $"{r.Kind}:{r.FileName}:{r.Phase}"));
    }

    [Fact]
    public void AModelFile_AndARuntime_OfTheSameName_AreTwoThings()
    {
        var seen = new List<DownloadProgress>();
        var progress = LoadProgress.Begin(new Sink(seen.Add))!;

        progress.Report(Complete("llama-server", DownloadKind.Runtime));
        progress.Report(Complete("llama-server", DownloadKind.Model));

        Assert.Equal(2, seen.Count);
    }

    [Fact]
    public void ALoadThatDelegates_StaysOneLoad_AndTheNextLoadReportsAgain()
    {
        var seen = new List<DownloadProgress>();
        var sink = new Sink(seen.Add);

        var first = LoadProgress.Begin(sink);
        Assert.Same(first, LoadProgress.Begin(first));
        first!.Report(Complete("onnxruntime", DownloadKind.Runtime));
        LoadProgress.Begin(first)!.Report(Complete("onnxruntime", DownloadKind.Runtime));

        // The caller reuses its reporter for another load: that load reports its runtime step too.
        LoadProgress.Begin(sink)!.Report(Complete("onnxruntime", DownloadKind.Runtime));

        Assert.Equal(2, seen.Count);
        Assert.Null(LoadProgress.Begin(null));
    }

    private static DownloadProgress Complete(string fileName, DownloadKind kind) =>
        new() { FileName = fileName, Kind = kind, Phase = DownloadPhase.Complete };

    private sealed class Sink(Action<DownloadProgress> report) : IProgress<DownloadProgress>
    {
        public void Report(DownloadProgress value) => report(value);
    }
}
