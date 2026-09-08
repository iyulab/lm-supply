using AwesomeAssertions;
using LMSupply.Download;

namespace LMSupply.Core.Tests.Download;

/// <summary>
/// The download loop reports once per 16 KB read; a 470 MB model produced 30,006 callbacks, which
/// a consumer binding <c>Progress&lt;T&gt;</c> to a UI thread receives as 30,006 posts. The
/// coalescer forwards first/last/percent-step/interval/file-change reports and drops the rest.
/// </summary>
public class CoalescingProgressTests
{
    private sealed class Sink : IProgress<DownloadProgress>
    {
        public List<DownloadProgress> Reports { get; } = [];
        public void Report(DownloadProgress value) => Reports.Add(value);
    }

    private static DownloadProgress At(long bytes, long total = 100_000_000, string file = "model.onnx") =>
        new() { FileName = file, BytesDownloaded = bytes, TotalBytes = total };

    [Fact]
    public void SixteenKbReads_OverAHundredMegabytes_ForwardAboutOnePerPercent()
    {
        var sink = new Sink();
        var clock = TimeSpan.Zero;
        var coalesced = new CoalescingProgress(sink, () => clock);
        const long total = 100_000_000;

        for (long b = 16_384; b < total; b += 16_384)
            coalesced.Report(At(b, total));
        coalesced.Report(At(total, total));

        var reads = total / 16_384;
        reads.Should().BeGreaterThan(6000, "the raw stream really is this chatty");
        sink.Reports.Count.Should().BeInRange(100, 110, "one report per percent plus first and last");
        sink.Reports.First().BytesDownloaded.Should().Be(16_384, "the first report always goes through");
        sink.Reports.Last().BytesDownloaded.Should().Be(total, "the completing report always goes through");
    }

    [Fact]
    public void QuietInterval_ForwardsEvenWithoutAPercentStep()
    {
        // A slow link: bytes trickle in far below one percent per read, but the consumer must still
        // see the bar move at least four times a second.
        var sink = new Sink();
        var clock = TimeSpan.Zero;
        var coalesced = new CoalescingProgress(sink, () => clock);

        coalesced.Report(At(1_000));                       // first — forwarded
        clock += TimeSpan.FromMilliseconds(100);
        coalesced.Report(At(2_000));                       // 0.001% later, 100 ms — dropped
        clock += TimeSpan.FromMilliseconds(200);
        coalesced.Report(At(3_000));                       // 300 ms since the last forwarded — forwarded

        sink.Reports.Select(r => r.BytesDownloaded).Should().Equal(1_000, 3_000);
    }

    [Fact]
    public void FileChange_IsAlwaysForwarded()
    {
        var sink = new Sink();
        var clock = TimeSpan.Zero;
        var coalesced = new CoalescingProgress(sink, () => clock);

        coalesced.Report(At(16_384, file: "a.bin"));
        coalesced.Report(At(32_768, file: "a.bin"));       // dropped
        coalesced.Report(At(16_384, file: "b.bin"));       // new file — forwarded

        sink.Reports.Select(r => r.FileName).Should().Equal("a.bin", "b.bin");
    }

    [Fact]
    public void Completion_IsForwarded_EvenRightAfterAForwardedReport()
    {
        var sink = new Sink();
        var clock = TimeSpan.Zero;
        var coalesced = new CoalescingProgress(sink, () => clock);

        coalesced.Report(At(99_990_000));                  // first
        coalesced.Report(At(100_000_000));                 // +0.01%, 0 ms, but complete — forwarded

        sink.Reports.Should().HaveCount(2);
        sink.Reports.Last().PercentComplete.Should().Be(100);
    }

    [Fact]
    public void Wrap_ReturnsNullForNull_SoTheLoopKeepsItsNullCheck()
    {
        CoalescingProgress.Wrap(null).Should().BeNull();
        CoalescingProgress.Wrap(new Sink()).Should().BeOfType<CoalescingProgress>();
    }
}
