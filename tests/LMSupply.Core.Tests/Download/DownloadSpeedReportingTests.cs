using System.Net;
using System.Net.Http.Headers;
using AwesomeAssertions;
using LMSupply.Download;

namespace LMSupply.Core.Tests.Download;

/// <summary>
/// <see cref="DownloadProgress.BytesPerSecond"/> and <see cref="DownloadProgress.EstimatedRemaining"/> are filled where the
/// bytes arrive — <c>ResumableFileDownload</c>, which every download in the library goes through — so a caller's progress
/// view shows a speed and a time left for model files, runtime packages and server builds alike.
/// </summary>
public sealed class DownloadSpeedReportingTests : IDisposable
{
    private const int ChunkSize = 40_000;

    private readonly string _dir = Path.Combine(Path.GetTempPath(), "lmsupply-speed-" + Guid.NewGuid().ToString("N"));

    private static CancellationToken Ct => TestContext.Current.CancellationToken;

    public DownloadSpeedReportingTests() => Directory.CreateDirectory(_dir);

    public void Dispose() => Directory.Delete(_dir, recursive: true);

    private sealed class Sink : IProgress<DownloadProgress>
    {
        public List<DownloadProgress> Reports { get; } = [];
        public void Report(DownloadProgress value) => Reports.Add(value);
    }

    /// <summary>Serves a file (or the tail of it, for a range request) in slow chunks; the sleep outlasts the tracker's 200 ms sample.</summary>
    private sealed class SlowServer(long total, int chunks) : HttpMessageHandler
    {
        protected override Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken)
        {
            var from = request.Headers.Range?.Ranges.FirstOrDefault()?.From ?? 0;
            var remaining = total - from;
            var response = new HttpResponseMessage(from > 0 ? HttpStatusCode.PartialContent : HttpStatusCode.OK)
            {
                Content = new StreamContent(new SlowStream(remaining, chunks)),
            };
            response.Content.Headers.ContentLength = remaining;
            if (from > 0)
                response.Content.Headers.ContentRange = new ContentRangeHeaderValue(from, total - 1, total);
            return Task.FromResult(response);
        }
    }

    private sealed class SlowStream(long length, int chunks) : Stream
    {
        private long _sent;
        private int _chunk;

        public override async ValueTask<int> ReadAsync(Memory<byte> buffer, CancellationToken cancellationToken = default)
        {
            if (_sent >= length)
                return 0;
            await Task.Delay(260, cancellationToken);
            var n = (int)Math.Min(Math.Min(buffer.Length, length / chunks + 1), length - _sent);
            buffer.Span[..n].Fill(7);
            _sent += n;
            _chunk++;
            return n;
        }

        public override bool CanRead => true;
        public override bool CanSeek => false;
        public override bool CanWrite => false;
        public override long Length => length;
        public override long Position { get => _sent; set => throw new NotSupportedException(); }
        public override void Flush() { }
        public override int Read(byte[] buffer, int offset, int count) => throw new NotSupportedException();
        public override long Seek(long offset, SeekOrigin origin) => throw new NotSupportedException();
        public override void SetLength(long value) => throw new NotSupportedException();
        public override void Write(byte[] buffer, int offset, int count) => throw new NotSupportedException();
    }

    private static ResumableFileDownload.Request RequestFor(string path, long expected, Sink sink) => new()
    {
        Url = "https://example.test/model.bin",
        DestinationPath = path,
        FileName = "model.bin",
        ExpectedSize = expected,
        Progress = sink,
    };

    [Fact]
    public async Task ASlowDownload_ReportsASpeedAndATimeLeft_WhileBytesMove()
    {
        const long total = ChunkSize * 4;
        var sink = new Sink();
        using var http = new HttpClient(new SlowServer(total, chunks: 4));

        await ResumableFileDownload.DownloadAsync(http, RequestFor(Path.Combine(_dir, "model.bin"), total, sink), Ct);

        var measured = sink.Reports.Where(r => r.BytesPerSecond is not null).ToList();
        measured.Should().NotBeEmpty("a transfer that lasts a second has a speed from its second report on");
        measured.Should().OnlyContain(r => r.BytesPerSecond > 0);
        measured.First().BytesPerSecond.Should().BeInRange(20_000, 400_000, "40 KB per 260 ms is about 150 KB/s");
        measured.First().SpeedDisplay.Should().EndWith("KB/s");

        var withEta = sink.Reports.Where(r => r.EstimatedRemaining is not null).ToList();
        withEta.Should().NotBeEmpty();
        withEta.Should().OnlyContain(r => r.BytesDownloaded < r.TotalBytes, "nothing is left to wait for once the file is whole");
        withEta.First().EtaDisplay.Should().NotBeEmpty();
    }

    [Fact]
    public async Task AResumedDownload_DoesNotCountTheBytesItAlreadyHeld_AsTransferred()
    {
        const long total = 10_000_000;
        const long held = 9_880_000;
        var path = Path.Combine(_dir, "model.bin");
        await File.WriteAllBytesAsync(path + ".part", new byte[held], Ct);
        var sink = new Sink();
        using var http = new HttpClient(new SlowServer(total, chunks: 3));

        await ResumableFileDownload.DownloadAsync(http, RequestFor(path, total, sink), Ct);

        var measured = sink.Reports.Where(r => r.BytesPerSecond is not null).ToList();
        measured.Should().NotBeEmpty();
        // 120 KB came over the wire in about 0.8 s. Counting the 9.88 MB already on disk as part of the first
        // sample would claim tens of MB/s.
        measured.Should().OnlyContain(r => r.BytesPerSecond < 2_000_000, "only the resumed bytes were transferred");
    }

    [Fact]
    public async Task AFileInPlace_ReportsNoSpeed()
    {
        var path = Path.Combine(_dir, "cached.bin");
        await File.WriteAllBytesAsync(path, new byte[100], Ct);

        var report = DownloadProgress.FileInPlace("cached.bin", path);

        report.BytesPerSecond.Should().BeNull();
        report.EstimatedRemaining.Should().BeNull();
        report.SpeedDisplay.Should().BeEmpty();
        report.EtaDisplay.Should().BeEmpty();
    }

    [Fact]
    public void ATracker_StartedOnAResume_TakesTheHeldBytesAsItsBaseline()
    {
        var tracker = new DownloadProgressTracker(sampleWindowSize: 5, sampleIntervalMs: 0);
        tracker.Start(alreadyDownloadedBytes: 9_000_000);
        Thread.Sleep(50);

        var report = tracker.CreateProgress("m.bin", bytesDownloaded: 9_001_000, totalBytes: 10_000_000);

        report.BytesPerSecond.Should().BeLessThan(1_000_000, "1,000 bytes arrived, not 9,001,000");
    }

    [Fact]
    public void AcrossSeveralFiles_TheTimeLeftIsTheWholeDownloads_NotTheCurrentFiles()
    {
        var sink = new Sink();
        // The third of three files, 1,000 bytes of 9,000 done before it; 500 of its 4,000 now, at 100 B/s.
        var multi = new HuggingFaceDownloader.MultiFileProgress(sink, currentFileIndex: 2, totalFileCount: 3, bytesBefore: 1_000, overallTotal: 9_000);

        multi.Report(new DownloadProgress
        {
            FileName = "b.bin", BytesDownloaded = 500, TotalBytes = 4_000,
            BytesPerSecond = 100, EstimatedRemaining = TimeSpan.FromSeconds(35),
        });

        var report = sink.Reports.Single();
        report.OverallBytesDownloaded.Should().Be(1_500);
        report.EstimatedRemaining.Should().Be(TimeSpan.FromSeconds(75), "(9,000 - 1,500) bytes at 100 B/s, not the file's 35 s");
    }

    [Fact]
    public void WithoutASpeedOrATotal_TheFilesOwnTimeLeftStands()
    {
        var sink = new Sink();
        var unknownTotal = new HuggingFaceDownloader.MultiFileProgress(sink, 1, 2, bytesBefore: null, overallTotal: null);
        var noSpeed = new HuggingFaceDownloader.MultiFileProgress(sink, 1, 2, bytesBefore: 0, overallTotal: 9_000);

        unknownTotal.Report(new DownloadProgress { FileName = "a", BytesDownloaded = 1, TotalBytes = 10, BytesPerSecond = 5, EstimatedRemaining = TimeSpan.FromSeconds(2) });
        noSpeed.Report(new DownloadProgress { FileName = "a", BytesDownloaded = 1, TotalBytes = 10 });

        sink.Reports[0].EstimatedRemaining.Should().Be(TimeSpan.FromSeconds(2));
        sink.Reports[1].EstimatedRemaining.Should().BeNull();
    }
}
