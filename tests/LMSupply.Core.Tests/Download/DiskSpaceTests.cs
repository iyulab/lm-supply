using System.Net;
using AwesomeAssertions;
using LMSupply.Download;
using LMSupply.Exceptions;

namespace LMSupply.Core.Tests.Download;

/// <summary>
/// A download onto a volume without room fails at once, with a typed exception that says how much is needed and how much is
/// free, instead of writing until the disk is full and retrying a write that cannot succeed.
/// </summary>
public sealed class DiskSpaceTests : IDisposable
{
    private readonly string _dir = Path.Combine(Path.GetTempPath(), "lmsupply-space-" + Guid.NewGuid().ToString("N"));

    private static CancellationToken Ct => TestContext.Current.CancellationToken;

    public DiskSpaceTests() => Directory.CreateDirectory(_dir);

    public void Dispose() => Directory.Delete(_dir, recursive: true);

    private sealed class Server(byte[] body, Func<Stream>? stream = null) : HttpMessageHandler
    {
        public int Calls { get; private set; }

        public bool BodyWasRead { get; private set; }

        protected override Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken)
        {
            Calls++;
            var content = new StreamContent(new Tracking(stream?.Invoke() ?? new MemoryStream(body), () => BodyWasRead = true));
            content.Headers.ContentLength = body.Length;
            return Task.FromResult(new HttpResponseMessage(HttpStatusCode.OK) { Content = content });
        }
    }

    /// <summary>A body that notes that it was read.</summary>
    private sealed class Tracking(Stream inner, Action read) : Stream
    {
        public override bool CanRead => true;

        public override bool CanSeek => false;

        public override bool CanWrite => false;

        public override long Length => throw new NotSupportedException();

        public override long Position { get => throw new NotSupportedException(); set => throw new NotSupportedException(); }

        public override void Flush()
        {
        }

        public override int Read(byte[] buffer, int offset, int count)
        {
            read();
            return inner.Read(buffer, offset, count);
        }

        public override async ValueTask<int> ReadAsync(Memory<byte> buffer, CancellationToken cancellationToken = default)
        {
            read();
            return await inner.ReadAsync(buffer, cancellationToken);
        }

        public override long Seek(long offset, SeekOrigin origin) => throw new NotSupportedException();

        public override void SetLength(long value) => throw new NotSupportedException();

        public override void Write(byte[] buffer, int offset, int count) => throw new NotSupportedException();
    }

    private ResumableFileDownload.Request RequestFor(long? free) => new()
    {
        Url = "https://example.test/model.bin",
        DestinationPath = Path.Combine(_dir, "model.bin"),
        FileName = "model.bin",
        FreeBytes = _ => free,
    };

    [Fact]
    public async Task AVolumeWithoutRoom_RefusesBeforeAByteIsWritten_AndSaysHowMuchIsNeeded()
    {
        var server = new Server(new byte[1000]);
        using var http = new HttpClient(server);

        var failure = await Assert.ThrowsAsync<InsufficientDiskSpaceException>(
            () => ResumableFileDownload.DownloadAsync(http, RequestFor(free: 100), Ct));

        failure.RequiredBytes.Should().Be(1000);
        failure.AvailableBytes.Should().Be(100);
        failure.Message.Should().Contain("model.bin").And.Contain("1,000").And.Contain("100");
        server.BodyWasRead.Should().BeFalse("the refusal came before the body was read");
        server.Calls.Should().Be(1, "a volume without room is not a transient failure: it is not tried again");
        File.Exists(Path.Combine(_dir, "model.bin")).Should().BeFalse();
        File.Exists(Path.Combine(_dir, "model.bin.part")).Should().BeFalse("nothing was written, so nothing is left to occupy the volume");
    }

    [Fact]
    public async Task AVolumeWithRoom_DownloadsAsBefore()
    {
        using var http = new HttpClient(new Server(new byte[1000]));

        await ResumableFileDownload.DownloadAsync(http, RequestFor(free: 1000), Ct);

        new FileInfo(Path.Combine(_dir, "model.bin")).Length.Should().Be(1000, "exactly the room it needs is room enough");
    }

    [Fact]
    public async Task AVolumeWhoseFreeSpaceCannotBeTold_IsNotRefusedForIt()
    {
        using var http = new HttpClient(new Server(new byte[1000]));

        await ResumableFileDownload.DownloadAsync(http, RequestFor(free: null), Ct);

        File.Exists(Path.Combine(_dir, "model.bin")).Should().BeTrue();
    }

    [Fact]
    public async Task AWriteThatFailsBecauseTheDiskIsFull_IsNotRetried_AndIsTheSameTypedException()
    {
        // The disk filling halfway shows as an IOException that carries the platform's "no space" code. Here it is raised from the
        // body's stream, which reaches the same handler as a failed write does.
        var server = new Server(new byte[1000], stream: () => new FailingStream(new IOException("No space left on device", hresult: 28)));
        using var http = new HttpClient(server);

        var failure = await Assert.ThrowsAsync<InsufficientDiskSpaceException>(
            () => ResumableFileDownload.DownloadAsync(http, RequestFor(free: null), Ct));

        failure.InnerException.Should().BeOfType<IOException>();
        server.Calls.Should().Be(1, "retrying a write that cannot succeed only delays the answer");
    }

    private sealed class FailingStream(IOException failure) : Stream
    {
        public override bool CanRead => true;

        public override bool CanSeek => false;

        public override bool CanWrite => false;

        public override long Length => throw new NotSupportedException();

        public override long Position { get => throw new NotSupportedException(); set => throw new NotSupportedException(); }

        public override void Flush()
        {
        }

        public override int Read(byte[] buffer, int offset, int count) => throw failure;

        public override ValueTask<int> ReadAsync(Memory<byte> buffer, CancellationToken cancellationToken = default) => throw failure;

        public override long Seek(long offset, SeekOrigin origin) => throw new NotSupportedException();

        public override void SetLength(long value) => throw new NotSupportedException();

        public override void Write(byte[] buffer, int offset, int count) => throw new NotSupportedException();
    }

    [Theory]
    [InlineData(28, true)]
    [InlineData(69, true)]
    [InlineData(122, true)]
    [InlineData(unchecked((int)0x80070070), true)]
    [InlineData(unchecked((int)0x80070027), true)]
    [InlineData(5, false)]
    [InlineData(32, false)]
    [InlineData(unchecked((int)0x80070020), false)]
    public void OnlyTheCodesOfAFullDisk_AreAFullDisk(int hresult, bool full)
    {
        DiskSpace.IsFull(new IOException("failed", hresult)).Should().Be(full);
    }

    [Fact]
    public void TheFreeSpaceOfARealDirectory_IsKnown()
    {
        DiskSpace.FreeBytes(Path.Combine(_dir, "not-yet")).Should().BeGreaterThan(0);
    }
}
