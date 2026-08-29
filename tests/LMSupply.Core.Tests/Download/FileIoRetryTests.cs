using AwesomeAssertions;
using LMSupply.Core.Download;

namespace LMSupply.Core.Tests.Download;

/// <summary>
/// Regression teeth for the transient file-lock retry used by the download-to-load handoff
/// (rename racing a second process/AV scanner opening the destination, or two callers racing
/// to acquire the same ".part" write handle). Found by dogfooding downstream — a raw
/// <see cref="IOException"/> from this window was bubbling straight to the caller with no
/// recovery attempt (docket iyulab/lm-supply#100).
/// </summary>
public class FileIoRetryTests
{
    [Fact]
    public async Task ExecuteAsync_SucceedsOnFirstAttempt_DoesNotRetry()
    {
        var calls = 0;

        var result = await FileIoRetry.ExecuteAsync(() =>
        {
            calls++;
            return 42;
        }, TestContext.Current.CancellationToken);

        result.Should().Be(42);
        calls.Should().Be(1);
    }

    [Fact]
    public async Task ExecuteAsync_RecoversAfterTransientIOExceptions()
    {
        var calls = 0;

        var result = await FileIoRetry.ExecuteAsync(() =>
        {
            calls++;
            if (calls < 3)
                throw new IOException("simulated transient lock");
            return "recovered";
        }, TestContext.Current.CancellationToken);

        result.Should().Be("recovered");
        calls.Should().Be(3);
    }

    [Fact]
    public async Task ExecuteAsync_ExhaustsAttempts_RethrowsLastIOException()
    {
        var calls = 0;

        var act = async () => await FileIoRetry.ExecuteAsync<object?>(() =>
        {
            calls++;
            throw new IOException($"attempt {calls}");
        });

        await act.Should().ThrowAsync<IOException>().WithMessage("attempt 5");
        calls.Should().Be(5);
    }

    [Fact]
    public async Task ExecuteAsync_NonIOException_PropagatesImmediately_NoRetry()
    {
        var calls = 0;

        var act = async () => await FileIoRetry.ExecuteAsync<object?>(() =>
        {
            calls++;
            throw new InvalidOperationException("not a lock");
        });

        await act.Should().ThrowAsync<InvalidOperationException>();
        calls.Should().Be(1);
    }

    [Fact]
    public async Task ExecuteAsync_TaskOverload_RecoversAfterTransientIOExceptions()
    {
        var calls = 0;

        var result = await FileIoRetry.ExecuteAsync(async () =>
        {
            calls++;
            await Task.Yield();
            if (calls < 2)
                throw new IOException("simulated transient lock");
            return "ok";
        }, TestContext.Current.CancellationToken);

        result.Should().Be("ok");
        calls.Should().Be(2);
    }

    /// <summary>
    /// The scenario the fix targets directly: a real, exclusively-locked file (as
    /// <c>FileShare.None</c> writers/movers produce) that a concurrent actor releases shortly
    /// after — the caller should recover instead of bubbling the raw <see cref="IOException"/>.
    /// </summary>
    [Fact]
    public async Task ExecuteAsync_RecoversFromRealExclusiveFileLock()
    {
        var path = Path.Combine(Path.GetTempPath(), $"file-io-retry-{Guid.NewGuid():N}.tmp");
        await File.WriteAllTextAsync(path, "locked", TestContext.Current.CancellationToken);

        try
        {
            var locker = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.None);
            var releaseAfter = Task.Delay(300, TestContext.Current.CancellationToken).ContinueWith(_ => locker.Dispose(), TestContext.Current.CancellationToken);

            using var opened = await FileIoRetry.ExecuteAsync(() => new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read), TestContext.Current.CancellationToken);

            opened.Should().NotBeNull();
            await releaseAfter;
        }
        finally
        {
            File.Delete(path);
        }
    }
}
