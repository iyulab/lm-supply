using AwesomeAssertions;
using LMSupply.Lifecycle;
using Xunit;

namespace LMSupply.Core.Tests.Lifecycle;

/// <summary>
/// <see cref="ModelLifecycle"/>: a status read never downloads; a download (the consent) is shared by its callers, streamed to
/// watchers as it runs, can be cancelled by any caller, and a failure stays visible until the next download.
/// </summary>
public sealed class ModelLifecycleTests
{
    private static CancellationToken Ct => TestContext.Current.CancellationToken;

    [Fact]
    public async Task AStatusRead_LooksAtTheCacheOnly_AndNeverDownloads()
    {
        var generator = new FakeArtifacts("generator");
        var lifecycle = new ModelLifecycle([generator]);

        (await lifecycle.GetStatusAsync("generator", "m", Ct)).State.Should().Be(ModelState.Missing);
        generator.Cached.Add("m");
        (await lifecycle.GetStatusAsync("generator", "m", Ct)).State.Should().Be(ModelState.Ready);

        generator.Downloads.Should().Be(0);
    }

    [Fact]
    public async Task ADownload_IsWatchedFromStartToReady_AndTwoCallersShareIt()
    {
        var generator = new FakeArtifacts("generator") { Gate = new TaskCompletionSource() };
        var lifecycle = new ModelLifecycle([generator]);
        using var watchCts = CancellationTokenSource.CreateLinkedTokenSource(Ct);
        var seen = new List<ModelStatus>();
        var watching = Task.Run(async () =>
        {
            await foreach (var status in lifecycle.WatchAsync(watchCts.Token))
            {
                lock (seen) seen.Add(status);
                if (status.State == ModelState.Ready) break;
            }
        }, Ct);
        await Task.Delay(50, Ct); // the watcher is subscribed

        var first = lifecycle.DownloadAsync("generator", "m", Ct);
        var second = lifecycle.DownloadAsync("generator", "m", Ct);
        await generator.Started.Task.WaitAsync(Ct);

        var during = await lifecycle.GetStatusAsync("generator", "m", Ct);
        during.State.Should().Be(ModelState.Downloading);

        generator.Cached.Add("m");
        generator.Gate.SetResult();
        (await first).State.Should().Be(ModelState.Ready);
        (await second).State.Should().Be(ModelState.Ready);
        await watching.WaitAsync(TimeSpan.FromSeconds(10), Ct);

        generator.Downloads.Should().Be(1, "two callers for one model share one download");
        lock (seen)
        {
            seen.First().State.Should().Be(ModelState.Downloading);
            seen.Should().Contain(s => s.State == ModelState.Downloading && s.Progress != null, "progress is streamed while it runs");
            seen.Last().State.Should().Be(ModelState.Ready);
        }
    }

    [Fact]
    public async Task CancelDownload_StopsIt_ForEveryCaller_AndTheModelIsMissingAgain()
    {
        var generator = new FakeArtifacts("generator") { Gate = new TaskCompletionSource() };
        var lifecycle = new ModelLifecycle([generator]);

        var waiting = lifecycle.DownloadAsync("generator", "m", Ct);
        await generator.Started.Task.WaitAsync(Ct);

        lifecycle.CancelDownload("generator", "m").Should().BeTrue();

        await waiting.Awaiting(t => t).Should().ThrowAsync<OperationCanceledException>();
        (await lifecycle.GetStatusAsync("generator", "m", Ct)).State.Should().Be(ModelState.Missing);
        lifecycle.CancelDownload("generator", "m").Should().BeFalse("nothing is running any more");
    }

    [Fact]
    public async Task ACallerThatStopsWaiting_DoesNotStopTheDownloadForOthers()
    {
        var generator = new FakeArtifacts("generator") { Gate = new TaskCompletionSource() };
        var lifecycle = new ModelLifecycle([generator]);
        using var impatient = new CancellationTokenSource();

        var leaving = lifecycle.DownloadAsync("generator", "m", impatient.Token);
        var staying = lifecycle.DownloadAsync("generator", "m", Ct);
        await generator.Started.Task.WaitAsync(Ct);
        await impatient.CancelAsync();
        await leaving.Awaiting(t => t).Should().ThrowAsync<OperationCanceledException>();

        generator.Cached.Add("m");
        generator.Gate.SetResult();

        (await staying).State.Should().Be(ModelState.Ready);
    }

    [Fact]
    public async Task AFailedDownload_IsReportedWithItsError_UntilTheNextDownload()
    {
        var generator = new FakeArtifacts("generator") { Failure = new InvalidOperationException("no build for this platform") };
        var lifecycle = new ModelLifecycle([generator]);

        await lifecycle.Awaiting(l => l.DownloadAsync("generator", "m", Ct)).Should().ThrowAsync<InvalidOperationException>();

        var failed = await lifecycle.GetStatusAsync("generator", "m", Ct);
        failed.State.Should().Be(ModelState.Failed);
        failed.Error!.Message.Should().Be("no build for this platform");

        generator.Failure = null;
        generator.Cached.Add("m");
        (await lifecycle.DownloadAsync("generator", "m", Ct)).State.Should().Be(ModelState.Ready);
        (await lifecycle.GetStatusAsync("generator", "m", Ct)).State.Should().Be(ModelState.Ready);
    }

    [Fact]
    public async Task AnUnknownKind_IsRefused_AndKindsAreDistinct()
    {
        var lifecycle = new ModelLifecycle([new FakeArtifacts("generator"), new FakeArtifacts("embedder")]);

        lifecycle.Kinds.Should().BeEquivalentTo(["generator", "embedder"]);
        await lifecycle.Awaiting(l => l.GetStatusAsync("reranker", "m", Ct)).Should().ThrowAsync<ArgumentException>();
        FluentActions.Invoking(() => new ModelLifecycle([new FakeArtifacts("generator"), new FakeArtifacts("Generator")]))
            .Should().Throw<ArgumentException>();
    }

    private sealed class FakeArtifacts(string kind) : IModelArtifacts
    {
        private int _downloads;

        public HashSet<string> Cached { get; } = [];
        public TaskCompletionSource? Gate { get; init; }
        public TaskCompletionSource Started { get; } = new(TaskCreationOptions.RunContinuationsAsynchronously);
        public Exception? Failure { get; set; }
        public int Downloads => Volatile.Read(ref _downloads);

        public string Kind => kind;

        public Task<bool> IsAvailableOfflineAsync(string modelId, CancellationToken cancellationToken = default)
        {
            lock (Cached) return Task.FromResult(Cached.Contains(modelId));
        }

        public Task<long> GetRemainingDownloadBytesAsync(string modelId, CancellationToken cancellationToken = default) => Task.FromResult(100L);

        public async Task DownloadAsync(string modelId, IProgress<DownloadProgress>? progress = null, CancellationToken cancellationToken = default)
        {
            Interlocked.Increment(ref _downloads);
            progress?.Report(new DownloadProgress { FileName = "model.gguf", BytesDownloaded = 10, TotalBytes = 100 });
            Started.TrySetResult();
            if (Failure is { } failure)
                throw failure;
            if (Gate is { } gate)
                await gate.Task.WaitAsync(cancellationToken);
        }
    }
}
