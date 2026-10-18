using AwesomeAssertions;
using LMSupply.Lifecycle;
using Xunit;

namespace LMSupply.Core.Tests.Lifecycle;

/// <summary>
/// <see cref="ModelLifecycle"/>: a status read never downloads; a download (the consent) is shared by its callers, streamed to
/// watchers as it runs, can be cancelled by any caller, and a failure stays visible until the next download. A delete is
/// refused while the model downloads, and a download while it is being deleted.
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
    public async Task Delete_RemovesTheModel_AndWatchersSeeItMissing()
    {
        var generator = new FakeArtifacts("generator");
        generator.Cached.Add("m");
        var lifecycle = new ModelLifecycle([generator]);
        using var watchCts = CancellationTokenSource.CreateLinkedTokenSource(Ct);
        var watching = Task.Run(async () =>
        {
            await foreach (var status in lifecycle.WatchAsync(watchCts.Token))
                return status;
            return null;
        }, Ct);
        await Task.Delay(50, Ct); // the watcher is subscribed

        (await lifecycle.DeleteAsync("generator", "m", Ct)).Should().BeTrue();

        (await lifecycle.GetStatusAsync("generator", "m", Ct)).State.Should().Be(ModelState.Missing);
        var published = await watching.WaitAsync(TimeSpan.FromSeconds(10), Ct);
        published!.State.Should().Be(ModelState.Missing);
        (await lifecycle.DeleteAsync("generator", "m", Ct)).Should().BeFalse("nothing is left to delete");
    }

    [Fact]
    public async Task Delete_WhileTheModelDownloads_IsRefused_AndWorksAfterACancel()
    {
        var generator = new FakeArtifacts("generator") { Gate = new TaskCompletionSource() };
        var lifecycle = new ModelLifecycle([generator]);

        var waiting = lifecycle.DownloadAsync("generator", "m", Ct);
        await generator.Started.Task.WaitAsync(Ct);

        await lifecycle.Awaiting(l => l.DeleteAsync("generator", "m", Ct)).Should().ThrowAsync<InvalidOperationException>()
            .WithMessage("*downloading*");
        generator.Deletes.Should().Be(0, "a refused delete touches no file");

        lifecycle.CancelDownload("generator", "m");
        await waiting.Awaiting(t => t).Should().ThrowAsync<OperationCanceledException>();
        generator.Cached.Add("m"); // a partial download left files behind

        (await lifecycle.DeleteAsync("generator", "m", Ct)).Should().BeTrue();
        generator.Deletes.Should().Be(1);
    }

    [Fact]
    public async Task ADownload_WhileTheModelIsBeingDeleted_IsRefused()
    {
        var generator = new FakeArtifacts("generator") { DeleteGate = new ManualResetEventSlim() };
        generator.Cached.Add("m");
        var lifecycle = new ModelLifecycle([generator]);

        var deleting = Task.Run(() => lifecycle.DeleteAsync("generator", "m", Ct), Ct);
        generator.DeleteStarted.Wait(Ct);

        await lifecycle.Awaiting(l => l.DownloadAsync("generator", "m", Ct)).Should().ThrowAsync<InvalidOperationException>()
            .WithMessage("*being deleted*");
        generator.Downloads.Should().Be(0);

        generator.DeleteGate.Set();
        (await deleting).Should().BeTrue();
        generator.Cached.Add("m");
        (await lifecycle.DownloadAsync("generator", "m", Ct)).State.Should().Be(ModelState.Ready, "the delete has returned");
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

    [Fact]
    public async Task CancelWhenAbandoned_TheOnlyWaiterLeaving_StopsTheDownload()
    {
        var generator = new FakeArtifacts("generator") { Gate = new TaskCompletionSource() };
        var lifecycle = new ModelLifecycle([generator]);
        using var dialog = new CancellationTokenSource();

        var waiting = lifecycle.DownloadAsync("generator", "m", DownloadWaitMode.CancelWhenAbandoned, dialog.Token);
        await generator.Started.Task.WaitAsync(Ct);
        await dialog.CancelAsync();

        await waiting.Awaiting(t => t).Should().ThrowAsync<OperationCanceledException>();
        await generator.Cancelled.Task.WaitAsync(TimeSpan.FromSeconds(10), Ct);
        await WaitForStateAsync(lifecycle, ModelState.Missing);
        lifecycle.CancelDownload("generator", "m").Should().BeFalse("the download ended with its last waiter");
    }

    [Fact]
    public async Task CancelWhenAbandoned_AStarts_BJoins_ALeaves_BStillGetsTheModel()
    {
        var generator = new FakeArtifacts("generator") { Gate = new TaskCompletionSource() };
        var lifecycle = new ModelLifecycle([generator]);
        using var starter = new CancellationTokenSource();

        var a = lifecycle.DownloadAsync("generator", "m", DownloadWaitMode.CancelWhenAbandoned, starter.Token);
        await generator.Started.Task.WaitAsync(Ct);
        var b = lifecycle.DownloadAsync("generator", "m", DownloadWaitMode.CancelWhenAbandoned, Ct);
        await starter.CancelAsync();
        await a.Awaiting(t => t).Should().ThrowAsync<OperationCanceledException>();

        generator.Cancelled.Task.IsCompleted.Should().BeFalse("B is still waiting");
        generator.Cached.Add("m");
        generator.Gate.SetResult();
        (await b).State.Should().Be(ModelState.Ready);
        generator.Downloads.Should().Be(1);
    }

    [Fact]
    public async Task CancelWhenAbandoned_TwoCallersStartingInTheSameInstant_DoNotStopEachOther()
    {
        var generator = new FakeArtifacts("generator") { Gate = new TaskCompletionSource() };
        var lifecycle = new ModelLifecycle([generator]);
        using var first = new CancellationTokenSource();
        using var go = new ManualResetEventSlim();

        var starts = Enumerable.Range(0, 2).Select(i => Task.Run(() =>
        {
            go.Wait(Ct);
            // Wrapped so Task.Run hands back the call itself rather than awaiting it.
            return new[] { lifecycle.DownloadAsync("generator", "m", DownloadWaitMode.CancelWhenAbandoned, i == 0 ? first.Token : Ct) };
        }, Ct)).ToArray();
        go.Set();
        var calls = (await Task.WhenAll(starts)).Select(c => c[0]).ToArray();
        await generator.Started.Task.WaitAsync(Ct);

        await first.CancelAsync();
        await calls[0].Awaiting(t => t).Should().ThrowAsync<OperationCanceledException>();

        generator.Cached.Add("m");
        generator.Gate.SetResult();
        (await calls[1]).State.Should().Be(ModelState.Ready, "the other caller is still waiting");
        generator.Downloads.Should().Be(1, "both calls share one download");
        generator.Cancelled.Task.IsCompleted.Should().BeFalse();
    }

    [Fact]
    public async Task CancelWhenAbandoned_AKeepRunningCallerPinsTheDownload()
    {
        var generator = new FakeArtifacts("generator") { Gate = new TaskCompletionSource() };
        var lifecycle = new ModelLifecycle([generator]);
        using var dialog = new CancellationTokenSource();
        using var background = new CancellationTokenSource();

        var a = lifecycle.DownloadAsync("generator", "m", DownloadWaitMode.CancelWhenAbandoned, dialog.Token);
        await generator.Started.Task.WaitAsync(Ct);
        var b = lifecycle.DownloadAsync("generator", "m", DownloadWaitMode.KeepRunning, background.Token);
        await background.CancelAsync();
        await b.Awaiting(t => t).Should().ThrowAsync<OperationCanceledException>();
        await dialog.CancelAsync();
        await a.Awaiting(t => t).Should().ThrowAsync<OperationCanceledException>();

        (await lifecycle.GetStatusAsync("generator", "m", Ct)).State.Should().Be(ModelState.Downloading, "a KeepRunning caller joined it");
        generator.Cached.Add("m");
        generator.Gate.SetResult();
        await WaitForStateAsync(lifecycle, ModelState.Ready);
        generator.Cancelled.Task.IsCompleted.Should().BeFalse();
    }

    [Fact]
    public async Task ACallerArrivingWhileAnAbandonedDownloadWindsDown_StartsANewOne()
    {
        var generator = new FakeArtifacts("generator") { Gate = new TaskCompletionSource(), WindDown = new TaskCompletionSource() };
        var lifecycle = new ModelLifecycle([generator]);
        using var dialog = new CancellationTokenSource();

        var a = lifecycle.DownloadAsync("generator", "m", DownloadWaitMode.CancelWhenAbandoned, dialog.Token);
        await generator.Started.Task.WaitAsync(Ct);
        await dialog.CancelAsync();
        await a.Awaiting(t => t).Should().ThrowAsync<OperationCanceledException>();
        await generator.Cancelled.Task.WaitAsync(TimeSpan.FromSeconds(10), Ct);

        var b = lifecycle.DownloadAsync("generator", "m", Ct); // the old download is still closing its files
        generator.Downloads.Should().Be(1);
        b.IsCompleted.Should().BeFalse("it waits for the abandoned download instead of joining its cancellation");

        generator.WindDown!.SetResult();
        generator.Cached.Add("m");
        generator.Gate.SetResult();
        (await b.WaitAsync(TimeSpan.FromSeconds(10), Ct)).State.Should().Be(ModelState.Ready);
        generator.Downloads.Should().Be(2, "a new download started after the abandoned one ended");
    }

    [Fact]
    public async Task AnUndefinedWaitMode_IsRefused()
    {
        var lifecycle = new ModelLifecycle([new FakeArtifacts("generator")]);
        await lifecycle.Awaiting(l => l.DownloadAsync("generator", "m", (DownloadWaitMode)7, Ct)).Should().ThrowAsync<ArgumentException>();
    }

    private static async Task WaitForStateAsync(ModelLifecycle lifecycle, ModelState state)
    {
        var deadline = DateTime.UtcNow.AddSeconds(10);
        while ((await lifecycle.GetStatusAsync("generator", "m", Ct)).State != state)
        {
            DateTime.UtcNow.Should().BeBefore(deadline, $"the model should reach {state}");
            await Task.Delay(10, Ct);
        }
    }

    private sealed class FakeArtifacts(string kind) : IModelArtifacts
    {
        private int _downloads;
        private int _deletes;

        public HashSet<string> Cached { get; } = [];
        public TaskCompletionSource? Gate { get; init; }
        public TaskCompletionSource Started { get; } = new(TaskCreationOptions.RunContinuationsAsynchronously);
        public Exception? Failure { get; set; }
        public int Downloads => Volatile.Read(ref _downloads);
        public int Deletes => Volatile.Read(ref _deletes);
        public ManualResetEventSlim? DeleteGate { get; init; }
        public ManualResetEventSlim DeleteStarted { get; } = new();

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
            {
                try
                {
                    await gate.Task.WaitAsync(cancellationToken);
                }
                catch (OperationCanceledException) when (cancellationToken.IsCancellationRequested)
                {
                    Cancelled.TrySetResult();
                    if (WindDown is { } windDown)
                        await windDown.Task;
                    throw;
                }
            }
        }

        /// <summary>Set when a download observed its cancellation.</summary>
        public TaskCompletionSource Cancelled { get; } = new(TaskCreationOptions.RunContinuationsAsynchronously);

        /// <summary>When set, a cancelled download keeps running until this completes (a provider still closing its files).</summary>
        public TaskCompletionSource? WindDown { get; init; }

        public bool Delete(string modelId)
        {
            Interlocked.Increment(ref _deletes);
            DeleteStarted.Set();
            DeleteGate?.Wait(TestContext.Current.CancellationToken);
            lock (Cached) return Cached.Remove(modelId);
        }
    }
}
