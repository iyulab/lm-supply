using AwesomeAssertions;
using LMSupply.Embedder;
using LMSupply.Lifecycle;
using Xunit;

namespace LMSupply.Integration.Tests;

/// <summary>
/// <see cref="ModelLifecycle"/> over a real download (the <c>fast</c> embedder from Hugging Face into an empty cache): a
/// watcher sees progress, a cancel leaves the model <see cref="ModelState.Missing"/> with its partial file kept for a resume,
/// the next download ends <see cref="ModelState.Ready"/>, and a delete by the alias takes the model back to Missing.
/// Run locally only — requires network access.
/// </summary>
[Trait("Category", "Integration")]
[Trait("Category", "LocalOnly")]
public sealed class ModelLifecycleDownloadFacts : IDisposable
{
    private const string Alias = "fast";
    private readonly string _cacheDir = Path.Combine(Path.GetTempPath(), "lmsupply-lifecycle-" + Guid.NewGuid().ToString("N"));

    public void Dispose()
    {
        try { Directory.Delete(_cacheDir, recursive: true); } catch { /* best-effort temp cleanup */ }
    }

    private static CancellationToken Ct => TestContext.Current.CancellationToken;

    private string[] PartFiles() => Directory.Exists(_cacheDir)
        ? Directory.GetFiles(_cacheDir, "*.part", SearchOption.AllDirectories)
        : [];

    [Fact]
    public async Task Download_Cancel_Resume_Delete()
    {
        var lifecycle = new ModelLifecycle([new EmbedderArtifacts(new EmbedderOptions { CacheDirectory = _cacheDir })]);
        (await lifecycle.GetStatusAsync("embedder", Alias, Ct)).State.Should().Be(ModelState.Missing);

        // Cancel once a few MB have arrived.
        using var watchCts = CancellationTokenSource.CreateLinkedTokenSource(Ct);
        var started = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        var watching = Task.Run(async () =>
        {
            await foreach (var status in lifecycle.WatchAsync(watchCts.Token))
            {
                if (status.Progress is { BytesDownloaded: > 4_000_000 })
                    started.TrySetResult();
            }
        }, Ct);
        await Task.Delay(50, Ct); // the watcher is subscribed

        var first = lifecycle.DownloadAsync("embedder", Alias, Ct);
        await started.Task.WaitAsync(TimeSpan.FromMinutes(5), Ct);
        lifecycle.CancelDownload("embedder", Alias).Should().BeTrue();
        await first.Awaiting(t => t).Should().ThrowAsync<OperationCanceledException>();

        (await lifecycle.GetStatusAsync("embedder", Alias, Ct)).State.Should().Be(ModelState.Missing);
        PartFiles().Should().Contain(p => new FileInfo(p).Length > 0, "a cancelled download keeps its partial file to resume from");

        // The next download resumes and ends Ready, with no partial file left.
        (await lifecycle.DownloadAsync("embedder", Alias, Ct).WaitAsync(TimeSpan.FromMinutes(20), Ct)).State.Should().Be(ModelState.Ready);
        (await lifecycle.GetStatusAsync("embedder", Alias, Ct)).State.Should().Be(ModelState.Ready);
        PartFiles().Should().BeEmpty();

        (await lifecycle.DeleteAsync("embedder", Alias, Ct)).Should().BeTrue();
        (await lifecycle.GetStatusAsync("embedder", Alias, Ct)).State.Should().Be(ModelState.Missing);

        await watchCts.CancelAsync();
        await watching.Awaiting(t => t).Should().ThrowAsync<OperationCanceledException>();
    }
}
