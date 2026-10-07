using AwesomeAssertions;

namespace LMSupply.Generator.Tests;

/// <summary>
/// A load from a local path still provisions the runtime on first use (llama-server for GGUF), so it must take the
/// caller's progress and cancellation the way a load by id does. Until 0.112.0 the path loads passed neither on:
/// the runtime step was invisible and could not be cancelled.
/// </summary>
public class LocalGeneratorLoadFromPathTests
{
    private sealed class Recorder : IProgress<DownloadProgress>
    {
        public List<DownloadProgress> Reports { get; } = [];

        public void Report(DownloadProgress value) => Reports.Add(value);
    }

    public static TheoryData<string> Entries => ["LoadFromPathAsync", "LoadAsync"];

    [Theory]
    [MemberData(nameof(Entries))]
    public async Task APathLoad_ReportsTheRuntimeStep_AndHonoursCancellation(string entry)
    {
        var path = Path.Combine(Path.GetTempPath(), "lmsupply-path-load-" + Guid.NewGuid().ToString("N") + ".gguf");
        await File.WriteAllBytesAsync(path, "not a model"u8.ToArray(), TestContext.Current.CancellationToken);
        try
        {
            var recorder = new Recorder();
            using var cancelled = new CancellationTokenSource();
            await cancelled.CancelAsync();

            Func<Task> load = entry == "LoadFromPathAsync"
                ? () => LocalGenerator.LoadFromPathAsync(path, progress: recorder, cancellationToken: cancelled.Token)
                : () => LocalGenerator.LoadAsync(path, progress: recorder, cancellationToken: cancelled.Token);

            await load.Should().ThrowAsync<OperationCanceledException>(
                "a cancelled token must stop the runtime step, not be replaced by CancellationToken.None");

            var first = recorder.Reports.Should().NotBeEmpty("the load's progress must reach the runtime step").And.Subject.First();
            first.Kind.Should().Be(DownloadKind.Runtime);
            first.FileName.Should().Be("llama-server");
            first.Phase.Should().Be(DownloadPhase.Preparing);
        }
        finally
        {
            File.Delete(path);
        }
    }
}
