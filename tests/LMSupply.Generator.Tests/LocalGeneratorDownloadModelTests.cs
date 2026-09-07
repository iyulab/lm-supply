using System.Diagnostics;
using AwesomeAssertions;
using LMSupply.Generator.Internal.Llama;

namespace LMSupply.Generator.Tests;

/// <summary>
/// <see cref="LocalGenerator.DownloadModelAsync"/> must resolve a model id exactly the way
/// <see cref="LocalGenerator.LoadAsync"/> does — a warmed cache that holds a different file than the
/// load opens is worse than no warming. These pin the resolution steps that need no network; the
/// one that touches the cache is tagged Integration.
/// </summary>
[Collection("TraceListeners")]
public class LocalGeneratorDownloadModelTests
{
    [Fact]
    public async Task DownloadModelAsync_LocalFile_ReturnsThePathWithoutTouchingAnything()
    {
        var path = Path.Combine(Path.GetTempPath(), "lmsupply-dl-" + Guid.NewGuid().ToString("N") + ".gguf");
        await File.WriteAllBytesAsync(path, "not a model"u8.ToArray(), TestContext.Current.CancellationToken);
        try
        {
            var result = await LocalGenerator.DownloadModelAsync(path, cancellationToken: TestContext.Current.CancellationToken);

            result.Should().Be(path, "a local path is already downloaded");
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public async Task DownloadModelAsync_UnregisteredGgufAlias_FailsLikeLoadAsync_ListingAliases()
    {
        var act = () => LocalGenerator.DownloadModelAsync("gguf:not-a-registered-alias", cancellationToken: TestContext.Current.CancellationToken);

        var ex = await act.Should().ThrowAsync<ArgumentException>(
            "a gguf: id that is not an alias must not be sent to HuggingFace as a repo id");
        ex.Which.Message.Should().Contain("Known aliases");
    }

    [Fact]
    public async Task DownloadModelAsync_Default_UsesTheSameAutoSelectionAsLoadAsync()
    {
        // Pre-cancelled token: the [LocalGenerator.auto] selection log fires synchronously before
        // any download can start, so the trace proves the routing without pulling weights.
        var listener = new CapturingListener();
        Trace.Listeners.Add(listener);
        try
        {
            using var cts = new CancellationTokenSource();
            cts.Cancel();
            try { await LocalGenerator.DownloadModelAsync("default", cancellationToken: cts.Token); }
            catch { /* cancellation or a resolution error after the selection log — either is fine here */ }
        }
        finally
        {
            Trace.Listeners.Remove(listener);
        }

        listener.Lines.Should().Contain(l => l.Contains("[LocalGenerator.auto]"),
            "default/auto must go through the hardware-aware selection LoadAsync uses");
    }

    [Fact]
    [Trait("Category", "Integration")]
    public async Task DownloadModelAsync_RegisteredGgufAlias_ReturnsTheFileLoadAsyncWouldOpen()
    {
        const string alias = "gguf:gemma4-default";
        var options = new GeneratorOptions();

        var path = await LocalGenerator.DownloadModelAsync(alias, options, cancellationToken: TestContext.Current.CancellationToken);

        File.Exists(path).Should().BeTrue();
        Path.GetExtension(path).Should().Be(".gguf");
        var info = GgufModelRegistry.Resolve(alias)!;
        path.Should().Contain("models--" + info.RepoId.Replace("/", "--"),
            "the file lives under the HuggingFace cache directory for the registry repo");
    }

    private sealed class CapturingListener : TraceListener
    {
        private readonly List<string> _lines = [];
        public IReadOnlyList<string> Lines => _lines;
        public override void Write(string? message) { if (message != null) _lines.Add(message); }
        public override void WriteLine(string? message) { if (message != null) _lines.Add(message); }
    }
}
