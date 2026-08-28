using System.Diagnostics;
using AwesomeAssertions;
using LMSupply.Generator.Internal.Llama;

namespace LMSupply.Generator.Tests;

/// <summary>
/// Verifies that LocalGenerator.LoadAsync("default") delegates to the
/// auto hardware-aware selection path. We detect the delegation by
/// listening for the "[LocalGenerator.auto]" Trace line emitted by
/// LoadAutoAsync, without actually downloading any model weights.
/// </summary>
[Collection("TraceListeners")]
public class LocalGeneratorDefaultRoutingTests
{
    private sealed class CapturingListener : TraceListener
    {
        private readonly List<string> _lines = new();
        public IReadOnlyList<string> Lines => _lines;

        public override void Write(string? message) { if (message != null) _lines.Add(message); }
        public override void WriteLine(string? message) { if (message != null) _lines.Add(message); }
    }

    private static async Task<IReadOnlyList<string>> CaptureTraceForLoadAsync(string modelId)
    {
        var listener = new CapturingListener();
        Trace.Listeners.Add(listener);
        try
        {
            try
            {
                // Pre-canceled token: LogAutoSelection is synchronous and fires
                // before any cancellation check in the downstream loader, so the
                // Trace line is always captured deterministically. The downstream
                // load either observes cancellation or hits another error path.
                using var cts = new CancellationTokenSource();
                cts.Cancel();
                await LocalGenerator.LoadAsync(modelId, cancellationToken: cts.Token);
            }
            catch
            {
                // Expected: cancellation, network failure, or path resolution error.
            }
        }
        finally
        {
            Trace.Listeners.Remove(listener);
        }
        return listener.Lines;
    }

    [Fact]
    public async Task LoadAsync_Default_EmitsAutoTraceLine()
    {
        var lines = await CaptureTraceForLoadAsync("default");
        lines.Should().Contain(l => l.Contains("[LocalGenerator.auto]"));
    }

    [Fact]
    public async Task LoadAsync_Auto_EmitsAutoTraceLine()
    {
        var lines = await CaptureTraceForLoadAsync("auto");
        lines.Should().Contain(l => l.Contains("[LocalGenerator.auto]"));
    }

    [Fact]
    public async Task LoadAsync_ExplicitRepoId_DoesNotEmitAutoTraceLine()
    {
        var lines = await CaptureTraceForLoadAsync("microsoft/Phi-4-mini-instruct-onnx");
        lines.Should().NotContain(l => l.Contains("[LocalGenerator.auto]"));
    }

    [Theory]
    [InlineData("gguf:gemma4-fast")]
    [InlineData("gguf:gemma4-default")]
    [InlineData("gguf:gemma4-balanced")]
    [InlineData("gguf:phi-4-mini")]   // registered alias — must not split on ':'
    [InlineData("gguf:qwen2.5-7b")]   // registered alias with hyphenated suffix
    public async Task LoadAsync_GgufPrefixedAlias_IsNotShreddedByQualifierSplit(string alias)
    {
        // Regression: SplitQualifier("gguf:fast") historically split into
        // ("gguf", "fast") and tried to download a repo literally named "gguf",
        // 401ing with "Access denied to repository 'gguf'". LocalGenerator must
        // recognize GGUF registry aliases before applying qualifier splitting.
        // We verify the negative signal: the error message must NOT say the
        // request targeted a repo called "gguf".
        var listener = new CapturingListener();
        Trace.Listeners.Add(listener);
        string? errorMessage = null;
        try
        {
            using var cts = new CancellationTokenSource();
            cts.Cancel();
            await LocalGenerator.LoadAsync(alias, cancellationToken: cts.Token);
        }
        catch (Exception ex)
        {
            errorMessage = ex.Message;
        }
        finally
        {
            Trace.Listeners.Remove(listener);
        }

        if (errorMessage is not null)
        {
            errorMessage.Should().NotContain("'gguf'",
                because: "the alias must not be split into modelId='gguf' + qualifier");
        }
    }

    [Fact]
    public async Task LoadAsync_UnregisteredGgufPrefixedAlias_ThrowsDescriptiveArgumentException()
    {
        // "gguf:unknown-model" is gguf:-prefixed but not in the registry.
        // Before the fix: SplitQualifier split it to ("gguf", "unknown-model"),
        // then DownloadAsync("gguf") called HF API with repoId="gguf" and got 401.
        // After the fix: ArgumentException with a helpful message listing known aliases.
        var ex = await Assert.ThrowsAsync<ArgumentException>(
            () => LocalGenerator.LoadAsync("gguf:unknown-model"));

        ex.Message.Should().Contain("gguf:unknown-model");
        ex.Message.Should().Contain("not a registered GGUF alias");
        ex.Message.Should().Contain("gguf:gemma4-fast"); // should list known aliases
    }

    [Fact]
    public void Phi4MiniGgufAlias_IsRegistered()
    {
        var info = GgufModelRegistry.Resolve("gguf:phi-4-mini");
        info.Should().NotBeNull();
        info!.RepoId.Should().Be("bartowski/microsoft_Phi-4-mini-instruct-GGUF");
        info.ChatFormat.Should().Be("phi3");
        info.NumLayers.Should().Be(32);
    }

    [Fact]
    public void FastAlias_StillResolvesToPhi4Mini()
    {
        var registry = GeneratorModelRegistry.Default;
        var resolved = registry.TryResolve("fast", out var info);

        resolved.Should().BeTrue();
        info.Should().NotBeNull();
        info!.ModelId.Should().Be("microsoft/Phi-4-mini-instruct-onnx");
    }

    [Fact]
    public void PhiMiniAlias_ResolvesToPhi4Mini()
    {
        var registry = GeneratorModelRegistry.Default;
        var resolved = registry.TryResolve("phi-4-mini", out var info);

        resolved.Should().BeTrue();
        info.Should().NotBeNull();
        info!.ModelId.Should().Be("microsoft/Phi-4-mini-instruct-onnx");
    }

    [Fact]
    public async Task LoadWithFallbackChainAsync_AllFail_ThrowsAggregateException()
    {
        // All candidates are unregistered gguf:* aliases → ArgumentException from LoadGgufAsync
        var ex = await Assert.ThrowsAsync<AggregateException>(
            () => LocalGenerator.LoadWithFallbackChainAsync(
                ["gguf:no-such-model-a", "gguf:no-such-model-b"]));

        ex.InnerExceptions.Should().HaveCount(2);
        ex.Message.Should().Contain("2 candidate model(s) failed");
    }

    [Fact]
    public async Task LoadWithFallbackChainAsync_OnFailureCallback_InvokedForEachFailure()
    {
        var failedIds = new List<string>();

        try
        {
            await LocalGenerator.LoadWithFallbackChainAsync(
                ["gguf:no-such-model-x", "gguf:no-such-model-y"],
                onFailure: (id, _) => failedIds.Add(id));
        }
        catch (AggregateException) { /* expected */ }

        failedIds.Should().Equal("gguf:no-such-model-x", "gguf:no-such-model-y");
    }

    [Fact]
    public async Task LoadWithFallbackChainAsync_EmptyCandidates_ThrowsArgumentException()
    {
        await Assert.ThrowsAsync<ArgumentException>(
            () => LocalGenerator.LoadWithFallbackChainAsync([]));
    }

    [Fact]
    public void GeneratorOptions_PreferredAutoModelId_DefaultsToNull()
    {
        new GeneratorOptions().PreferredAutoModelId.Should().BeNull();
    }

    [Fact]
    public async Task LoadAsync_AutoWithFailingPreferredModelId_FallsBackToAutoSelection()
    {
        // "gguf:no-such-preferred-model" is unregistered → ArgumentException from LoadGgufAsync.
        // LoadAutoAsync catches it (not OperationCanceledException) and falls through to standard
        // hardware-aware selection, which emits the [LocalGenerator.auto] trace line.
        //
        // Invariant: LoadGgufAsync throws ArgumentException synchronously before any cancellation
        // check (the guard at GeneratorModelLoader.cs:125-132 runs before any async I/O).
        // If a future refactor adds early ThrowIfCancellationRequested(), this test needs adjustment.
        var options = new GeneratorOptions
        {
            PreferredAutoModelId = "gguf:no-such-preferred-model"
        };

        var listener = new CapturingListener();
        Trace.Listeners.Add(listener);
        try
        {
            using var cts = new CancellationTokenSource();
            cts.Cancel();
            try { await LocalGenerator.LoadAsync("auto", options, cancellationToken: cts.Token); }
            catch { }
        }
        finally
        {
            Trace.Listeners.Remove(listener);
        }

        listener.Lines.Should().Contain(l => l.Contains("[LocalGenerator.auto]"),
            because: "fallback to hardware selection must occur when preferred model is unavailable");
    }
}
