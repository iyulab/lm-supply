using AwesomeAssertions;
using LMSupply.Llama.Server;
using Xunit;

namespace LMSupply.Llama.Tests.Server;

/// <summary>
/// Tests for <see cref="LlamaServerDownloader.RunGatedOnceAsync"/>, the run-at-most-once primitive
/// behind CUDA-runtime provisioning. The main consumer (Filer) loads embedder + generator into the
/// same cuda12 versionDir via <c>Task.WhenAll</c>; without serialization both threads see the runtime
/// missing and race the same multi-hundred-MB download/extract (corruption + redundant downloads).
/// A single gate + a post-acquire re-check converges concurrent callers to exactly one execution.
///
/// Network-free / HW-free: the gated work is an arbitrary delegate, and every case passes its own
/// <see cref="SemaphoreSlim"/> so cases never share static state (which would reproduce the parallel
/// static-state flake class seen elsewhere).
/// </summary>
public sealed class LlamaServerGatedOnceTests
{
    [Fact]
    public async Task ConcurrentCallers_RunWorkExactlyOnce()
    {
        var gate = new SemaphoreSlim(1, 1);
        var done = false;
        var calls = 0;

        async Task Work()
        {
            Interlocked.Increment(ref calls);
            await Task.Delay(150); // widen the window so a missing re-check would race
            Volatile.Write(ref done, true);
        }

        var tasks = Enumerable.Range(0, 5).Select(_ =>
            LlamaServerDownloader.RunGatedOnceAsync(
                gate, () => Volatile.Read(ref done), Work, CancellationToken.None));
        await Task.WhenAll(tasks);

        calls.Should().Be(1,
            "the gate plus the post-acquire re-check must converge concurrent callers to a single run");
        gate.CurrentCount.Should().Be(1, "the gate must be released back to its initial count");
    }

    [Fact]
    public async Task AlreadyDone_SkipsWork_WithoutEnteringGate()
    {
        var gate = new SemaphoreSlim(1, 1);
        var calls = 0;

        await LlamaServerDownloader.RunGatedOnceAsync(
            gate, alreadyDone: () => true,
            work: () => { Interlocked.Increment(ref calls); return Task.CompletedTask; },
            CancellationToken.None);

        calls.Should().Be(0, "the lock-free fast path must skip work when it is already done");
        gate.CurrentCount.Should().Be(1, "the fast path must not enter the gate");
    }

    [Fact]
    public async Task PreCancelledToken_PropagatesAndLeavesGateUsable()
    {
        var gate = new SemaphoreSlim(1, 1);
        var calls = 0;
        Task Work() { Interlocked.Increment(ref calls); return Task.CompletedTask; }

        using var cts = new CancellationTokenSource();
        cts.Cancel();

        var act = () => LlamaServerDownloader.RunGatedOnceAsync(
            gate, alreadyDone: () => false, Work, cts.Token);
        await act.Should().ThrowAsync<OperationCanceledException>(
            "cancellation propagates; the best-effort swallow belongs to the caller, not the primitive");

        // The acquired guard must prevent releasing a gate that WaitAsync never acquired.
        gate.CurrentCount.Should().Be(1, "a cancelled wait must not corrupt the gate count");
        calls.Should().Be(0);

        // A subsequent normal call still acquires the gate and runs exactly once.
        await LlamaServerDownloader.RunGatedOnceAsync(
            gate, alreadyDone: () => false, Work, CancellationToken.None);
        calls.Should().Be(1);
    }
}
