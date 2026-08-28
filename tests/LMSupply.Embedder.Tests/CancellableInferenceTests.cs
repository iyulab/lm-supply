using System.Diagnostics;
using AwesomeAssertions;
using LMSupply.Embedder.Inference;

namespace LMSupply.Embedder.Tests;

/// <summary>
/// Tests for <see cref="CancellableInference"/> control-return guarantee.
/// ISSUE: LMSupply-20260601-094128-embedder-ct-native-hang
/// AC#1/#3: a cancelled token must return control to the caller within bound even when the
/// wrapped synchronous work ignores cancellation and remains blocked (simulated native hang).
/// </summary>
public class CancellableInferenceTests
{
    [Fact]
    public async Task RunAsync_CompletedWork_ReturnsResult()
    {
        // Normal (non-hang) path: result is returned unchanged. (AC#2 — no regression.)
        var result = await CancellableInference.RunAsync(() => 42, CancellationToken.None);
        result.Should().Be(42);
    }

    [Fact]
    public async Task RunAsync_CancelledBeforeStart_ThrowsOperationCanceled()
    {
        using var cts = new CancellationTokenSource();
        cts.Cancel();

        var act = async () => await CancellableInference.RunAsync(() => 1, cts.Token);

        await act.Should().ThrowAsync<OperationCanceledException>();
    }

    [Fact]
    public async Task RunAsync_SyntheticHang_ReturnsControlWithinBound()
    {
        // Simulate a native call that ignores cancellation and blocks indefinitely.
        // The hang is released only when the test completes, so the work delegate itself never
        // observes the token — mirroring a cold ONNX/DirectML init hang.
        using var released = new ManualResetEventSlim(false);
        using var cts = new CancellationTokenSource(TimeSpan.FromMilliseconds(200));

        Func<int> hang = () =>
        {
            released.Wait(); // blocks until the test signals release; ignores cts entirely
            return 0;
        };

        var sw = Stopwatch.StartNew();
        try
        {
            var act = async () => await CancellableInference.RunAsync(hang, cts.Token);

            // Control must return to the caller shortly after the 200ms deadline, NOT block forever.
            await act.Should().ThrowAsync<OperationCanceledException>();
            sw.Stop();
            sw.Elapsed.Should().BeLessThan(TimeSpan.FromSeconds(5),
                "control must return to the caller on cancellation even though the inner work is still blocked");
        }
        finally
        {
            // Release the leaked thread so it does not linger past the test.
            released.Set();
        }
    }
}
