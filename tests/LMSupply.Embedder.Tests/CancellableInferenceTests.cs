using System.Diagnostics;
using AwesomeAssertions;
using LMSupply.Embedder.Inference;
using LMSupply.Exceptions;

namespace LMSupply.Embedder.Tests;

/// <summary>
/// Tests for <see cref="CancellableInference"/> control-return guarantee.
/// ISSUE: LMSupply-20260601-094128-embedder-ct-native-hang
/// AC#1/#3: a cancelled token must return control to the caller within bound even when the
/// wrapped synchronous work ignores cancellation and remains blocked (simulated native hang).
///
/// ISSUE: claudedocs/lm-supply/issues/ISSUE-lm-supply-20260903-122754-directml-embed-hang-no-default-timeout.md
/// AC#3: a caller that supplies no cancellable token (CancellationToken.None) must still fail
/// within a bounded default timeout rather than hang forever.
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

    [Fact]
    public async Task RunAsync_NoTokenSupplied_SyntheticHang_ThrowsInferenceTimeoutWithinDefaultBound()
    {
        // CancellationToken.None cannot itself time out -- before the fix, this hung forever.
        // Overriding `timeout` here (instead of waiting the real DefaultTimeout) is what keeps
        // this test fast; the mechanism under test (a caller-supplied timeout bound applying even
        // when cancellationToken is CancellationToken.None) is identical.
        using var released = new ManualResetEventSlim(false);
        Func<int> hang = () =>
        {
            released.Wait(); // ignores cancellation entirely, mirrors a cold DirectML kernel hang
            return 0;
        };

        var sw = Stopwatch.StartNew();
        try
        {
            var act = async () => await CancellableInference.RunAsync(
                hang, CancellationToken.None, TimeSpan.FromMilliseconds(200));

            await act.Should().ThrowAsync<InferenceTimeoutException>()
                .WithMessage("*DirectML*");
            sw.Stop();
            sw.Elapsed.Should().BeLessThan(TimeSpan.FromSeconds(5),
                "control must return to the caller once the default timeout elapses, " +
                "even though CancellationToken.None can never itself signal cancellation");
        }
        finally
        {
            released.Set();
        }
    }

    [Fact]
    public async Task RunAsync_ExplicitCancellation_StillThrowsOperationCanceled_NotTimeout()
    {
        // A real caller-initiated cancellation must remain OperationCanceledException, not be
        // reclassified as InferenceTimeoutException, even though both now share the same
        // internal linked-token mechanism.
        using var cts = new CancellationTokenSource();
        cts.Cancel();

        var act = async () => await CancellableInference.RunAsync(() => 1, cts.Token);

        await act.Should().ThrowAsync<OperationCanceledException>();
    }

    [Fact]
    public void DefaultTimeout_IsGenerousButBounded()
    {
        // Sanity check on the constant itself: long enough not to false-positive on legitimate
        // slow-but-progressing CPU inference, short enough that a caller is never left waiting
        // silently forever.
        CancellableInference.DefaultTimeout.Should().BeGreaterThan(TimeSpan.FromSeconds(10));
        CancellableInference.DefaultTimeout.Should().BeLessThan(TimeSpan.FromMinutes(5));
    }
}
