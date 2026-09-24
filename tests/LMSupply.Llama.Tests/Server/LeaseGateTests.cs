using AwesomeAssertions;
using LMSupply.Llama.Server;

namespace LMSupply.Llama.Tests.Server;

/// <summary>
/// <see cref="LeaseGate"/> is what lets the pool stop an idle llama-server without racing a load that
/// is about to lease it: a retire succeeds only while no lease is held, and no lease is granted after it.
/// </summary>
public class LeaseGateTests
{
    [Fact]
    public void Retire_WhileNoLeaseIsHeld_Succeeds_AndNoLeaseIsGrantedAfterwards()
    {
        var gate = new LeaseGate();

        gate.TryRetire().Should().BeTrue();
        gate.IsRetired.Should().BeTrue();
        gate.TryAcquire().Should().BeFalse("a retired server is being stopped");
        gate.TryRetire().Should().BeFalse("it is already retired");
    }

    [Fact]
    public void Retire_WhileALeaseIsHeld_Fails_UntilTheLeaseIsReleased()
    {
        var gate = new LeaseGate();
        gate.TryAcquire().Should().BeTrue();
        gate.TryAcquire().Should().BeTrue();

        gate.TryRetire().Should().BeFalse("a model is using the server");
        gate.Release();
        gate.TryRetire().Should().BeFalse("one lease is still held");
        gate.Release();

        gate.IsHeld.Should().BeFalse();
        gate.TryRetire().Should().BeTrue();
    }

    [Fact]
    public void Release_WithoutALease_Throws()
    {
        var gate = new LeaseGate();

        var act = gate.Release;

        act.Should().Throw<InvalidOperationException>();
    }

    [Fact]
    public async Task ConcurrentAcquiresAndARetire_NeverBothWin()
    {
        // Many loads race one idle sweep. Either the sweep retires first and every acquire fails,
        // or some acquire wins and the retire fails — never a granted lease on a retired server.
        for (var round = 0; round < 200; round++)
        {
            var gate = new LeaseGate();
            var acquired = 0;
            var retired = false;
            using var start = new Barrier(9);

            var ct = TestContext.Current.CancellationToken;
            var tasks = Enumerable.Range(0, 8).Select(_ => Task.Run(() =>
            {
                start.SignalAndWait(ct);
                if (gate.TryAcquire())
                    Interlocked.Increment(ref acquired);
            }, ct)).ToList();
            tasks.Add(Task.Run(() =>
            {
                start.SignalAndWait(ct);
                retired = gate.TryRetire();
            }, ct));
            await Task.WhenAll(tasks);

            if (retired)
                acquired.Should().Be(0, "no lease may be granted on a server the sweep retired");
            else
                acquired.Should().BeGreaterThan(0, "the retire fails only because a lease was held");
        }
    }
}
