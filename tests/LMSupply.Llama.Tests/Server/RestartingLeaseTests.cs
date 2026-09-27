using AwesomeAssertions;
using LMSupply.Llama.Server;

namespace LMSupply.Llama.Tests.Server;

/// <summary>
/// The replacement rule every GGUF model (generator, embedder, reranker) uses for a server that exited: keep a live
/// lease, replace a dead one once — even under concurrent callers — and release the dead one.
/// </summary>
public class RestartingLeaseTests
{
    private sealed class FakeLease(int id) : IAsyncDisposable
    {
        public int Id { get; } = id;
        public bool Alive { get; set; } = true;
        public bool Disposed { get; private set; }

        public ValueTask DisposeAsync()
        {
            Disposed = true;
            return ValueTask.CompletedTask;
        }
    }

    private static (RestartingLease<FakeLease> Holder, Func<int> ReletCount) Make(FakeLease initial)
    {
        var next = initial.Id;
        var relets = 0;
        var holder = new RestartingLease<FakeLease>(
            initial,
            _ =>
            {
                Interlocked.Increment(ref relets);
                return Task.FromResult(new FakeLease(Interlocked.Increment(ref next)));
            },
            l => l.Alive,
            _ => "exit code 137",
            "test");
        return (holder, () => Volatile.Read(ref relets));
    }

    [Fact]
    public async Task A_live_lease_is_kept()
    {
        var first = new FakeLease(1);
        var (holder, relets) = Make(first);

        (await holder.GetAsync(TestContext.Current.CancellationToken)).Should().BeSameAs(first);
        relets().Should().Be(0);
    }

    [Fact]
    public async Task A_dead_lease_is_replaced_and_released()
    {
        var first = new FakeLease(1);
        var (holder, relets) = Make(first);
        first.Alive = false;

        var second = await holder.GetAsync(TestContext.Current.CancellationToken);

        second.Id.Should().Be(2);
        first.Disposed.Should().BeTrue("the dead lease goes back to the pool");
        holder.Current.Should().BeSameAs(second);
        relets().Should().Be(1);
    }

    [Fact]
    public async Task Concurrent_callers_replace_a_dead_lease_once()
    {
        var first = new FakeLease(1);
        var (holder, relets) = Make(first);
        first.Alive = false;

        var leases = await Task.WhenAll(Enumerable.Range(0, 16)
            .Select(_ => holder.GetAsync(TestContext.Current.CancellationToken).AsTask()));

        relets().Should().Be(1);
        leases.Select(l => l.Id).Distinct().Should().Equal(2);
    }

    [Fact]
    public async Task Dispose_releases_the_current_lease()
    {
        var first = new FakeLease(1);
        var (holder, _) = Make(first);

        await holder.DisposeAsync();

        first.Disposed.Should().BeTrue();
        var act = () => holder.GetAsync(TestContext.Current.CancellationToken).AsTask();
        await act.Should().ThrowAsync<ObjectDisposedException>();
    }
}
