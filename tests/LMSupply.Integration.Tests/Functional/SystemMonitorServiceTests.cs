using LMSupply.Console.Host.Services;
using Microsoft.Extensions.Logging.Abstractions;

namespace LMSupply.Integration.Tests.Functional;

/// <summary>
/// The console host's process figures are read live: a figure taken from a Process object stays at its first read
/// until that object is refreshed, so the status and the metrics stream reported the memory the host had when the
/// service started.
/// </summary>
[Trait("Category", "Functional")]
[Trait("Domain", "ConsoleHost")]
public class SystemMonitorServiceTests
{
    private const int HeldMegabytes = 256;

    [Fact]
    public async Task ProcessMemory_InTheStatusAndTheStream_FollowsWhatTheProcessHolds()
    {
        var monitor = new SystemMonitorService(NullLogger<SystemMonitorService>.Instance);
        var statusBefore = monitor.GetStatus().ProcessMemoryMB;
        var streamBefore = await FirstStreamedRamAsync(monitor);

        // Touch every page so the working set grows, and keep the block reachable until both reads are done.
        var held = new byte[HeldMegabytes * 1024 * 1024];
        for (var i = 0; i < held.Length; i += 4096)
            held[i] = 1;

        var statusAfter = monitor.GetStatus().ProcessMemoryMB;
        var streamAfter = await FirstStreamedRamAsync(monitor);
        GC.KeepAlive(held);

        statusAfter.Should().BeGreaterThan(statusBefore + HeldMegabytes / 2,
            "the status reports the process's working set as it is now");
        streamAfter.Should().BeGreaterThan(streamBefore + HeldMegabytes / 2,
            "each metrics tick reports the working set as it is then");
    }

    private static async Task<double> FirstStreamedRamAsync(SystemMonitorService monitor)
    {
        await foreach (var metrics in monitor.StreamMetricsAsync(TestContext.Current.CancellationToken))
            return metrics.RamUsageMB;
        throw new InvalidOperationException("The metrics stream ended before its first tick.");
    }
}
