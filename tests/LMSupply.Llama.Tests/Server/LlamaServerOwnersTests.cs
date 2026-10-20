using System.Diagnostics;
using System.Globalization;
using AwesomeAssertions;
using LMSupply.Llama.Server;

namespace LMSupply.Llama.Tests.Server;

public sealed class LlamaServerOwnersTests : IDisposable
{
    private readonly string _directory = Path.Combine(Path.GetTempPath(), "lmsupply-owners-" + Guid.NewGuid().ToString("N"));

    public void Dispose()
    {
        if (Directory.Exists(_directory))
            Directory.Delete(_directory, recursive: true);
    }

    // A stand-in for a server: a process that runs until it is stopped.
    private static Process LongRunning()
    {
        var startInfo = OperatingSystem.IsWindows()
            ? new ProcessStartInfo("cmd.exe", "/c ping -n 120 127.0.0.1 > nul")
            : new ProcessStartInfo("/bin/sleep", "120");
        startInfo.UseShellExecute = false;
        startInfo.CreateNoWindow = true;
        return Process.Start(startInfo)!;
    }

    // A process that has already ended: its id and start time name an owner that is gone. It is ended only after its
    // start time is read — a process that exits on its own may be gone before then.
    private static async Task<(int Id, long StartTicks)> EndedAsync()
    {
        using var process = LongRunning();
        var start = process.StartTime.ToUniversalTime().Ticks;
        process.Kill(entireProcessTree: true);
        await process.WaitForExitAsync(TestContext.Current.CancellationToken);
        return (process.Id, start);
    }

    private string Record(Process server, int ownerId, long ownerStartTicks, long? serverStartTicks = null, string? name = null)
    {
        Directory.CreateDirectory(_directory);
        var path = Path.Combine(_directory, server.Id.ToString(CultureInfo.InvariantCulture) + ".owner");
        File.WriteAllText(path, string.Join('\n',
            server.Id, serverStartTicks ?? server.StartTime.ToUniversalTime().Ticks, name ?? server.ProcessName, ownerId, ownerStartTicks));
        return path;
    }

    [Fact]
    public async Task AServerWhoseOwnerIsGone_IsStopped_AndItsRecordRemoved()
    {
        using var server = LongRunning();
        var (owner, ownerStart) = await EndedAsync();
        var record = Record(server, owner, ownerStart);

        LlamaServerOwners.StopOrphans(_directory).Should().Be(1);

        server.WaitForExit(10_000).Should().BeTrue("the orphan is stopped");
        File.Exists(record).Should().BeFalse();
    }

    [Fact]
    public void AServerWhoseOwnerRuns_IsLeftAlone()
    {
        using var server = LongRunning();
        try
        {
            var record = LlamaServerOwners.Take(server, _directory);
            record.Should().NotBeNull();

            LlamaServerOwners.StopOrphans(_directory).Should().Be(0);

            server.HasExited.Should().BeFalse("its owner, this process, is running");
            File.Exists(record).Should().BeTrue();

            LlamaServerOwners.Release(record);
            File.Exists(record).Should().BeFalse();
        }
        finally
        {
            server.Kill(entireProcessTree: true);
        }
    }

    [Theory]
    [InlineData("start time")]
    [InlineData("name")]
    public async Task AProcessThatOnlySharesTheServersId_IsNeverStopped(string differs)
    {
        // The id was reused: the process running under it started at another time, or is another program.
        using var other = LongRunning();
        try
        {
            var (owner, ownerStart) = await EndedAsync();
            var record = differs == "name"
                ? Record(other, owner, ownerStart, name: "llama-server")
                : Record(other, owner, ownerStart, serverStartTicks: other.StartTime.ToUniversalTime().AddMinutes(-5).Ticks);

            LlamaServerOwners.StopOrphans(_directory).Should().Be(0);

            other.HasExited.Should().BeFalse();
            File.Exists(record).Should().BeFalse("the record names nothing that runs, so it is removed");
        }
        finally
        {
            other.Kill(entireProcessTree: true);
        }
    }

    [Fact]
    public void ARecordThisCodeDidNotWrite_IsRemoved_AndStopsNothing()
    {
        Directory.CreateDirectory(_directory);
        var junk = Path.Combine(_directory, "1.owner");
        File.WriteAllText(junk, "1\nnot a number");

        LlamaServerOwners.StopOrphans(_directory).Should().Be(0);
        File.Exists(junk).Should().BeFalse();
    }

    [Fact]
    public void NoDirectory_StopsNothing() =>
        LlamaServerOwners.StopOrphans(_directory).Should().Be(0);
}
