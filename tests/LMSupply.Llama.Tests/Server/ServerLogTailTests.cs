using System.Diagnostics;
using AwesomeAssertions;
using LMSupply.Llama.Server;

namespace LMSupply.Llama.Tests.Server;

/// <summary>
/// A server that dies in the middle of a request leaves the consumer one piece of evidence: what it wrote last.
/// These facts pin that the evidence is kept (both streams, after startup, after exit) and that keeping it is bounded.
/// </summary>
public class ServerLogTailTests
{
    [Fact]
    public void Empty_tail_reads_as_empty()
    {
        new ServerLogTail(3).ToString().Should().BeEmpty();
    }

    [Fact]
    public void Keeps_only_the_last_lines_in_order()
    {
        var tail = new ServerLogTail(3);
        foreach (var line in new[] { "a", "b", "c", "d", "e" })
            tail.Append(line);

        Lines(tail).Should().Equal("c", "d", "e");
    }

    [Fact]
    public void Below_capacity_keeps_everything_in_order()
    {
        var tail = new ServerLogTail(5);
        tail.Append("first");
        tail.Append("second");

        Lines(tail).Should().Equal("first", "second");
    }

    [Fact]
    public void Concurrent_appends_never_exceed_the_capacity()
    {
        var tail = new ServerLogTail(50);
        Parallel.For(0, 10_000, i => tail.Append($"line {i}"));

        Lines(tail).Should().HaveCount(50);
    }

    [Fact]
    public void Capacity_below_one_is_rejected()
    {
        var act = () => new ServerLogTail(0);
        act.Should().Throw<ArgumentOutOfRangeException>();
    }

    [Fact]
    public async Task CaptureOutput_reads_both_streams_and_they_stay_readable_after_exit()
    {
        var startInfo = OperatingSystem.IsWindows()
            ? new ProcessStartInfo("cmd.exe", "/c echo to-stdout& echo to-stderr 1>&2& exit 3")
            : new ProcessStartInfo("/bin/sh", "-c \"echo to-stdout; echo to-stderr 1>&2; exit 3\"");
        startInfo.UseShellExecute = false;
        startInfo.CreateNoWindow = true;
        startInfo.RedirectStandardOutput = true;
        startInfo.RedirectStandardError = true;

        using var process = Process.Start(startInfo)!;
        var tail = new ServerLogTail();
        var stderrSeen = new List<string>();
        LlamaServerProcess.CaptureOutput(process, tail, line => { lock (stderrSeen) stderrSeen.Add(line); });

        // WaitForExitAsync also waits for both redirected streams to reach EOF.
        await process.WaitForExitAsync(TestContext.Current.CancellationToken);

        process.ExitCode.Should().Be(3);
        Lines(tail).Should().Contain("to-stdout", "stdout is drained into the tail, not left to fill its pipe");
        Lines(tail).Should().Contain("to-stderr");
        // cmd's `echo x 1>&2` keeps the space before the redirection.
        stderrSeen.Select(l => l.Trim()).Should().Equal("to-stderr");
    }

    private static string[] Lines(ServerLogTail tail) =>
        tail.ToString().Split(Environment.NewLine, StringSplitOptions.RemoveEmptyEntries)
            .Select(l => l.Trim()).ToArray();
}
