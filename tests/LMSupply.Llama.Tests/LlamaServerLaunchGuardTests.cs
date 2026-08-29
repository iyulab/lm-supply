using System.Diagnostics;
using AwesomeAssertions;
using LMSupply.Llama.Server;

namespace LMSupply.Llama.Tests;

/// <summary>
/// Covers <see cref="LlamaServerProcess.HasLaunched"/>, the guard that decides whether a
/// <see cref="Process"/> handed back by the process guardian actually has an OS process behind it.
///
/// <para>
/// Why the guard exists: a launch that fails still yields a <c>Process</c> instance, and the first
/// member touching the underlying handle throws <c>InvalidOperationException: No process is
/// associated with this object</c>. In <c>StartInternalAsync</c> that first member was
/// <c>BeginErrorReadLine()</c>, three lines above a carefully built diagnostic naming the binary
/// path, backend and exit code — so the bare exception replaced the message written for exactly
/// that failure, and the operator was told the least when the failure was most opaque.
/// </para>
///
/// <para>
/// The discriminator has to separate <em>never started</em> from <em>started and exited</em>: an
/// exited process must NOT be reported as un-launched, because that path has real diagnostics
/// (exit code, captured stderr) worth reaching.
/// </para>
/// </summary>
public class LlamaServerLaunchGuardTests
{
    [Fact]
    public void HasLaunched_IsFalse_ForAProcessThatWasNeverStarted()
    {
        // Exactly what the guardian returns when the launch fails.
        using var neverStarted = new Process();

        LlamaServerProcess.HasLaunched(neverStarted).Should().BeFalse();
    }

    [Fact]
    public void DidNotLaunchMessage_NamesWhatTheOperatorNeedsToCheck()
    {
        // The message is shared by both detection points -- the guardian throwing mid-start, and a
        // returned object with no process behind it -- so that the same failure cannot be described
        // two different ways depending on which one fired.
        var message = LlamaServerProcess.DidNotLaunchMessage(
            "/opt/llama/llama-server", LlamaServerBackend.Cpu, "/opt/llama");

        message.Should().Contain("/opt/llama/llama-server", "the operator has to know WHICH binary failed");
        message.Should().Contain("Cpu", "the backend decides which binary was even attempted");
        message.Should().NotContain("No process is associated",
            "that is the bare framework message this replaces");
    }

    [Fact]
    public void HasLaunched_IsFalse_ForNull()
    {
        LlamaServerProcess.HasLaunched(null).Should().BeFalse();
    }

    [Fact]
    public async Task HasLaunched_IsTrue_ForAProcessThatStartedAndAlreadyExited()
    {
        // The distinction that matters: this one has an exit code and possibly stderr to report,
        // so it must reach the real diagnostic rather than the "did not launch" message.
        var startInfo = OperatingSystem.IsWindows()
            ? new ProcessStartInfo("cmd.exe", "/c exit 0")
            : new ProcessStartInfo("/bin/sh", "-c \"exit 0\"");
        startInfo.UseShellExecute = false;
        startInfo.CreateNoWindow = true;

        using var started = Process.Start(startInfo);
        started.Should().NotBeNull();
        await started!.WaitForExitAsync(TestContext.Current.CancellationToken);

        LlamaServerProcess.HasLaunched(started).Should().BeTrue();
    }
}
