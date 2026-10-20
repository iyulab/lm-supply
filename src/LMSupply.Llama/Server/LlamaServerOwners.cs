using System.Diagnostics;
using System.Globalization;

namespace LMSupply.Llama.Server;

/// <summary>
/// Records which process owns each llama-server it started, so that a server whose owner died without stopping it is
/// stopped by the next process that starts one.
/// </summary>
/// <remarks>
/// <para>
/// A server is a child process. On Windows the kernel ends it with its owner (a job object), but on Linux and macOS
/// nothing does when the owner is killed or crashes: the exit hooks that stop the server never run, and the server
/// keeps its port and the model's memory until someone ends it by hand.
/// </para>
/// <para>
/// Each started server leaves an owner record, one small file named after its process id under <c>owners/</c> in the
/// llama-server cache directory: the server's id, start time and process name, and its owner's id and start time.
/// A clean stop removes it. Before starting a server, the records are read: one whose owner is no longer running —
/// no process with that id, or one that started at another time, which is a reused id — has its server stopped,
/// provided a process with the server's id, start time and name is still running, and the record is removed. A
/// process is matched by its start time as well as its id, so a reused id is never taken for the server. Everything
/// here is best-effort: a record that cannot be written, read or removed is reported through tracing and never stops a
/// server from starting.
/// </para>
/// </remarks>
internal static class LlamaServerOwners
{
    // Two reads of one process's start time agree to well under this.
    private static readonly TimeSpan StartTimeTolerance = TimeSpan.FromSeconds(1);

    internal static string DefaultDirectory => Path.Combine(LMSupplyCachePaths.GetLlamaServerDirectory(), "owners");

    /// <summary>Records <paramref name="server"/> as owned by the current process; returns the record's path, or null when none was written.</summary>
    internal static string? Take(Process server, string? directory = null)
    {
        try
        {
            directory ??= DefaultDirectory;
            Directory.CreateDirectory(directory);
            using var owner = Process.GetCurrentProcess();
            var path = Path.Combine(directory, server.Id.ToString(CultureInfo.InvariantCulture) + ".owner");
            File.WriteAllText(path, string.Join('\n',
                server.Id.ToString(CultureInfo.InvariantCulture),
                StartTicks(server).ToString(CultureInfo.InvariantCulture),
                server.ProcessName,
                owner.Id.ToString(CultureInfo.InvariantCulture),
                StartTicks(owner).ToString(CultureInfo.InvariantCulture)));
            return path;
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException or InvalidOperationException or System.ComponentModel.Win32Exception or NotSupportedException)
        {
            Trace.TraceInformation($"[LlamaServerOwners] Owner record not written: {ex.Message}");
            return null;
        }
    }

    /// <summary>Removes a record written by <see cref="Take"/>.</summary>
    internal static void Release(string? path)
    {
        if (path is null)
            return;
        try
        {
            File.Delete(path);
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
        {
            Trace.TraceInformation($"[LlamaServerOwners] Owner record not removed: {ex.Message}");
        }
    }

    /// <summary>Stops every recorded server whose owner is no longer running, and removes its record; returns how many servers were stopped.</summary>
    internal static int StopOrphans(string? directory = null)
    {
        directory ??= DefaultDirectory;
        string[] records;
        try
        {
            if (!Directory.Exists(directory))
                return 0;
            records = Directory.GetFiles(directory, "*.owner");
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
        {
            Trace.TraceInformation($"[LlamaServerOwners] Owner records not read: {ex.Message}");
            return 0;
        }

        var stopped = 0;
        foreach (var path in records)
        {
            try
            {
                var fields = File.ReadAllText(path).Split('\n');
                if (fields.Length != 5
                    || !int.TryParse(fields[0], NumberStyles.None, CultureInfo.InvariantCulture, out var serverId)
                    || !long.TryParse(fields[1], NumberStyles.None, CultureInfo.InvariantCulture, out var serverStart)
                    || !int.TryParse(fields[3], NumberStyles.None, CultureInfo.InvariantCulture, out var ownerId)
                    || !long.TryParse(fields[4], NumberStyles.None, CultureInfo.InvariantCulture, out var ownerStart))
                {
                    // Not a record this code wrote: nothing in it can be trusted to name a process.
                    File.Delete(path);
                    continue;
                }

                using (var owner = Running(ownerId, ownerStart, name: null))
                {
                    if (owner is not null)
                        continue;
                }

                using (var server = Running(serverId, serverStart, fields[2]))
                {
                    if (server is not null)
                    {
                        Trace.TraceWarning($"[LlamaServerOwners] Stopping llama-server {serverId}: the process that started it ({ownerId}) is gone");
                        server.Kill(entireProcessTree: true);
                        stopped++;
                    }
                }

                File.Delete(path);
            }
            catch (Exception ex) when (ex is IOException or UnauthorizedAccessException or InvalidOperationException or System.ComponentModel.Win32Exception or NotSupportedException)
            {
                Trace.TraceInformation($"[LlamaServerOwners] Owner record {Path.GetFileName(path)} not handled: {ex.Message}");
            }
        }

        return stopped;
    }

    // The running process with this id, when it started at the recorded time (and, given a name, has that name); null otherwise.
    private static Process? Running(int id, long startTicks, string? name)
    {
        Process process;
        try
        {
            process = Process.GetProcessById(id);
        }
        catch (ArgumentException)
        {
            return null;
        }

        try
        {
            if (!process.HasExited
                && Math.Abs(StartTicks(process) - startTicks) <= StartTimeTolerance.Ticks
                && (name is null || string.Equals(process.ProcessName, name, StringComparison.Ordinal)))
            {
                return process;
            }
        }
        catch (Exception ex) when (ex is InvalidOperationException or System.ComponentModel.Win32Exception or NotSupportedException)
        {
        }

        process.Dispose();
        return null;
    }

    private static long StartTicks(Process process) => process.StartTime.ToUniversalTime().Ticks;
}
