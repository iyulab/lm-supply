using System.Collections.Concurrent;
using System.Diagnostics;
using System.Text.RegularExpressions;

namespace LMSupply.Llama.Server;

/// <summary>
/// Reads the build of a llama-server binary this library did not download (a consumer-supplied
/// <see cref="LlamaServerUpdateOptions.ServerBinaryPath"/>) by running it with <c>--version</c>.
/// Every flag whose spelling changed across llama.cpp builds is gated on the build number; without
/// it an external binary would be treated as "unknown build" and receive spellings a newer server
/// rejects before it opens its port.
/// </summary>
internal static partial class LlamaServerBinaryVersion
{
    private static readonly TimeSpan s_probeTimeout = TimeSpan.FromSeconds(15);

    // Keyed on path + last write time: replacing the binary in place re-probes it.
    private static readonly ConcurrentDictionary<(string Path, DateTime WriteTime), string?> s_cache = new();

    // "version: 0.5.0-dev (build 11146, commit 7fe450e19)" — versioned-release builds (2026-09+).
    [GeneratedRegex(@"\(build\s+(\d+)\b", RegexOptions.IgnoreCase)]
    private static partial Regex BuildWordPattern();

    // "version: 10488 (1a2b3c4d)" — the long-standing form. The "(" anchor keeps it off the
    // leading "0" of a semantic version.
    [GeneratedRegex(@"^\s*version:\s*(\d+)\s*\(", RegexOptions.IgnoreCase | RegexOptions.Multiline)]
    private static partial Regex VersionLinePattern();

    /// <summary>
    /// Maps llama-server's <c>--version</c> output to a build tag (<c>b11146</c>), or null when the
    /// output names no build number.
    /// </summary>
    public static string? ParseBuildTag(string? output)
    {
        if (string.IsNullOrWhiteSpace(output))
            return null;

        var match = BuildWordPattern().Match(output);
        if (!match.Success)
            match = VersionLinePattern().Match(output);

        return match.Success ? "b" + match.Groups[1].Value : null;
    }

    /// <summary>
    /// Runs <paramref name="serverPath"/> with <c>--version</c> and returns the build tag it reports,
    /// or null when the binary cannot be run, does not answer in time, or reports no build number.
    /// Never throws for a binary that misbehaves — the caller falls back to "unknown build".
    /// </summary>
    public static async Task<string?> ProbeAsync(string serverPath, CancellationToken cancellationToken = default)
    {
        DateTime writeTime;
        try
        {
            writeTime = File.GetLastWriteTimeUtc(serverPath);
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
        {
            return null;
        }

        var key = (Path.GetFullPath(serverPath), writeTime);
        if (s_cache.TryGetValue(key, out var cached))
            return cached;

        var tag = await RunAsync(serverPath, cancellationToken).ConfigureAwait(false);
        s_cache[key] = tag;
        return tag;
    }

    private static async Task<string?> RunAsync(string serverPath, CancellationToken cancellationToken)
    {
        var startInfo = new ProcessStartInfo
        {
            FileName = serverPath,
            UseShellExecute = false,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            CreateNoWindow = true,
            WorkingDirectory = Path.GetDirectoryName(Path.GetFullPath(serverPath)) ?? string.Empty,
        };
        startInfo.ArgumentList.Add("--version");

        using var timeout = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
        timeout.CancelAfter(s_probeTimeout);

        Process? process = null;
        try
        {
            process = Process.Start(startInfo);
            if (process is null)
                return null;

            // llama-server prints the version to stderr on most builds and to stdout on some.
            var stdout = process.StandardOutput.ReadToEndAsync(timeout.Token);
            var stderr = process.StandardError.ReadToEndAsync(timeout.Token);
            await process.WaitForExitAsync(timeout.Token).ConfigureAwait(false);

            return ParseBuildTag(await stderr.ConfigureAwait(false))
                ?? ParseBuildTag(await stdout.ConfigureAwait(false));
        }
        catch (OperationCanceledException) when (!cancellationToken.IsCancellationRequested)
        {
            Trace.TraceInformation($"[LlamaServerBinaryVersion] '{serverPath} --version' did not answer within {s_probeTimeout.TotalSeconds:0} s.");
            return null;
        }
        catch (Exception ex) when (ex is System.ComponentModel.Win32Exception or IOException or InvalidOperationException)
        {
            Trace.TraceInformation($"[LlamaServerBinaryVersion] Could not run '{serverPath} --version': {ex.Message}");
            return null;
        }
        finally
        {
            if (process is not null)
            {
                try
                {
                    if (!process.HasExited)
                        process.Kill(entireProcessTree: true);
                }
                catch (InvalidOperationException)
                {
                    // Already gone.
                }

                process.Dispose();
            }
        }
    }
}
