using System.Diagnostics;
using System.Formats.Tar;
using System.IO.Compression;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Threading.Channels;

namespace LMSupply.Console.Host.Services;

/// <summary>
/// Self-update service based on GitHub Releases
/// </summary>
public sealed partial class UpdateService
{
    private const string GitHubApiUrl = "https://api.github.com/repos/iyulab/lm-supply/releases/latest";
    private static readonly HttpClient HttpClient = CreateHttpClient();
    private static readonly TimeSpan CacheDuration = TimeSpan.FromMinutes(30);

    private readonly IHostApplicationLifetime _lifetime;
    private readonly ILogger<UpdateService> _logger;
    private UpdateCheckResult? _cachedResult;
    private DateTime _cacheExpiry = DateTime.MinValue;

    public string CurrentVersion { get; }
    public string CurrentRid { get; }

    public UpdateService(IHostApplicationLifetime lifetime, ILogger<UpdateService> logger)
    {
        _lifetime = lifetime;
        _logger = logger;
        CurrentVersion = GetCurrentVersion();
        CurrentRid = DetectRid();
        CleanupPreviousUpdate();
    }

    public async Task<UpdateCheckResult> CheckForUpdateAsync(bool forceCheck = false)
    {
        if (!forceCheck && _cachedResult is not null && DateTime.UtcNow < _cacheExpiry)
            return _cachedResult;

        try
        {
            var json = await HttpClient.GetStringAsync(GitHubApiUrl);
            var release = JsonSerializer.Deserialize<GitHubRelease>(json);
            if (release is null)
                return new UpdateCheckResult { CurrentVersion = CurrentVersion };

            var latestVersion = release.TagName.TrimStart('v');
            var hasNewerVersion = CompareVersions(CurrentVersion, latestVersion) < 0;

            var assetName = GetAssetName(CurrentRid);
            var asset = release.Assets?.FirstOrDefault(a => a.Name == assetName);
            var updateAvailable = hasNewerVersion && asset?.BrowserDownloadUrl is not null;

            var result = new UpdateCheckResult
            {
                CurrentVersion = CurrentVersion,
                LatestVersion = latestVersion,
                UpdateAvailable = updateAvailable,
                ReleaseUrl = release.HtmlUrl,
                DownloadUrl = asset?.BrowserDownloadUrl,
                AssetSize = asset?.Size ?? 0,
                ReleaseNotes = release.Body
            };

            _cachedResult = result;
            _cacheExpiry = DateTime.UtcNow + CacheDuration;
            return result;
        }
        catch (Exception ex)
        {
            LogCheckFailed(_logger, ex);
            return new UpdateCheckResult
            {
                CurrentVersion = CurrentVersion,
                Error = ex.Message
            };
        }
    }

    public IAsyncEnumerable<UpdateProgress> ApplyUpdateAsync(CancellationToken ct = default)
    {
        var channel = Channel.CreateUnbounded<UpdateProgress>();
        _ = ExecuteUpdateAsync(channel.Writer, ct);
        return channel.Reader.ReadAllAsync(ct);
    }

    private async Task ExecuteUpdateAsync(ChannelWriter<UpdateProgress> writer, CancellationToken ct)
    {
        var tempDir = (string?)null;
        var scriptLaunched = false;

        try
        {
            var check = await CheckForUpdateAsync(forceCheck: true);
            if (!check.UpdateAvailable || check.DownloadUrl is null)
            {
                await writer.WriteAsync(new UpdateProgress { Status = "Error", Error = "No update available" }, ct);
                return;
            }

            tempDir = Path.Combine(Path.GetTempPath(), $"lm-supply-update-{check.LatestVersion}");
            var extractDir = Path.Combine(tempDir, "extracted");
            Directory.CreateDirectory(tempDir);
            Directory.CreateDirectory(extractDir);

            var assetName = GetAssetName(CurrentRid);
            var downloadPath = Path.Combine(tempDir, assetName);

            // 1. Download
            await writer.WriteAsync(new UpdateProgress { Status = "Downloading", Percent = 0 }, ct);

            using (var response = await HttpClient.GetAsync(check.DownloadUrl, HttpCompletionOption.ResponseHeadersRead, ct))
            {
                response.EnsureSuccessStatusCode();
                var totalBytes = response.Content.Headers.ContentLength ?? check.AssetSize;
                var downloadedBytes = 0L;
                var lastReportedPercent = -1;

                await using var contentStream = await response.Content.ReadAsStreamAsync(ct);
                await using var fileStream = File.Create(downloadPath);
                var buffer = new byte[81920];
                int bytesRead;
                while ((bytesRead = await contentStream.ReadAsync(buffer, ct)) > 0)
                {
                    await fileStream.WriteAsync(buffer.AsMemory(0, bytesRead), ct);
                    downloadedBytes += bytesRead;
                    if (totalBytes > 0)
                    {
                        var percent = (int)(downloadedBytes * 100 / totalBytes);
                        if (percent != lastReportedPercent)
                        {
                            lastReportedPercent = percent;
                            await writer.WriteAsync(new UpdateProgress { Status = "Downloading", Percent = percent }, ct);
                        }
                    }
                }
            }

            // 2. Extract
            await writer.WriteAsync(new UpdateProgress { Status = "Extracting", Percent = 0 }, ct);

            if (assetName.EndsWith(".zip", StringComparison.OrdinalIgnoreCase))
                ZipFile.ExtractToDirectory(downloadPath, extractDir, overwriteFiles: true);
            else
                await ExtractTarGzAsync(downloadPath, extractDir);

            // 3. Prepare restart script
            // Wait for the current process to exit, replace the files, then start a new process (avoids port conflicts and file locks)
            await writer.WriteAsync(new UpdateProgress { Status = "Replacing", Percent = 0 }, ct);

            var currentExePath = Environment.ProcessPath
                ?? throw new InvalidOperationException("Cannot determine current executable path");
            var currentDir = Path.GetDirectoryName(currentExePath)!;
            var currentPid = Environment.ProcessId;
            // CLI command args (such as "update") are not passed on restart — the new process always starts in server mode
            var userArgs = string.Join(" ", Environment.GetCommandLineArgs().Skip(1)
                .Where(a => !a.Equals("update", StringComparison.OrdinalIgnoreCase))
                .Select(a => a.Contains(' ') ? $"\"{a}\"" : a));

            if (OperatingSystem.IsWindows())
            {
                var scriptPath = Path.Combine(tempDir, "restart.cmd");
                var script = string.Join("\r\n",
                    "@echo off",
                    // Wait for the current process to exit.
                    // Match on findstr /I "lm-supply.exe": when the PID no longer exists, tasklist prints
                    // "INFO: No tasks are running..." to stdout, which a ^[A-Za-z] pattern would also match, looping forever.
                    ":wait",
                    $"tasklist /FI \"PID eq {currentPid}\" /NH 2>NUL | findstr /I \"lm-supply.exe\" >NUL",
                    "if not errorlevel 1 (timeout /t 1 /nobreak >NUL & goto wait)",
                    // Wait for file locks to be released
                    "timeout /t 1 /nobreak >NUL",
                    // Copy the new files (no locks, since the process has exited)
                    $"xcopy \"{extractDir}\\*\" \"{currentDir}\\\" /s /y /q >NUL",
                    // Start the new process (/D sets the working directory)
                    $"start \"\" /D \"{currentDir}\" \"{currentExePath}\" {userArgs}");
                await File.WriteAllTextAsync(scriptPath, script, ct);

                Process.Start(new ProcessStartInfo
                {
                    FileName = "cmd.exe",
                    Arguments = $"/c \"{scriptPath}\"",
                    UseShellExecute = true,
                    WindowStyle = ProcessWindowStyle.Hidden,
                });
            }
            else
            {
                // Unix: a running executable can be overwritten, so copy first
                foreach (var file in Directory.GetFiles(extractDir, "*", SearchOption.AllDirectories))
                {
                    var relativePath = Path.GetRelativePath(extractDir, file);
                    var destPath = Path.Combine(currentDir, relativePath);
                    Directory.CreateDirectory(Path.GetDirectoryName(destPath)!);
                    File.Copy(file, destPath, overwrite: true);
                }

                var newExe = Path.Combine(currentDir, Path.GetFileName(currentExePath));
                using var chmod = Process.Start("chmod", ["+x", newExe]);
                chmod?.WaitForExit(5000);

                // Wait for the process to exit, then start the new process
                var scriptPath = Path.Combine(tempDir, "restart.sh");
                var script = string.Join("\n",
                    "#!/bin/bash",
                    $"while kill -0 {currentPid} 2>/dev/null; do sleep 1; done",
                    $"\"{currentExePath}\" {userArgs} &");
                await File.WriteAllTextAsync(scriptPath, script, ct);

                using var chmod2 = Process.Start("chmod", ["+x", scriptPath]);
                chmod2?.WaitForExit(5000);

                Process.Start(new ProcessStartInfo
                {
                    FileName = "/bin/bash",
                    Arguments = scriptPath,
                    UseShellExecute = false,
                    CreateNoWindow = true,
                });
            }

            scriptLaunched = true;

            // 4. Restart
            await writer.WriteAsync(new UpdateProgress { Status = "Restarting", Percent = 100 }, ct);

            // Schedule shutdown after the response is sent
            _ = Task.Run(async () =>
            {
                await Task.Delay(1500);
                _lifetime.StopApplication();
            }, CancellationToken.None);
        }
        catch (Exception ex)
        {
            LogUpdateFailed(_logger, ex);
            await writer.WriteAsync(new UpdateProgress { Status = "Error", Error = ex.Message }, CancellationToken.None);
        }
        finally
        {
            writer.Complete();

            // Keep the temp directory if the script was launched (cleaned up on next start)
            if (tempDir is not null && !scriptLaunched)
            {
                try { Directory.Delete(tempDir, true); }
                catch (Exception ex)
                {
                    Trace.TraceInformation($"[UpdateService] cleanup temp dir: {ex.Message}");
                }
            }
        }
    }

    private static string GetCurrentVersion()
    {
        var asm = Assembly.GetEntryAssembly();
        var infoVersion = asm?.GetCustomAttribute<AssemblyInformationalVersionAttribute>()?.InformationalVersion;
        if (infoVersion is not null)
            return infoVersion.Split('+')[0];
        return asm?.GetName().Version?.ToString(3) ?? "0.0.0";
    }

    private static string DetectRid()
    {
        if (OperatingSystem.IsWindows()) return "win-x64";
        if (OperatingSystem.IsLinux()) return "linux-x64";
        if (OperatingSystem.IsMacOS())
            return RuntimeInformation.OSArchitecture == Architecture.Arm64 ? "osx-arm64" : "osx-x64";
        return RuntimeInformation.RuntimeIdentifier;
    }

    private static string GetAssetName(string rid) => rid switch
    {
        "win-x64" => "lm-supply-win-x64.zip",
        "linux-x64" => "lm-supply-linux-x64.tar.gz",
        "osx-x64" => "lm-supply-osx-x64.tar.gz",
        "osx-arm64" => "lm-supply-osx-arm64.tar.gz",
        _ => $"lm-supply-{rid}.zip"
    };

    private static int CompareVersions(string current, string latest)
    {
        var cur = current.Split(['+', '-'])[0];
        var lat = latest.Split(['+', '-'])[0];
        return Version.TryParse(cur, out var v1) && Version.TryParse(lat, out var v2)
            ? v1.CompareTo(v2)
            : string.Compare(current, latest, StringComparison.Ordinal);
    }

    private static async Task ExtractTarGzAsync(string archivePath, string extractDir)
    {
        await using var fs = File.OpenRead(archivePath);
        await using var gz = new GZipStream(fs, CompressionMode.Decompress);
        await TarFile.ExtractToDirectoryAsync(gz, extractDir, overwriteFiles: true);
    }

    private static void CleanupPreviousUpdate()
    {
        try
        {
            var exePath = Environment.ProcessPath;
            if (exePath is null) return;

            // Remove the .bak file
            var backupPath = exePath + ".bak";
            if (File.Exists(backupPath))
                File.Delete(backupPath);

            // Remove temp directories from previous updates
            foreach (var dir in Directory.GetDirectories(Path.GetTempPath(), "lm-supply-update-*"))
            {
                try { Directory.Delete(dir, true); }
                catch { /* ignore if in use */ }
            }
        }
        catch (Exception ex)
        {
            Trace.TraceInformation($"[UpdateService] cleanup previous update: {ex.Message}");
        }
    }

    private static HttpClient CreateHttpClient()
    {
        var client = new HttpClient();
        client.DefaultRequestHeaders.Add("User-Agent", "LMSupply-Console");
        client.DefaultRequestHeaders.Add("Accept", "application/vnd.github.v3+json");
        return client;
    }

    [LoggerMessage(Level = LogLevel.Warning, Message = "Failed to check for updates")]
    private static partial void LogCheckFailed(ILogger logger, Exception exception);

    [LoggerMessage(Level = LogLevel.Error, Message = "Update failed")]
    private static partial void LogUpdateFailed(ILogger logger, Exception exception);
}

// GitHub API models
public sealed class GitHubRelease
{
    [JsonPropertyName("tag_name")] public string TagName { get; set; } = "";
    [JsonPropertyName("html_url")] public string HtmlUrl { get; set; } = "";
    [JsonPropertyName("body")] public string? Body { get; set; }
    [JsonPropertyName("assets")] public List<GitHubAsset>? Assets { get; set; }
}

public sealed class GitHubAsset
{
    [JsonPropertyName("name")] public string Name { get; set; } = "";
    [JsonPropertyName("browser_download_url")] public string BrowserDownloadUrl { get; set; } = "";
    [JsonPropertyName("size")] public long Size { get; set; }
}

// API response models
public sealed class UpdateCheckResult
{
    public string CurrentVersion { get; set; } = "";
    public string? LatestVersion { get; set; }
    public bool UpdateAvailable { get; set; }
    public string? ReleaseUrl { get; set; }
    public string? DownloadUrl { get; set; }
    public long AssetSize { get; set; }
    public string? ReleaseNotes { get; set; }
    public string? Error { get; set; }
}

public sealed class UpdateProgress
{
    public string Status { get; set; } = "";
    public int Percent { get; set; }
    public string? Error { get; set; }
}
