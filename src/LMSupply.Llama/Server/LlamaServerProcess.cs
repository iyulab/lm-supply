using System.Diagnostics;
using System.Net;
using System.Net.Sockets;
using System.Text.RegularExpressions;
using ChildProcessGuard;

namespace LMSupply.Llama.Server;

/// <summary>
/// Manages llama-server process lifecycle with automatic cleanup.
/// </summary>
public sealed partial class LlamaServerProcess : IAsyncDisposable
{
    /// <summary>
    /// The address the server listens on and every client connects to. A literal, not <c>localhost</c>: <c>localhost</c>
    /// resolves to <c>::1</c> first, the server listens on IPv4 only, and on Windows a refused IPv6 connect takes about two
    /// seconds — so every new connection (the startup health poll, the first request of each client) paid it.
    /// </summary>
    public const string LoopbackHost = "127.0.0.1";

    private readonly ProcessGuardian _guardian;
    private readonly LlamaServerConfig _config;
    private readonly string _serverPath;
    private readonly LlamaServerBackend _backend;
    private readonly HttpClient _httpClient;

    /// <summary>The variable llama-server reads its key from (the environment form of <c>--api-key</c>).</summary>
    private const string ApiKeyVariable = "LLAMA_API_KEY";

    /// <summary>The environment form of <c>--api-key-file</c>.</summary>
    private const string ApiKeyFileVariable = "LLAMA_ARG_API_KEY_FILE";

    private Process? _process;

    // This server's owner record (LlamaServerOwners), removed when it is disposed.
    private string? _ownerRecord;
    private int _port;
    private bool _disposed;
    private readonly ServerLogTail _recentLog = new();

    /// <summary>
    /// Gets information about the running server.
    /// </summary>
    public LlamaServerInfo? Info { get; private set; }

    /// <summary>
    /// The key this server requires, to be sent as <c>Authorization: Bearer</c> by any client that calls
    /// <see cref="LlamaServerInfo.BaseUrl"/> itself; <see langword="null"/> when
    /// <see cref="LlamaServerConfig.RequireApiKey"/> is off. A secret: do not log it. Kept off
    /// <see cref="LlamaServerInfo"/> because a record's generated <c>ToString()</c> prints every property.
    /// </summary>
    public string? ApiKey { get; }

    /// <summary>
    /// Gets whether the server is running.
    /// </summary>
    public bool IsRunning => _process is { HasExited: false };

    /// <summary>The process exit code once it has exited, else null.</summary>
    public int? ExitCode => _process is { HasExited: true } exited ? exited.ExitCode : null;

    /// <summary>
    /// The last lines the server wrote (stderr and stdout, at most <see cref="ServerLogTail.DefaultCapacity"/> lines,
    /// oldest first) — kept for the server's whole life and still readable after it exits, so a server that dies in the
    /// middle of a request can say why. <see cref="LlamaServerInfo.StartupLog"/> is the complete log up to readiness only.
    /// </summary>
    public string RecentLog => _recentLog.ToString();

    private LlamaServerProcess(
        string serverPath,
        LlamaServerConfig config,
        LlamaServerBackend backend)
    {
        _serverPath = serverPath;
        _config = config;
        _backend = backend;

        _guardian = new ProcessGuardian(new ProcessGuardianOptions
        {
            ProcessKillTimeout = config.ShutdownTimeout
        });

        _httpClient = new HttpClient
        {
            Timeout = TimeSpan.FromSeconds(5)
        };

        if (config.RequireApiKey)
        {
            ApiKey = Convert.ToHexString(System.Security.Cryptography.RandomNumberGenerator.GetBytes(32));
            _httpClient.DefaultRequestHeaders.Authorization = new System.Net.Http.Headers.AuthenticationHeaderValue("Bearer", ApiKey);
        }
    }

    /// <summary>
    /// Starts a new llama-server process.
    /// </summary>
    public static async Task<LlamaServerProcess> StartAsync(
        string serverPath,
        LlamaServerConfig config,
        LlamaServerBackend backend,
        CancellationToken cancellationToken = default)
    {
        if (config.RequireApiKey && FindApiKeyArgument(config.AdditionalArgs) is { } keyArgument)
        {
            throw new ArgumentException(
                $"AdditionalArgs contains '{keyArgument}', but LlamaServerConfig.RequireApiKey is on and LMSupply sets the " +
                "server's key itself. Remove the argument, or set RequireApiKey = false to manage the key yourself " +
                "(LMSupply's own client then sends no key).",
                nameof(config));
        }

        var server = new LlamaServerProcess(serverPath, config, backend);

        try
        {
            await server.StartInternalAsync(cancellationToken);
            return server;
        }
        catch
        {
            await server.DisposeAsync();
            throw;
        }
    }

    private async Task StartInternalAsync(CancellationToken cancellationToken)
    {
        // Find available port if not specified
        _port = _config.Port > 0 ? _config.Port : FindAvailablePort();

        // Build arguments
        var args = BuildArguments();

        // Get the directory containing llama-server for DLL resolution
        var workingDir = Path.GetDirectoryName(_serverPath)!;

        // A server whose owner died without stopping it holds a port and a model's memory: stop it first.
        LlamaServerOwners.StopOrphans();

        // Start process via guardian
        var startInfo = new ProcessStartInfo
        {
            FileName = _serverPath,
            Arguments = args,
            WorkingDirectory = workingDir,
            UseShellExecute = false,
            CreateNoWindow = true,
            RedirectStandardOutput = true,
            RedirectStandardError = true
        };

        // The key travels in the environment, not the command line, so other processes cannot read it from the
        // process list. A key inherited from this process's own environment is removed either way: with
        // RequireApiKey on it would be replaced, and with it off it would make the server refuse LMSupply's client.
        startInfo.Environment.Remove(ApiKeyVariable);
        startInfo.Environment.Remove(ApiKeyFileVariable);
        if (ApiKey is not null)
        {
            startInfo.Environment[ApiKeyVariable] = ApiKey;
        }

        // Enable CUDA graph optimization for NVIDIA GPUs (reduces token generation latency)
        if (_backend == LlamaServerBackend.Cuda12 || _backend == LlamaServerBackend.Cuda13)
        {
            startInfo.Environment["GGML_CUDA_GRAPH_OPT"] = "1";
        }

        // A launch that never happened yields "No process is associated with this object" -- a bare
        // message with no binary path, no backend, nothing to act on. It replaces the diagnostic a
        // few lines further down, which is the one written for exactly this situation, so the caller
        // is told the LEAST precisely when the failure is most opaque.
        //
        // The throw comes from two places and both have to be covered. The guardian reads Id while
        // starting, so the exception can surface before this method ever gets a Process back; and if
        // a future guardian returns without touching it, the first member here that needs the
        // handle throws instead. Guarding only the returned object -- as this did between 0.37.2 and
        // 0.38.1 -- covers the second and misses the first, which is the one that actually occurs.
        try
        {
            _process = _guardian.StartProcessWithStartInfo(startInfo);
        }
        catch (InvalidOperationException ex)
        {
            throw new InvalidOperationException(DidNotLaunchMessage(_serverPath, _backend, workingDir), ex);
        }

        if (!HasLaunched(_process))
        {
            throw new InvalidOperationException(DidNotLaunchMessage(_serverPath, _backend, workingDir));
        }

        _ownerRecord = LlamaServerOwners.Take(_process);

        // Capture output for diagnostics. The complete startup log is kept only until the server is ready (a running
        // server logs every request, so an unbounded buffer grows for its whole life); after that only the bounded
        // tail keeps growing. stdout is drained too: a redirected stream nobody reads fills its pipe and stalls the server.
        var stderrBuilder = new System.Text.StringBuilder();
        var startupDone = false;
        CaptureOutput(_process, _recentLog, line =>
        {
            if (!Volatile.Read(ref startupDone))
            {
                lock (stderrBuilder)
                    stderrBuilder.AppendLine(line);
            }
        });

        // Wait for server to be ready
        var startTime = DateTimeOffset.UtcNow;
        var failure = await WaitForServerReadyAsync(startTime.UtcDateTime, stderrBuilder, cancellationToken);

        if (failure != null)
        {
            // Collect error output
            string error;
            lock (stderrBuilder)
                error = stderrBuilder.ToString();
            if (string.IsNullOrEmpty(error) && _process.HasExited)
            {
                error = "Process exited without error output";
            }

            throw new InvalidOperationException(
                $"llama-server failed to start: {failure}. " +
                $"Exit code: {(_process.HasExited ? _process.ExitCode : "still running")}. " +
                $"Error: {error}");
        }

        string startupLog;
        lock (stderrBuilder)
        {
            Volatile.Write(ref startupDone, true);
            startupLog = stderrBuilder.ToString();
        }

        Info = new LlamaServerInfo
        {
            ProcessId = _process.Id,
            Port = _port,
            ModelPath = _config.ModelPath,
            Backend = _backend,
            StartTime = startTime,
            StartupLog = startupLog
        };

        // Silent-fallback guard: a GPU backend whose runtime cannot load (e.g. cudart/cublas missing
        // alongside the cuda binary, or a driver/runtime mismatch) starts and serves normally but
        // llama.cpp enumerates only a CPU device and runs on CPU with no error. Surface it as a
        // warning so the consumer is not silently downgraded to CPU performance.
        if (IsGpuBackend(_backend) && !(StartupLogDeviceEvidence(startupLog) ?? ProbeListsGpuDevice(_serverPath)))
        {
            Trace.TraceWarning(
                $"[LlamaServerProcess] {_backend} backend was selected but llama-server initialized " +
                "CPU-only (no GPU device enumerated). The GPU runtime likely failed to load " +
                "(e.g. CUDA runtime cudart/cublas missing alongside the server binary, or a " +
                "driver/runtime mismatch). Inference will run on CPU. Install the GPU runtime or " +
                "ensure the runtime libraries are present next to the llama-server binary.");
        }
    }

    /// <summary>
    /// Polls <c>/health</c> until the server answers, the process dies, stderr shows a fatal CLI
    /// error, or <see cref="StartupProgressTracker"/> declares the start stalled or capped.
    /// Returns <c>null</c> when ready, otherwise a one-line reason for the failure.
    /// </summary>
    private async Task<string?> WaitForServerReadyAsync(
        DateTime startedAt,
        System.Text.StringBuilder stderrBuilder,
        CancellationToken cancellationToken)
    {
        var tracker = new StartupProgressTracker(startedAt, _config.StartupStallTimeout, _config.StartupTimeout);

        while (true)
        {
            cancellationToken.ThrowIfCancellationRequested();

            if (_process?.HasExited == true)
            {
                return "the process exited before answering /health";
            }

            // Fail fast on fatal CLI-parse errors — llama.cpp prints these before the HTTP port
            // is up and the process may linger briefly, so don't wait for HasExited or the full timeout.
            if (HasFatalStartupError(stderrBuilder))
            {
                return "llama-server rejected its command line (see stderr)";
            }

            try
            {
                var response = await _httpClient.GetAsync($"http://{LoopbackHost}:{_port}/health", cancellationToken);
                if (response.IsSuccessStatusCode)
                {
                    return null;
                }
            }
            catch (HttpRequestException)
            {
                // Server not ready yet
            }
            catch (TaskCanceledException)
            {
                // Timeout, retry
            }

            var now = DateTime.UtcNow;
            var sample = SampleProcess(_process);
            if (sample == null)
            {
                // The process went away between the HasExited check and the read.
                return "the process exited before answering /health";
            }

            var verdict = tracker.Observe(now, stderrBuilder.Length, sample.Value.WorkingSet, sample.Value.CpuTime);
            if (verdict != StartupWaitVerdict.Waiting)
            {
                return tracker.Describe(now);
            }

            await Task.Delay(100, cancellationToken);
        }
    }

    /// <summary>
    /// Reads the process metrics the stall detector watches. <c>null</c> when the process is gone —
    /// <see cref="Process.Refresh"/> and the counters throw once the process has exited, and the
    /// caller's earlier <see cref="Process.HasExited"/> check can race that.
    /// </summary>
    private static (long WorkingSet, TimeSpan CpuTime)? SampleProcess(Process? process)
    {
        if (process == null)
            return null;

        try
        {
            process.Refresh();
            return (process.WorkingSet64, process.TotalProcessorTime);
        }
        catch (InvalidOperationException)
        {
            return null;
        }
    }

    private static bool HasFatalStartupError(System.Text.StringBuilder stderrBuilder)
    {
        string text;
        lock (stderrBuilder)
        {
            if (stderrBuilder.Length == 0)
                return false;
            text = stderrBuilder.ToString();
        }
        return text.Contains("error while handling argument", StringComparison.Ordinal)
            || text.Contains("error: invalid argument", StringComparison.Ordinal)
            || text.Contains("unknown argument", StringComparison.Ordinal);
    }

    /// <summary>
    /// Checks if the server is healthy.
    /// </summary>
    public async Task<bool> CheckHealthAsync(CancellationToken cancellationToken = default)
    {
        if (!IsRunning)
            return false;

        try
        {
            var response = await _httpClient.GetAsync($"http://{LoopbackHost}:{_port}/health", cancellationToken);
            return response.IsSuccessStatusCode;
        }
        catch (Exception ex) when (ex is not OperationCanceledException || !cancellationToken.IsCancellationRequested)
        {
            Trace.TraceInformation($"[LlamaServerProcess] Health check failed: {ex.Message}");
            return false;
        }
    }

    /// <summary>
    /// Stops the server gracefully.
    /// </summary>
    public async Task StopAsync(CancellationToken cancellationToken = default)
    {
        if (_process == null || _process.HasExited)
            return;

        try
        {
            // Try graceful shutdown first
            _process.Kill(entireProcessTree: true);
            await _process.WaitForExitAsync(cancellationToken);
        }
        catch (InvalidOperationException)
        {
            // Process already exited
        }
    }

    private static int FindAvailablePort()
    {
        using var listener = new TcpListener(IPAddress.Loopback, 0);
        listener.Start();
        var port = ((IPEndPoint)listener.LocalEndpoint).Port;
        listener.Stop();
        return port;
    }

    // device_info bullet line for an accelerated (non-CPU) compute device, e.g.
    // "  - CUDA0   : NVIDIA GeForce RTX 4060 ..." or "  - Vulkan0 : ...". A CPU-only line
    // ("  - CPU : ...") does not match. The leading "- " anchors to the device list so model
    // names or system_info text containing a backend keyword are not mistaken for a device.
    private static readonly Regex GpuDeviceLineRegex = new(
        @"-\s*(?:CUDA|Vulkan|Metal|ROCm|HIP|SYCL|CANN|OpenCL)\d*\s*:",
        RegexOptions.IgnoreCase | RegexOptions.Compiled);

    /// <summary>
    /// The message for a launch that produced no process. Shared by both detection points so they
    /// cannot drift into saying different things about the same failure.
    /// </summary>
    internal static string DidNotLaunchMessage(string serverPath, LlamaServerBackend backend, string workingDir) =>
        $"llama-server did not launch: '{serverPath}' (backend {backend}, working directory " +
        $"'{workingDir}'). The process was never created, so there is no exit code or error output " +
        "to report. Check that the binary exists at that path, is executable, and that its runtime " +
        "dependencies are resolvable on this machine.";

    /// <summary>
    /// True when the object actually has an OS process behind it. A failed launch still yields a
    /// <see cref="Process"/> instance, and every member that needs the underlying handle -- including
    /// <c>Id</c> -- throws <see cref="InvalidOperationException"/> on it. Probing <c>Id</c> is the
    /// discriminator: it succeeds for a process that has already exited, and fails only for one that
    /// was never created.
    /// </summary>
    /// <summary>
    /// Reads both redirected streams into <paramref name="tail"/> for the process's whole life, and hands each stderr
    /// line to <paramref name="onStderr"/> as well. Both streams, because a redirected stream nobody reads fills its
    /// pipe and blocks the writer.
    /// </summary>
    internal static void CaptureOutput(Process process, ServerLogTail tail, Action<string>? onStderr = null)
    {
        process.ErrorDataReceived += (_, e) =>
        {
            if (e.Data == null)
                return;
            tail.Append(e.Data);
            onStderr?.Invoke(e.Data);
        };
        process.OutputDataReceived += (_, e) =>
        {
            if (e.Data != null)
                tail.Append(e.Data);
        };
        process.BeginErrorReadLine();
        process.BeginOutputReadLine();
    }

    internal static bool HasLaunched(Process? process)
    {
        if (process is null)
            return false;

        try
        {
            _ = process.Id;
            return true;
        }
        catch (InvalidOperationException)
        {
            return false;
        }
    }

    /// <summary>True for llama-server backends that offload to a GPU (i.e. should engage a device).</summary>
    internal static bool IsGpuBackend(LlamaServerBackend backend) => backend switch
    {
        LlamaServerBackend.Cuda12 => true,
        LlamaServerBackend.Cuda13 => true,
        LlamaServerBackend.Vulkan => true,
        LlamaServerBackend.Hip => true,
        LlamaServerBackend.Sycl => true,
        LlamaServerBackend.Metal => true,
        _ => false
    };

    public async ValueTask DisposeAsync()
    {
        if (_disposed)
            return;

        _disposed = true;

        await StopAsync();
        _httpClient.Dispose();
        _guardian.Dispose();
        LlamaServerOwners.Release(_ownerRecord);
        _process?.Dispose();
    }
}
