using System.Diagnostics;
using System.Runtime.ExceptionServices;
using LMSupply.Exceptions;
using Microsoft.ML.OnnxRuntime;

namespace LMSupply.Inference;

/// <summary>
/// An ONNX <see cref="InferenceSession"/> that recovers from a failing execution provider at run
/// time by moving to the next provider in the Auto fallback chain, for both failure shapes a GPU
/// provider produces: a run-time crash (<see cref="OnnxRuntimeException"/>, e.g. an unsupported
/// kernel) and a hang that only the caller-side bound in <see cref="CancellableInference"/> can
/// detect (<see cref="InferenceTimeoutException"/>).
/// </summary>
/// <remarks>
/// <para>
/// The session and the gate that serializes access to it travel together as one immutable handle
/// that is swapped atomically. This matters for the timeout case: the timed-out native run may
/// still be blocked inside <c>Run</c> holding the old gate, so the old session is never disposed
/// or reused — it is abandoned, a fresh session/gate pair is published, and a thread that later
/// acquires the old gate notices the handle changed and re-dispatches onto the current one.
/// </para>
/// <para>
/// The blacklist of providers that have failed is a <see cref="ProviderBlacklist"/> that several
/// sessions of one model can share (encoder + decoder, detector + recognizer). A session whose
/// current provider was blacklisted by a sibling leaves that provider before its next run instead
/// of hitting the same crash or hang itself.
/// </para>
/// <para>
/// Originally implemented inside <c>LMSupply.Embedder</c>'s inference engine and promoted here so
/// every ONNX-backed module (Reranker, Transcriber, Translator, Synthesizer, Segmenter, Ocr,
/// ImageGenerator, Detector, Captioner, …) shares one recovery path instead of each holding a
/// bare <c>InferenceSession</c> that fails outright on the first GPU crash or hang.
/// </para>
/// <para>
/// Lock order, for anyone extending this class: a handle's <c>Gate</c> is always acquired before
/// <c>_recoveryGate</c>, never after it while blocking. <see cref="Dispose"/> only ever polls a gate
/// (<c>Wait(0)</c>) under <c>_recoveryGate</c>.
/// </para>
/// </remarks>
public sealed class RecoverableOnnxSession : IDisposable
{
    private sealed class SessionHandle
    {
        public SessionHandle(InferenceSession session, IReadOnlyList<string> activeProviders, bool isGpuActive)
        {
            Session = session;
            ActiveProviders = activeProviders;
            IsGpuActive = isGpuActive;
        }

        public InferenceSession Session { get; }
        public SemaphoreSlim Gate { get; } = new(1, 1);
        public IReadOnlyList<string> ActiveProviders { get; }
        public bool IsGpuActive { get; }
    }

    private volatile SessionHandle _handle;
    private readonly ExecutionProvider _requestedProvider;
    private readonly string _modelPath;
    private readonly Action<SessionOptions>? _configureOptions;
    private readonly string _logPrefix;
    private readonly ProviderBlacklist _blacklist;
    private readonly int _deviceId;
    // Providers this session itself has already tried to recover from — a second failure on the
    // same provider gives up instead of paying for another (doomed) session creation.
    private readonly HashSet<ExecutionProvider> _attemptedProviders = new();
    private readonly List<SessionHandle> _abandonedHandles = new();
    private readonly object _recoveryGate = new();
    private bool _disposed;

    /// <summary>
    /// Wraps an already-created session.
    /// </summary>
    /// <param name="session">The live session (may be null only in tests that never run inference).</param>
    /// <param name="activeProviders">Execution providers active on <paramref name="session"/>, primary first.</param>
    /// <param name="isGpuActive">Whether a GPU provider is active on <paramref name="session"/>.</param>
    /// <param name="requestedProvider">The provider the caller asked for; <see cref="ExecutionProvider.Cpu"/> disables recovery.</param>
    /// <param name="modelPath">Model file, needed to create a replacement session on the next provider.</param>
    /// <param name="configureOptions">Session options to reapply to a replacement session.</param>
    /// <param name="logPrefix">Prefix for the <see cref="Trace"/> diagnostics this class emits.</param>
    /// <param name="blacklist">
    /// Blacklist shared with the other sessions of the same model, so a provider that failed on one
    /// of them is left by all of them. Omit for a single-session model.
    /// </param>
    /// <param name="deviceId">GPU device index the session was created for; a replacement session targets the same device.</param>
    public RecoverableOnnxSession(
        InferenceSession session,
        IReadOnlyList<string> activeProviders,
        bool isGpuActive,
        ExecutionProvider requestedProvider,
        string modelPath,
        Action<SessionOptions>? configureOptions = null,
        string logPrefix = "[RecoverableOnnxSession]",
        ProviderBlacklist? blacklist = null,
        int deviceId = 0)
    {
        _handle = new SessionHandle(session, activeProviders, isGpuActive);
        _requestedProvider = requestedProvider;
        _modelPath = modelPath;
        _configureOptions = configureOptions;
        _logPrefix = logPrefix;
        _blacklist = blacklist ?? new ProviderBlacklist();
        _deviceId = deviceId;
    }

    /// <summary>
    /// Wraps the session produced by <see cref="OnnxSessionFactory"/>'s <c>CreateWithInfoAsync</c>.
    /// Providers the load-time chain already saw fail for this model
    /// (<see cref="SessionCreationResult.FailedProviders"/>) are put on the blacklist up front, so a
    /// sibling session sharing it does not try them again.
    /// </summary>
    public static RecoverableOnnxSession FromResult(
        SessionCreationResult result,
        string modelPath,
        Action<SessionOptions>? configureOptions = null,
        string logPrefix = "[RecoverableOnnxSession]",
        ProviderBlacklist? blacklist = null)
    {
        var session = new RecoverableOnnxSession(
            result.Session, result.ActiveProviders, result.IsGpuActive, result.RequestedProvider, modelPath, configureOptions, logPrefix, blacklist, result.DeviceId);
        foreach (var failed in result.FailedProviders)
            session._blacklist.Add(failed);
        return session;
    }

    /// <summary>Execution providers active on the current session, primary first.</summary>
    public IReadOnlyList<string> ActiveProviders => _handle.ActiveProviders;

    /// <summary>Whether a GPU provider is active on the current session.</summary>
    public bool IsGpuActive => _handle.IsGpuActive;

    /// <summary>The provider the caller asked for.</summary>
    public ExecutionProvider RequestedProvider => _requestedProvider;

    /// <summary>The blacklist this session consults and contributes to (shared or private).</summary>
    public ProviderBlacklist Blacklist => _blacklist;

    /// <summary>
    /// The current session, for diagnostics and model metadata only (input/output names and shapes
    /// are the same on every provider). Do not run inference on it directly — use
    /// <see cref="Run{T}"/> so the gate, cancellation translation and provider fallback apply.
    /// </summary>
    public InferenceSession Session => _handle.Session;

    /// <summary>
    /// Runs <paramref name="work"/> against the current session under its gate. A cancelled token
    /// asks the native run to terminate cooperatively via <see cref="RunOptions.Terminate"/> and
    /// surfaces as <see cref="OperationCanceledException"/>. An <see cref="OnnxRuntimeException"/>
    /// on a non-CPU provider triggers one fallback to the next provider and exactly one retry.
    /// </summary>
    /// <param name="work">Receives the session and per-run options; must call <c>session.Run(..., runOptions)</c>.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    public T Run<T>(Func<InferenceSession, RunOptions, T> work, CancellationToken cancellationToken = default)
        => RunInternal(work, allowRetry: true, cancellationToken);

    /// <summary>
    /// Runs <paramref name="work"/> under <see cref="CancellableInference"/>'s bound as well as
    /// <see cref="Run{T}"/>'s crash fallback. If the bound elapses while a GPU provider is active,
    /// the session moves to the next provider (<see cref="TryRecoverAfterTimeout"/>) and the work
    /// is retried exactly once; a timeout on CPU, or a second timeout, surfaces unchanged.
    /// </summary>
    public async Task<T> RunWithRecoveryAsync<T>(
        Func<InferenceSession, RunOptions, T> work,
        TimeSpan? timeout = null,
        CancellationToken cancellationToken = default)
    {
        try
        {
            return await CancellableInference.RunAsync(() => Run(work, cancellationToken), cancellationToken, timeout);
        }
        catch (InferenceTimeoutException ex) when (TryRecoverAfterTimeout(ex))
        {
            return await CancellableInference.RunAsync(() => Run(work, cancellationToken), cancellationToken, timeout);
        }
    }

    private T RunInternal<T>(Func<InferenceSession, RunOptions, T> work, bool allowRetry, CancellationToken cancellationToken)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        var handle = AcquireCurrentHandle(cancellationToken);
        OnnxRuntimeException? crash;
        try
        {
            // Terminate is checked between operators; it cannot preempt a hang inside a single
            // kernel (e.g. cold DirectML init) — that is what RunWithRecoveryAsync's bound is for.
            using var runOptions = new RunOptions();
            using var ctRegistration = cancellationToken.CanBeCanceled
                ? cancellationToken.Register(static state => ((RunOptions)state!).Terminate = true, runOptions)
                : default;
            try
            {
                return work(handle.Session, runOptions);
            }
            catch (OnnxRuntimeException) when (cancellationToken.IsCancellationRequested)
            {
                // RunOptions.Terminate surfaces as an OnnxRuntimeException; translate it so callers
                // see a cancellation, not a provider crash, and so fallback is not entered.
                throw new OperationCanceledException(cancellationToken);
            }
            catch (OnnxRuntimeException ex) when (allowRetry && _requestedProvider != ExecutionProvider.Cpu)
            {
                crash = ex;
            }
        }
        finally
        {
            handle.Gate.Release();
        }

        // Only a recoverable crash gets here, and only after this thread has released the gate — so
        // TryFallback can reclaim the crashed handle's gate to dispose it safely.
        if (TryFallback(crash, handle))
        {
            // Retry exactly once on the replacement session (allowRetry: false prevents
            // runaway recursion if the next provider also fails on the same input).
            return RunInternal(work, allowRetry: false, cancellationToken);
        }

        ExceptionDispatchInfo.Throw(crash);
        return default!; // unreachable — ExceptionDispatchInfo.Throw does not return
    }

    /// <summary>
    /// Acquires the gate of the current handle. If the handle was replaced while this thread waited
    /// on the old gate, re-dispatches onto the current one rather than running on the abandoned
    /// session. If a sibling session blacklisted the provider this handle is on, moves off it first.
    /// </summary>
    private SessionHandle AcquireCurrentHandle(CancellationToken cancellationToken)
    {
        while (true)
        {
            var handle = _handle;
            handle.Gate.Wait(cancellationToken);
            if (!ReferenceEquals(handle, _handle))
            {
                handle.Gate.Release();
                continue;
            }

            if (TryLeaveBlacklistedProvider(handle))
            {
                handle.Gate.Release();
                continue;
            }

            return handle;
        }
    }

    /// <summary>
    /// With <paramref name="handle"/>'s gate held: if its provider is on the shared blacklist (a
    /// sibling session failed on it), swaps in a replacement on the next provider and disposes the
    /// old session. Holding the gate is what makes the dispose safe — nobody else is inside a run on
    /// it, and anyone waiting on it re-dispatches once it sees the handle change.
    /// </summary>
    private bool TryLeaveBlacklistedProvider(SessionHandle handle)
    {
        if (_requestedProvider == ExecutionProvider.Cpu)
            return false;

        var provider = MapActiveToProvider(handle.ActiveProviders);
        if (provider is null or ExecutionProvider.Cpu || !_blacklist.Contains(provider.Value))
            return false;

        lock (_recoveryGate)
        {
            if (!ReferenceEquals(handle, _handle))
                return true;   // someone else already moved on — re-dispatch

            var replacement = TryCreateReplacementHandle(
                provider.Value,
                $"{provider} was blacklisted by another session of this model",
                originalMessage: null);
            if (replacement is null)
                return false;  // nothing better available — run where we are

            handle.Session?.Dispose();
            _handle = replacement;
            return true;
        }
    }

    /// <summary>
    /// After a run on the current provider threw, blacklists that provider and recreates the
    /// session on the next one in the Auto chain. Returns false when nothing can be done.
    /// </summary>
    /// <remarks>
    /// Safe to call from outside <see cref="Run{T}"/> (e.g. a module that catches the crash itself):
    /// the old session is disposed only if no run is in flight on it, and abandoned to
    /// <see cref="Dispose"/> otherwise.
    /// </remarks>
    public bool TryFallback(OnnxRuntimeException ex) => TryFallback(ex, _handle);

    private bool TryFallback(OnnxRuntimeException ex, SessionHandle failed)
    {
        // Lock order: gate first, then _recoveryGate (see class remarks). Wait(0) rather than Wait():
        // another thread may legitimately be mid-run on this handle (or hung on it), and blocking
        // behind it here would stall the caller's retry for no benefit — abandoning is just as safe.
        var gateHeld = failed.Gate.Wait(0);
        try
        {
            lock (_recoveryGate)
            {
                if (!ReferenceEquals(failed, _handle))
                    return true;   // a concurrent fallback already replaced it — retry on the current handle

                var failedProvider = MapActiveToProvider(failed.ActiveProviders);
                if (failedProvider is null)
                {
                    Trace.TraceWarning($"{_logPrefix} Inference failed but active provider could not be identified; not attempting fallback.");
                    return false;
                }

                var replacement = TryCreateReplacementHandle(
                    failedProvider.Value,
                    $"Inference failed on {failedProvider} ({ex.GetType().Name})",
                    ex.Message);
                if (replacement is null)
                    return false;

                if (gateHeld)
                    failed.Session?.Dispose();
                else
                    _abandonedHandles.Add(failed);

                _handle = replacement;
                return true;
            }
        }
        finally
        {
            if (gateHeld)
                failed.Gate.Release();
        }
    }

    /// <summary>
    /// After a run exceeded the caller-side bound, moves to the next provider without touching the
    /// old session (its native run may still be blocked holding the old gate — it is abandoned and
    /// reclaimed by <see cref="Dispose"/> only once its gate is free). Returns true when the caller
    /// should retry once; false when CPU was requested, CPU is already active, the provider was
    /// already tried, or session recreation failed.
    /// </summary>
    public bool TryRecoverAfterTimeout(InferenceTimeoutException ex)
    {
        if (_requestedProvider == ExecutionProvider.Cpu)
            return false;

        lock (_recoveryGate)
        {
            var stale = _handle;
            var failedProvider = MapActiveToProvider(stale.ActiveProviders);
            if (failedProvider is null or ExecutionProvider.Cpu)
                return false;   // nothing below CPU to fall back to

            var replacement = TryCreateReplacementHandle(
                failedProvider.Value,
                $"Inference timed out on {failedProvider} after {ex.Timeout.TotalSeconds:F0}s (likely a cold GPU kernel initialization hang)",
                ex.Message);
            if (replacement is null)
                return false;

            _abandonedHandles.Add(stale);
            _handle = replacement;
            return true;
        }
    }

    // Caller must hold _recoveryGate.
    private SessionHandle? TryCreateReplacementHandle(ExecutionProvider failedProvider, string reason, string? originalMessage)
    {
        if (!_attemptedProviders.Add(failedProvider))
            return null;   // this session already tried to recover from this provider — give up

        _blacklist.Add(failedProvider);

        var detail = originalMessage is null ? "" : $" Original message: {Truncate(originalMessage, 200)}";
        Trace.TraceWarning($"{_logPrefix} {reason}. Attempting fallback to next provider.{detail}");

        try
        {
            var skip = _blacklist.ToArray();
            var result = OnnxSessionFactory.CreateWithInfoAsync(
                _modelPath,
                ExecutionProvider.Auto,
                skip,
                _configureOptions,
                deviceId: _deviceId).GetAwaiter().GetResult();

            var newProvider = MapActiveToProvider(result.ActiveProviders);
            if (newProvider is null || _blacklist.Contains(newProvider.Value))
            {
                Trace.TraceWarning($"{_logPrefix} Fallback produced no usable alternative provider (blacklist=[{string.Join(",", skip)}]). Surfacing original exception.");
                result.Session.Dispose();
                return null;
            }

            Trace.TraceWarning($"{_logPrefix} Recovered: now running on {string.Join("+", result.ActiveProviders)}.");
            return new SessionHandle(result.Session, result.ActiveProviders, result.IsGpuActive);
        }
        catch (Exception fallbackEx)
        {
            // Actionable: the usual cause is a model variant that was selected for the GPU provider
            // and that the CPU provider cannot load, so tell the caller what to change.
            Trace.TraceError(
                $"{_logPrefix} Fallback session creation for '{Path.GetFileName(_modelPath)}' failed: {Truncate(fallbackEx.Message, 200)} " +
                "The original exception will surface. If this repeats, load the model with Provider = ExecutionProvider.Cpu, or " +
                "choose a QuantizationHint (e.g. \"fp32\") whose files the CPU provider can load.");
            return null;
        }
    }

    private static ExecutionProvider? MapActiveToProvider(IReadOnlyList<string> activeProviders)
    {
        foreach (var p in activeProviders)
        {
            if (p.Contains("CUDA", StringComparison.OrdinalIgnoreCase))
                return ExecutionProvider.Cuda;
            if (p.Contains("DML", StringComparison.OrdinalIgnoreCase) || p.Contains("DirectML", StringComparison.OrdinalIgnoreCase))
                return ExecutionProvider.DirectML;
            if (p.Contains("CoreML", StringComparison.OrdinalIgnoreCase))
                return ExecutionProvider.CoreML;
        }
        if (activeProviders.Any(p => p.Contains("CPU", StringComparison.OrdinalIgnoreCase)))
            return ExecutionProvider.Cpu;
        return null;
    }

    private static string Truncate(string s, int max) => s.Length <= max ? s : s[..max] + "...";

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;

        lock (_recoveryGate)
        {
            // The current handle needs the same care as an abandoned one, and for a reason that is
            // easy to miss: a run only becomes "abandoned" when TryRecoverAfterTimeout actually
            // moves it aside, and that returns false whenever there is nothing to fall back to --
            // CPU was requested, CPU is already active, or replacement creation failed. In those
            // cases the timed-out run is still inside native code on the *current* handle, the
            // caller has already been handed an InferenceTimeoutException, and teardown follows.
            // Disposing that session is an access violation, not an exception: it takes the whole
            // process down with no managed stack to catch. See docket iyulab/lm-supply#193.
            ReclaimOrLeak(_handle);

            foreach (var abandoned in _abandonedHandles)
            {
                ReclaimOrLeak(abandoned);
            }
            _abandonedHandles.Clear();
        }
    }

    /// <summary>
    /// Disposes a handle only if no run is in flight on it; otherwise leaks it deliberately.
    /// Holding the gate is the proof that the native call has returned -- <see cref="RunInternal"/>
    /// releases it in a <c>finally</c>, so a gate that cannot be taken means a call that never came
    /// back. Polls (<c>Wait(0)</c>) rather than blocking, preserving this class's lock order.
    /// </summary>
    private void ReclaimOrLeak(SessionHandle handle)
    {
        if (handle.Gate.Wait(0))
        {
            handle.Session?.Dispose();
            handle.Gate.Release();
            handle.Gate.Dispose();
        }
        else
        {
            Trace.TraceWarning($"{_logPrefix} Leaking a session on {string.Join("+", handle.ActiveProviders)}: its native run has not returned, so it cannot be disposed safely.");
        }
    }
}
