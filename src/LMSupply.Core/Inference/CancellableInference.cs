using LMSupply.Exceptions;

namespace LMSupply.Inference;

/// <summary>
/// Runs synchronous, potentially non-cancellable native inference work on the thread pool while
/// guaranteeing that control returns to the caller as soon as the supplied token is cancelled, or
/// once <see cref="DefaultTimeout"/> elapses if it never is.
/// </summary>
/// <remarks>
/// ONNX Runtime inference is a blocking native call. <c>Task.Run(work, token)</c> only checks the
/// token before scheduling, so once the native call starts the token is ignored. Wrapping the
/// resulting task in <c>WaitAsync(token)</c> re-projects the same token onto the await, so the
/// caller is unblocked the moment cancellation is requested — even when the underlying thread
/// remains blocked in native code. Best-effort native termination (freeing that thread) is handled
/// separately via ONNX <c>RunOptions.Terminate</c>, where a caller wires it up.
///
/// A caller that passes <see cref="CancellationToken.None"/> (or any token that is never
/// cancelled) previously had no bound at all: a cold DirectML kernel-init hang would block
/// forever, silently, with no exception and no progress indication. <see cref="DefaultTimeout"/>
/// closes that gap by always applying a bound, derived from the caller's own token when it can
/// time out and falling back to the default otherwise (see docket iyulab/lm-supply, the "DirectML
/// 콜드 커널 행이 임베딩 경로에서 무기한" issue, 2026-09-03).
///
/// Originally introduced in <c>LMSupply.Embedder</c> and promoted here so every ONNX-backed
/// module (Transcriber, Translator, Synthesizer, Segmenter, Ocr, Reranker, ImageGenerator,
/// Detector, Captioner) can share the same bound instead of re-deriving it per module (see docket
/// iyulab/lm-supply, "콜드 GPU 커널 행 방어가 Embedder 하나에만 있음", 2026-09-03).
/// </remarks>
public static class CancellableInference
{
    /// <summary>
    /// Upper bound applied when the caller's own <see cref="CancellationToken"/> cannot time out
    /// on its own (e.g. <see cref="CancellationToken.None"/>). Chosen generously — large enough
    /// that it never fires for a genuinely slow-but-progressing CPU inference, tight enough that a
    /// caller is never left waiting silently forever.
    /// </summary>
    public static readonly TimeSpan DefaultTimeout = TimeSpan.FromSeconds(60);

    /// <summary>
    /// Executes <paramref name="work"/> on the thread pool and returns when it completes, when
    /// <paramref name="cancellationToken"/> is cancelled, or when <paramref name="timeout"/>
    /// (<see cref="DefaultTimeout"/> if omitted) elapses — whichever comes first.
    /// </summary>
    /// <typeparam name="T">Result type produced by the inference delegate.</typeparam>
    /// <param name="work">The synchronous inference delegate to run.</param>
    /// <param name="cancellationToken">Token whose cancellation returns control to the caller.</param>
    /// <param name="timeout">
    /// Overrides <see cref="DefaultTimeout"/> for this call. Applies regardless of whether
    /// <paramref name="cancellationToken"/> can itself time out.
    /// </param>
    /// <returns>The inference result.</returns>
    /// <exception cref="OperationCanceledException">Thrown when <paramref name="cancellationToken"/> is cancelled before completion.</exception>
    /// <exception cref="InferenceTimeoutException">Thrown when the timeout elapses without the caller having cancelled.</exception>
    public static async Task<T> RunAsync<T>(Func<T> work, CancellationToken cancellationToken, TimeSpan? timeout = null)
    {
        var effectiveTimeout = timeout ?? DefaultTimeout;
        using var timeoutCts = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
        timeoutCts.CancelAfter(effectiveTimeout);

        try
        {
            return await Task.Run(work, cancellationToken).WaitAsync(timeoutCts.Token);
        }
        catch (OperationCanceledException) when (!cancellationToken.IsCancellationRequested)
        {
            // timeoutCts fired, and it wasn't because the caller's own token was cancelled --
            // the timeout itself elapsed.
            throw new InferenceTimeoutException(effectiveTimeout);
        }
    }
}
