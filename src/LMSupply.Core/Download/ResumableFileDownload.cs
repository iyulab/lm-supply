using System.Collections.Concurrent;
using System.Globalization;
using System.Net;
using System.Net.Http.Headers;
using LMSupply.Core.Download;
using LMSupply.Exceptions;

namespace LMSupply.Download;

/// <summary>
/// Downloads one file into place through a ".part" it owns, resuming an interrupted transfer with a range
/// request and treating the result as the file only when it is as long as the server announced and, when
/// known, as long as the repository listed. Every downloader in this library that writes a large binary
/// goes through here — the length check, the resume, and the rules for what escapes are made once.
/// </summary>
/// <remarks>
/// <para>
/// <b>Why the ".part" is opened before it is measured.</b> Its length is the resume offset, and a caller
/// that measured it while another was still writing would ask the server for a range it does not hold.
/// With <see cref="FileShare.None"/> the second caller waits in <see cref="FileIoRetry"/> instead and,
/// once it gets in, finds the finished file or an honest offset.
/// </para>
/// <para>
/// <b>Why a short final file is deleted rather than resumed.</b> A final file of an unexpected length has
/// no known provenance — an interrupted copy, another revision, a rename that clobbered a complete file —
/// so it is not a prefix; only a ".part" is. Appending the remainder after it would produce a file that
/// again passes every gate.
/// </para>
/// </remarks>
internal static class ResumableFileDownload
{
    private const int BufferSize = 81920;
    private const int MaxStalls = 3;

    // A resume that keeps making progress may take many attempts on a link that drops every few tens of
    // megabytes; this caps the total so a server that always answers with one short body cannot loop.
    private const int MaxResumeAttempts = 20;

    // One gate per destination path: same-process callers wanting the same file wait for one another
    // instead of racing for the ".part".
    private static readonly ConcurrentDictionary<string, SemaphoreSlim> s_fileGates = new(StringComparer.OrdinalIgnoreCase);

    /// <summary>What one attempt needs to know about the file it is fetching.</summary>
    internal sealed class Request
    {
        /// <summary>The URL to GET.</summary>
        public required string Url { get; init; }

        /// <summary>Where the finished file goes; the ".part" lives beside it.</summary>
        public required string DestinationPath { get; init; }

        /// <summary>Name used in progress reports and messages (the repository path of the file).</summary>
        public required string FileName { get; init; }

        /// <summary>Identifies the model in exceptions.</summary>
        public string? ModelId { get; init; }

        /// <summary>
        /// The length the repository listed, when known. A cached final file of another length is not this
        /// file; a completed body of another length is a stale listing.
        /// </summary>
        public long? ExpectedSize { get; init; }

        /// <summary>
        /// Runs after a successful response header set and before the body is read — the place for a
        /// source-specific check such as "is this a Git LFS pointer instead of the binary".
        /// </summary>
        public Func<HttpResponseMessage, CancellationToken, Task>? InspectResponse { get; init; }

        /// <summary>Whether a failed HTTP status is worth retrying (rate limits, gateway errors).</summary>
        public Func<HttpRequestException, bool>? IsTransient { get; init; }

        /// <summary>Total attempts for transient HTTP failures and timeouts.</summary>
        public int MaxRetries { get; init; } = 3;

        public IProgress<DownloadProgress>? Progress { get; init; }
    }

    /// <summary>
    /// Downloads the file, retrying transient HTTP failures and resuming a body that ended early.
    /// Callers in the same process that want the same file wait here for one another; the one that
    /// arrives second finds the file complete and returns.
    /// </summary>
    public static async Task DownloadAsync(HttpClient httpClient, Request request, CancellationToken cancellationToken)
    {
        var gate = s_fileGates.GetOrAdd(Path.GetFullPath(request.DestinationPath), _ => new SemaphoreSlim(1, 1));
        await gate.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            if (IsCompleteFile(request.DestinationPath, request.ExpectedSize))
                return;

            var tempPath = request.DestinationPath + ".part";
            var lastLength = PartLength(tempPath);
            var stalls = 0;

            for (var attempt = 1; ; attempt++)
            {
                try
                {
                    await AttemptAsync(httpClient, request, cancellationToken).ConfigureAwait(false);
                    return;
                }
                catch (HttpRequestException ex) when (request.IsTransient?.Invoke(ex) == true && attempt < request.MaxRetries)
                {
                    // Exponential backoff below.
                }
                catch (TaskCanceledException) when (!cancellationToken.IsCancellationRequested && attempt < request.MaxRetries)
                {
                    // Timeout, not user cancellation.
                }
                catch (Exception ex) when (ex is IOException or TruncatedDownloadException)
                {
                    // A body that ended early — EOF before the announced length, or a dropped connection —
                    // leaves a valid prefix in the ".part", and so does another process still writing it.
                    // Resuming that prefix is progress, so the budget counts stalls rather than attempts: an
                    // attempt after which the ".part" is no longer than before is a stall, and MaxStalls
                    // stalls in a row (or MaxResumeAttempts attempts overall) give up.
                    var length = PartLength(tempPath);
                    stalls = length > lastLength ? 0 : stalls + 1;
                    lastLength = length;
                    if (stalls >= MaxStalls || attempt >= MaxResumeAttempts)
                        throw;
                }

                var delay = TimeSpan.FromSeconds(Math.Pow(2, Math.Min(attempt, 4)));
                await Task.Delay(delay, cancellationToken).ConfigureAwait(false);
            }
        }
        finally
        {
            gate.Release();
        }
    }

    /// <summary>
    /// A file already in place is this file when it is as long as expected and holds real content. One of
    /// another length is deleted so the download starts over (see the class remarks); when
    /// <paramref name="readOnly"/> it is left alone and simply not counted as cached.
    /// </summary>
    public static bool IsUsableCachedFile(string path, long? expectedSize, bool readOnly = false)
    {
        if (!CacheManager.IsCachedFile(path))
            return false;
        if (expectedSize is not { } expected)
            return true;
        if (new FileInfo(path).Length == expected)
            return true;
        if (!readOnly)
            File.Delete(path);
        return false;
    }

    public static bool IsCompleteFile(string path, long? expectedSize) =>
        expectedSize is { } expected
        && File.Exists(path)
        && new FileInfo(path).Length == expected
        && !CacheManager.IsLfsPointerFile(path);

    private static long PartLength(string tempPath) =>
        File.Exists(tempPath) ? new FileInfo(tempPath).Length : 0;

    /// <summary>
    /// One attempt. Resumes from the ".part" it owns; a body that ends early leaves the ".part" for the
    /// next attempt and throws <see cref="TruncatedDownloadException"/>.
    /// </summary>
    public static async Task AttemptAsync(HttpClient httpClient, Request request, CancellationToken cancellationToken)
    {
        var dir = Path.GetDirectoryName(request.DestinationPath);
        if (!string.IsNullOrEmpty(dir))
            Directory.CreateDirectory(dir);

        var tempPath = request.DestinationPath + ".part";

        // Two callers racing to acquire the same ".part" (a caller bypassing a model pool's lock, or a
        // genuinely concurrent second process) hit this as IOException; FileIoRetry waits it out.
        await using var fileStream = await FileIoRetry.ExecuteAsync(
            () => new FileStream(tempPath, FileMode.OpenOrCreate, FileAccess.ReadWrite, FileShare.None, BufferSize, true),
            cancellationToken).ConfigureAwait(false);

        try
        {
            await DownloadIntoPartAsync(httpClient, request, fileStream, tempPath, cancellationToken).ConfigureAwait(false);
        }
        catch
        {
            // Nothing landed (a 404, a refused request, a body that never started): an empty ".part" is not
            // a resume point, and the directory validator reads any ".part" as an unfinished download.
            if (fileStream.Length == 0)
            {
                fileStream.Close();
                File.Delete(tempPath);
            }
            throw;
        }
    }

    private static async Task DownloadIntoPartAsync(
        HttpClient httpClient, Request request, FileStream fileStream, string tempPath, CancellationToken cancellationToken)
    {
        var destinationPath = request.DestinationPath;
        var expectedSize = request.ExpectedSize;

        // Whoever held the ".part" before us may have finished the download we came for.
        if (IsCompleteFile(destinationPath, expectedSize))
        {
            fileStream.Close();
            File.Delete(tempPath);
            return;
        }

        var startPosition = fileStream.Length;
        if (expectedSize is { } listedLength && startPosition >= listedLength)
        {
            // A ".part" at least as long as the file cannot be a prefix of it.
            fileStream.SetLength(0);
            startPosition = 0;
        }

        using var httpRequest = new HttpRequestMessage(HttpMethod.Get, request.Url);
        if (startPosition > 0)
            httpRequest.Headers.Range = new RangeHeaderValue(startPosition, null);

        using var response = await httpClient.SendAsync(
            httpRequest, HttpCompletionOption.ResponseHeadersRead, cancellationToken).ConfigureAwait(false);

        // 416: the server has nothing past our offset. That is completion only when the listing says we
        // hold the whole file; otherwise the ".part" is not a prefix of this file and the next attempt
        // starts over.
        if (response.StatusCode == HttpStatusCode.RequestedRangeNotSatisfiable)
        {
            if (expectedSize is { } whole && startPosition == whole)
            {
                fileStream.Close();
                await MoveIntoPlaceAsync(tempPath, destinationPath, cancellationToken).ConfigureAwait(false);
                return;
            }

            fileStream.SetLength(0);
            throw new TruncatedDownloadException(
                $"Server has no bytes past offset {startPosition} of '{request.FileName}' (HTTP 416), " +
                "but the file is not known to be complete; restarting the download.",
                request.ModelId);
        }

        if (!response.IsSuccessStatusCode)
        {
            throw new HttpRequestException(
                $"Failed to download '{request.FileName}'. Status: {response.StatusCode}",
                inner: null,
                statusCode: response.StatusCode);
        }

        if (request.InspectResponse is { } inspect)
            await inspect(response, cancellationToken).ConfigureAwait(false);

        // Determine total size
        long totalBytes = response.Content.Headers.ContentLength ?? 0;
        if (response.StatusCode == HttpStatusCode.PartialContent)
        {
            var contentRange = response.Content.Headers.ContentRange;
            if (contentRange?.From is { } from && from != startPosition)
            {
                // The server resumed somewhere else; appending its bytes after ours would interleave two
                // offsets into one file.
                fileStream.SetLength(0);
                throw new TruncatedDownloadException(
                    $"Server resumed '{request.FileName}' at offset {from} while {startPosition} bytes were held; restarting the download.",
                    request.ModelId);
            }

            totalBytes = contentRange?.Length.HasValue == true
                ? contentRange.Length.Value
                : startPosition + (response.Content.Headers.ContentLength ?? 0);
        }
        else
        {
            // Full body (the server ignored the range, or none was sent): write from the start.
            fileStream.SetLength(0);
            startPosition = 0;
        }

        // The length the file must reach: what the server announced, else what the repository listed.
        var expectedTotal = totalBytes > 0 ? totalBytes : expectedSize ?? 0;

        await using var contentStream = await response.Content.ReadAsStreamAsync(cancellationToken).ConfigureAwait(false);
        fileStream.Seek(0, SeekOrigin.End);

        var buffer = new byte[BufferSize];
        long bytesDownloaded = startPosition;
        int bytesRead;

        // One report per read is one per 80 KB — thousands of callbacks for a large model, each of which a
        // UI-bound Progress<T> posts to its thread. Coalesce here, once, to first/last/1%/250 ms.
        var progress = CoalescingProgress.Wrap(request.Progress);

        while ((bytesRead = await contentStream.ReadAsync(buffer, cancellationToken).ConfigureAwait(false)) > 0)
        {
            await fileStream.WriteAsync(buffer.AsMemory(0, bytesRead), cancellationToken).ConfigureAwait(false);
            bytesDownloaded += bytesRead;

            progress?.Report(new DownloadProgress
            {
                FileName = request.FileName,
                BytesDownloaded = bytesDownloaded,
                TotalBytes = expectedTotal
            });
        }

        await fileStream.FlushAsync(cancellationToken).ConfigureAwait(false);
        var length = fileStream.Length;

        // The body ended (end of stream, not an exception) before the announced length. The ".part" is a
        // valid prefix — keep it for the resume — but it is not the file.
        if (expectedTotal > 0 && length != expectedTotal)
        {
            throw new TruncatedDownloadException(
                $"Download of '{request.FileName}' ended after {length} of {expectedTotal} bytes " +
                $"(HTTP {(int)response.StatusCode}, content-length {response.Content.Headers.ContentLength?.ToString(CultureInfo.InvariantCulture) ?? "absent"}, " +
                $"content-range {response.Content.Headers.ContentRange?.ToString() ?? "absent"}, resumed from {startPosition}); " +
                "the partial file is kept for a resume.",
                request.ModelId);
        }

        // The server delivered everything it announced, but that is not the file the repository listed —
        // a stale listing or a revision that moved between the two requests. Not a prefix of anything.
        if (expectedSize is { } expected && length != expected)
        {
            fileStream.Close();
            File.Delete(tempPath);
            throw new ModelDownloadException(
                $"'{request.FileName}' is {length} bytes but the repository listing says {expected}; the listing may be stale.",
                request.ModelId);
        }

        fileStream.Close();
        await MoveIntoPlaceAsync(tempPath, destinationPath, cancellationToken).ConfigureAwait(false);
    }

    // Move to final location atomically. Retry: the destination path may be transiently held open by
    // another process/AV scanner immediately after this rename.
    private static Task MoveIntoPlaceAsync(string tempPath, string destinationPath, CancellationToken cancellationToken) =>
        FileIoRetry.ExecuteAsync(
            () => File.Move(tempPath, destinationPath, overwrite: true),
            cancellationToken);
}
