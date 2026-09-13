using LMSupply.Exceptions;

namespace LMSupply.Download;

/// <summary>
/// A download attempt whose body ended before the announced length, or whose resume offset the server
/// would not honour. The ".part" file holds a valid prefix (or has been emptied), so the attempt is
/// retried — resumed — rather than reported; it surfaces to callers only when the retries stall.
/// </summary>
internal sealed class TruncatedDownloadException(string message, string? modelId = null)
    : ModelDownloadException(message, modelId);
