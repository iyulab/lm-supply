using LMSupply.Core.Download;

namespace LMSupply.Reranker.Utils;

/// <summary>
/// Downloads GGUF reranker model files from HuggingFace — the shared <see cref="SingleFileGgufDownloader"/> over the
/// private tree earlier reranker versions wrote.
/// </summary>
internal sealed class GgufDownloader : SingleFileGgufDownloader
{
    // Where earlier versions kept downloaded files (still read, never written).
    internal const string LegacyTreeName = "gguf-rerankers";

    /// <param name="cacheDirectory">Where downloaded GGUF files are kept.</param>
    /// <param name="localFilesOnly">True: serve from the cache only (<c>DisableAutoDownload</c>).</param>
    public GgufDownloader(string cacheDirectory, bool localFilesOnly = false)
        : base(cacheDirectory, LegacyTreeName, localFilesOnly, handler: null)
    {
    }

    /// <summary>The cached GGUF file an offline load opens, or null (see <see cref="SingleFileGgufDownloader"/>).</summary>
    internal static string? TrySelectFromLocalCache(string cacheDirectory, string repoId, string? preferredQuantization)
        => TrySelectFromLocalCache(cacheDirectory, repoId, preferredQuantization, LegacyTreeName);
}
