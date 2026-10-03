using LMSupply.Core.Download;

namespace LMSupply.Embedder.Utils;

/// <summary>
/// Downloads GGUF embedding model files from HuggingFace — the shared <see cref="SingleFileGgufDownloader"/> over the
/// private tree earlier embedder versions wrote.
/// </summary>
internal sealed class GgufDownloader : SingleFileGgufDownloader
{
    // Where earlier versions kept downloaded files (still read, never written).
    internal const string LegacyTreeName = "gguf-embeddings";

    /// <param name="cacheDirectory">Where downloaded GGUF files are kept.</param>
    /// <param name="localFilesOnly">True: serve from the cache only (<c>DisableAutoDownload</c>).</param>
    public GgufDownloader(string cacheDirectory, bool localFilesOnly = false)
        : base(cacheDirectory, LegacyTreeName, localFilesOnly, handler: null)
    {
    }

    /// <summary>Test seam: the same downloader over a caller-supplied transport, listing included.</summary>
    internal GgufDownloader(string cacheDirectory, bool localFilesOnly, HttpMessageHandler? handler)
        : base(cacheDirectory, LegacyTreeName, localFilesOnly, handler)
    {
    }
}
