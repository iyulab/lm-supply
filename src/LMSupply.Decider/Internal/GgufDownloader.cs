using LMSupply.Core.Download;

namespace LMSupply.Decider.Internal;

/// <summary>
/// Downloads the GGUF file a decider loads from HuggingFace — the shared <see cref="SingleFileGgufDownloader"/>.
/// </summary>
internal sealed class GgufDownloader : SingleFileGgufDownloader
{
    /// <param name="cacheDirectory">Where downloaded GGUF files are kept.</param>
    /// <param name="localFilesOnly">True: serve from the cache only (<c>DisableAutoDownload</c>).</param>
    public GgufDownloader(string cacheDirectory, bool localFilesOnly = false)
        : base(cacheDirectory, legacyTreeName: null, localFilesOnly, handler: null)
    {
    }
}
