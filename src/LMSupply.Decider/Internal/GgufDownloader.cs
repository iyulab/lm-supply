using LMSupply.Core.Download;

namespace LMSupply.Decider.Internal;

/// <summary>
/// Downloads the GGUF file a decider loads from HuggingFace — the shared <see cref="SingleFileGgufDownloader"/>.
/// </summary>
internal sealed class GgufDownloader : SingleFileGgufDownloader
{
    // The decider has no earlier versions with a private tree of their own; this name is one no version ever wrote,
    // so the legacy lookup finds nothing and only the shared cache layout is read.
    private const string NoLegacyTree = "gguf-deciders";

    /// <param name="cacheDirectory">Where downloaded GGUF files are kept.</param>
    /// <param name="localFilesOnly">True: serve from the cache only (<c>DisableAutoDownload</c>).</param>
    public GgufDownloader(string cacheDirectory, bool localFilesOnly = false)
        : base(cacheDirectory, NoLegacyTree, localFilesOnly, handler: null)
    {
    }
}
