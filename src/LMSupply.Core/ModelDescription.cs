namespace LMSupply;

/// <summary>
/// What a model id loads on this host, before loading it: the repository or file, the backend that runs it, a name to
/// show and the licence — for a consent screen, a licence inventory, or a log of which model served a request.
/// </summary>
/// <remarks>
/// Answered by the domains that resolve an id in more than one step (<c>LocalReranker.Describe</c>,
/// <c>LocalEmbedder.Describe</c>): a user alias is followed, <c>auto</c> is resolved for this host, and a built-in alias
/// that routes to a GGUF build (served by llama-server) is followed too — the same resolution the load performs. Nothing
/// is downloaded or loaded; the download size is <c>GetDownloadSizeBytesAsync</c>'s question.
/// </remarks>
/// <param name="RequestedId">The id that was asked about, as given.</param>
/// <param name="ResolvedId">The repository id (or local path) the load fetches and opens.</param>
/// <param name="Backend">What runs the model.</param>
/// <param name="DisplayName">A name to show: the catalog's display name, else the repository's or file's own name.</param>
/// <param name="License">
/// The model's licence as curated in the catalog; for a converted or quantized build, the licence of the model it was
/// made from. <see langword="null"/> when the catalog does not know the model — never a guessed default.
/// </param>
/// <param name="CatalogEntry">The registry entry the load uses, or <see langword="null"/> when there is none (a GGUF build, an uncatalogued repository, a path).</param>
public sealed record ModelDescription(
    string RequestedId,
    string ResolvedId,
    ModelBackend Backend,
    string DisplayName,
    string? License,
    IModelInfoBase? CatalogEntry);

/// <summary>What runs a model.</summary>
public enum ModelBackend
{
    /// <summary>ONNX Runtime, in this process.</summary>
    Onnx = 1,

    /// <summary>A GGUF build served by a llama-server process the library manages.</summary>
    Gguf = 2,
}
