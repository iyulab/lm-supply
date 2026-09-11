namespace LMSupply.Ocr;

/// <summary>
/// Whether the model files <see cref="LocalOcr.LoadForLanguageAsync"/> needs for one language are in the
/// local cache — the answer to "would loading this language download anything?".
/// </summary>
/// <param name="LanguageCode">The language code that was asked about.</param>
/// <param name="RecognitionModel">The recognition model the language resolves to (e.g., "crnn-korean-v3").</param>
/// <param name="Files">Every file the load needs: the detection model, then the recognizer and its dictionary.</param>
public sealed record OcrCacheStatus(string LanguageCode, string RecognitionModel, IReadOnlyList<OcrModelFile> Files)
{
    /// <summary>
    /// True when every file is cached, so loading the language makes no download.
    /// </summary>
    public bool IsCached => Files.All(f => f.IsCached);

    /// <summary>
    /// The files a load would download.
    /// </summary>
    public IReadOnlyList<OcrModelFile> Missing => [.. Files.Where(f => !f.IsCached)];
}

/// <summary>
/// One model file an OCR load needs, and whether it is in the local cache.
/// </summary>
/// <param name="RepoId">The HuggingFace repository the file comes from.</param>
/// <param name="Subfolder">The repository subfolder that holds it, if any (e.g., "languages/korean").</param>
/// <param name="FileName">The file name within that subfolder.</param>
/// <param name="IsCached">Whether the file is in the local cache.</param>
public sealed record OcrModelFile(string RepoId, string? Subfolder, string FileName, bool IsCached);
