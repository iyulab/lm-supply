namespace LMSupply.Generator.Internal.Llama;

/// <summary>
/// What a GGUF load was asked for, and the name that describes the file it actually opened.
/// </summary>
/// <param name="ModelId">Display name of the loaded file. It names the loaded quantization when the load took another file than the alias's default.</param>
/// <param name="RequestedModelId">The id the caller passed: a registry alias, a HuggingFace repo id, or a path.</param>
/// <param name="RequestedFile">The requested alias's default file, or <see langword="null"/> when no registry alias was requested.</param>
/// <param name="KnownIssues">The requested alias's registered known issues.</param>
internal sealed record GgufLoadIdentity(
    string ModelId,
    string RequestedModelId,
    string? RequestedFile,
    IReadOnlyList<string> KnownIssues)
{
    /// <summary>Identity of a load that opened <paramref name="modelPath"/> for <paramref name="requestedModelId"/>.</summary>
    public static GgufLoadIdentity Describe(string requestedModelId, GgufModelInfo? registryInfo, string modelPath)
    {
        if (registryInfo is null)
            return new GgufLoadIdentity(requestedModelId, requestedModelId, null, []);

        var loadedFile = Path.GetFileName(modelPath);
        var displayName = string.Equals(loadedFile, registryInfo.DefaultFile, StringComparison.OrdinalIgnoreCase)
            ? registryInfo.DisplayName
            : DescribeSubstitute(registryInfo, loadedFile);

        return new GgufLoadIdentity(displayName, requestedModelId, registryInfo.DefaultFile, registryInfo.KnownIssues);
    }

    // "Gemma 4 E4B Instruct (Q8_0)" loaded as Q4_0 → "Gemma 4 E4B Instruct (Q4_0)". A display name that ends in
    // its default's quantization drops it; any other parenthetical ("(MoE)") stays.
    private static string DescribeSubstitute(GgufModelInfo registryInfo, string loadedFile)
    {
        var baseName = registryInfo.DisplayName;
        var defaultQuant = GgufQuantizationLabel.FromFileName(registryInfo.DefaultFile);
        if (defaultQuant is not null)
        {
            var suffix = $" ({defaultQuant})";
            if (baseName.EndsWith(suffix, StringComparison.OrdinalIgnoreCase))
                baseName = baseName[..^suffix.Length];
        }

        var loadedQuant = GgufQuantizationLabel.FromFileName(loadedFile);
        return $"{baseName} ({loadedQuant ?? Path.GetFileNameWithoutExtension(loadedFile)})";
    }
}
