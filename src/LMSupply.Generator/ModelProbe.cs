using LMSupply.Generator.Gguf;

namespace LMSupply.Generator;

/// <summary>What <see cref="LocalGenerator.DescribeAsync(string, GeneratorOptions?, CancellationToken)"/> found for a model id.</summary>
public enum ModelProbeStatus
{
    /// <summary>The model exists and has files this library can load; the rest of the <see cref="ModelProbe"/> is filled in.</summary>
    Found = 1,

    /// <summary>
    /// No such repository. The Hub answers an anonymous request for a repository that does not exist with the same
    /// refusal it gives a private or gated one, so this also covers a repository the caller may not read — set
    /// <c>HF_TOKEN</c> for those.
    /// </summary>
    NotFound = 2,

    /// <summary>The repository exists but holds nothing this library loads: no GGUF file and no ONNX model.</summary>
    NoLoadableFiles = 3,

    /// <summary>
    /// The answer could not be checked: the Hub could not be reached or answered with an error, or downloads are disabled
    /// and the cache does not hold what was asked. Not "does not exist" — ask again when the cause is gone.
    /// </summary>
    Unreachable = 4,
}

/// <summary>
/// A model id looked at before anything is downloaded: what it is, how large its download is, and whether it runs on this
/// machine — the same facts a catalog entry shows, for an id the catalog does not list (a repository a person typed), and a
/// clear answer for an id that is not usable. From <see cref="LocalGenerator.DescribeAsync(string, GeneratorOptions?, CancellationToken)"/>.
/// </summary>
public sealed record ModelProbe
{
    /// <summary>The id that was asked about.</summary>
    public required string RequestedId { get; init; }

    /// <summary>The outcome. Everything below is meaningful only for <see cref="ModelProbeStatus.Found"/>, except <see cref="Description"/> for <see cref="ModelProbeStatus.NoLoadableFiles"/>.</summary>
    public required ModelProbeStatus Status { get; init; }

    /// <summary>
    /// The model's name, backend (GGUF or ONNX), licence and catalog entry (<see cref="ModelDescription"/>). The name is the
    /// catalog's display name, else the repository's own name without a <c>-GGUF</c> suffix; the licence is the catalog's, null
    /// when the catalog does not know the model.
    /// </summary>
    public ModelDescription? Description { get; init; }

    /// <summary>
    /// Bytes a load with the same options downloads into an empty cache — what <see cref="LocalGenerator.GetDownloadSizeBytesAsync"/>
    /// answers; 0 for a model on local disk.
    /// </summary>
    public long DownloadBytes { get; init; }

    /// <summary>
    /// Whether the model runs on this machine, judged with the rule automatic selection and the loader share
    /// (<see cref="GgufModelRegistry.Evaluate(GgufModelInfo, ExecutionProvider, int?)"/>): the catalog's own file for a catalog entry, the
    /// file the load would pick for another repository. Null for an ONNX model — its load does not use that rule, and a verdict from
    /// a different rule is not offered as the same one.
    /// </summary>
    public ModelSelectionCandidate? Judgement { get; init; }

    /// <summary>For a catalog entry, its known-issue tags (<see cref="GgufModelKnownIssues"/>); empty otherwise.</summary>
    public IReadOnlyList<string> KnownIssues { get; init; } = [];

    /// <summary>Why the status is not <see cref="ModelProbeStatus.Found"/>, in the library's words; null when it is.</summary>
    public string? Detail { get; init; }
}
