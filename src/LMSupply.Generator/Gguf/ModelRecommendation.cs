namespace LMSupply.Generator.Gguf;

/// <summary>
/// One rung of <see cref="ModelRecommendation.Models"/>: a catalog model judged against this machine. A stable contract an
/// application may show people — its name, its tier, whether it fits and what it holds in memory — in place of a hand-written
/// table that goes stale when the catalog moves an alias.
/// </summary>
public sealed record RecommendedModel
{
    /// <summary>The catalog alias that loads this model (for example <c>gguf:qwen3-balanced</c>).</summary>
    public required string Alias { get; init; }

    /// <summary>The rung of the ladder, in the library's words.</summary>
    public required ModelTier Tier { get; init; }

    /// <summary>
    /// A name for people: the catalog's display name without a trailing parenthetical (the quantization or a note such as
    /// "MoE"). The quantization is <see cref="GgufModelInfo.QuantizationType"/> of <see cref="Model"/>.
    /// </summary>
    public required string DisplayName { get; init; }

    /// <summary>
    /// The model judged with the rule automatic selection and the loader share: <see cref="ModelSelectionCandidate.Fits"/> = runs
    /// fully on the GPU, <see cref="ModelSelectionCandidate.FitsInSystemRam"/> without it = runs from system memory (slower),
    /// neither = too large for this machine. The byte figures are estimates at <see cref="ModelRecommendation.ContextLength"/>.
    /// </summary>
    public required ModelSelectionCandidate Judgement { get; init; }

    /// <summary>The catalog entry: repository, file, quantization, licence, known issues.</summary>
    public GgufModelInfo Model => Judgement.Model;

    /// <summary>Whether the model runs on this machine at all (on the GPU, or from system memory).</summary>
    public bool IsUsable => Judgement.Fits || Judgement.FitsInSystemRam;

    /// <summary>The model <c>gguf:auto</c> picks for the same inputs — the list and the automatic pick never disagree.</summary>
    public bool IsAutomaticPick { get; init; }

    /// <summary>The smallest rung that runs on this machine.</summary>
    public bool IsSmallestUsable { get; init; }

    /// <summary>
    /// The largest rung that runs fully on the GPU; when none does, the largest that runs from system memory. On a small
    /// machine this is the same rung as <see cref="IsSmallestUsable"/>; on a large one it reaches the upper tiers.
    /// </summary>
    public bool IsBestThatRunsWell { get; init; }
}

/// <summary>
/// The catalog's recommendation ladder judged against one machine, from <see cref="GgufModelRegistry.Recommend(ExecutionProvider, int?, AutoSelectionGoal, IReadOnlyCollection{string}?)"/>.
/// </summary>
public sealed record ModelRecommendation
{
    /// <summary>Every rung, smallest first by the memory it holds at <see cref="ContextLength"/>. Rungs that do not fit stay in the list, marked by their <see cref="RecommendedModel.Judgement"/>.</summary>
    public required IReadOnlyList<RecommendedModel> Models { get; init; }

    /// <summary>The VRAM budget the judgement used, in bytes; 0 = no GPU.</summary>
    public required long AvailableVramBytes { get; init; }

    /// <summary>The system memory budget the judgement used, in bytes, after the reserve; 0 = no RAM path.</summary>
    public required long AvailableSystemRamBytes { get; init; }

    /// <summary>The context length each model's KV cache was sized for.</summary>
    public required int ContextLength { get; init; }

    /// <summary>What the automatic pick optimized for.</summary>
    public AutoSelectionGoal Goal { get; init; }

    /// <summary>The catalog's revision (<see cref="GeneratorCatalog.Revision"/>) this list was made from.</summary>
    public required string CatalogRevision { get; init; }
}
