namespace LMSupply.Generator.Gguf;

/// <summary>
/// A rung of the catalog's recommendation ladder (<see cref="GgufModelRegistry.Recommend(ExecutionProvider, int?, AutoSelectionGoal, IReadOnlyCollection{string}?)"/>),
/// named the way the tier aliases are (<c>gguf:qwen3-fast</c>, <c>-default</c>, <c>-balanced</c>, <c>-quality</c>,
/// <c>gguf:xlarge</c>). The names are the library's: an application shows its own words for them. The order is the order
/// of size — each rung holds more than the one before it, and a machine may not fit the upper ones.
/// </summary>
public enum ModelTier
{
    /// <summary>The smallest rung: a model that starts quickly and runs on most machines.</summary>
    Fast = 1,

    /// <summary>The plain default size.</summary>
    Default = 2,

    /// <summary>A step up from <see cref="Default"/> in quality for a step up in memory.</summary>
    Balanced = 3,

    /// <summary>The largest rung the automatic selection draws from.</summary>
    Quality = 4,

    /// <summary>Beyond <see cref="Quality"/>: a model only a machine with a great deal of memory runs.</summary>
    ExtraLarge = 5,
}
