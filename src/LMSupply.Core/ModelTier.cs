namespace LMSupply;

/// <summary>
/// A rung of a domain's model ladder. The same five names serve every domain — a language model, an embedder, a
/// transcriber — and are ordered by capability and cost: each rung holds a model that is larger or more accurate than the
/// one below it, and a machine may not fit the upper ones. A rung names one distinct model: presets of one model (steps,
/// guidance), voices and language variants are not rungs.
/// </summary>
/// <remarks>
/// A domain fills only the rungs it has a model for; a rung without one is absent, and asking for it fails with the
/// rungs the domain does have (see <see cref="IModelRegistry{TModelInfo}.GetTiers"/>). The rung alias — <c>entry</c>,
/// <c>basic</c>, <c>balanced</c>, <c>advanced</c>, <c>frontier</c> — names no model family or generation, so it keeps
/// working when the catalog moves to a newer model; what it resolves to is the library's choice and a change of it
/// is announced in the release notes. The names are the library's: an application shows its own words for them.
/// </remarks>
public enum ModelTier
{
    /// <summary>The lightest rung: a model that starts quickly and runs on nearly any machine.</summary>
    Entry = 1,

    /// <summary>The plain default size. <c>default</c> names this rung wherever a domain has one.</summary>
    Basic = 2,

    /// <summary>A step up from <see cref="Basic"/> in quality for a step up in memory and time.</summary>
    Balanced = 3,

    /// <summary>The most accurate rung an ordinary workstation or a single consumer GPU runs.</summary>
    Advanced = 4,

    /// <summary>Beyond <see cref="Advanced"/>: a model only a machine with a great deal of memory runs.</summary>
    Frontier = 5,
}

/// <summary>Names of the rungs.</summary>
public static class ModelTierExtensions
{
    /// <summary>The alias a rung is called by: its name, lower case (<c>balanced</c>).</summary>
    public static string Alias(this ModelTier tier) => tier.ToString().ToLowerInvariant();
}

/// <summary>What to do when the rung asked for holds no model in the domain.</summary>
public enum TierFallback
{
    /// <summary>Nothing: an absent rung is an absent rung.</summary>
    None = 0,

    /// <summary>The nearest rung below that holds a model.</summary>
    Lower = 1,

    /// <summary>The nearest rung above that holds a model.</summary>
    Higher = 2,
}

/// <summary>One occupied rung of a domain's ladder.</summary>
/// <param name="Tier">The rung.</param>
/// <param name="Alias">The alias the rung is called by (<c>entry</c>, <c>basic</c>, …).</param>
/// <param name="Model">The model the rung holds.</param>
public sealed record ModelTierEntry<TModelInfo>(ModelTier Tier, string Alias, TModelInfo Model)
    where TModelInfo : IModelInfoBase;
