namespace LMSupply.Inference;

/// <summary>
/// The set of execution providers that have failed at run time for one model, shared by every
/// <see cref="RecoverableOnnxSession"/> that belongs to that model (for example an encoder and a
/// decoder session loaded from the same checkpoint).
/// </summary>
/// <remarks>
/// A provider that crashes or hangs on one part of a model is not going to behave on the other
/// parts — the failure is a property of the provider/kernel combination, not of one session. Sharing
/// the blacklist means that once the encoder has moved off DirectML, the decoder leaves it too before
/// its next run instead of paying the same crash (or the same 60 s hang) a second time. A session
/// created without an explicit blacklist gets a private one, which is the single-session behaviour.
/// </remarks>
public sealed class ProviderBlacklist
{
    private readonly HashSet<ExecutionProvider> _providers = new();
    private readonly object _lock = new();

    /// <summary>Whether <paramref name="provider"/> has been blacklisted.</summary>
    public bool Contains(ExecutionProvider provider)
    {
        lock (_lock) return _providers.Contains(provider);
    }

    /// <summary>Adds <paramref name="provider"/>; returns false when it was already present.</summary>
    public bool Add(ExecutionProvider provider)
    {
        lock (_lock) return _providers.Add(provider);
    }

    /// <summary>A snapshot of the blacklisted providers, for passing to session creation.</summary>
    public ExecutionProvider[] ToArray()
    {
        lock (_lock) return _providers.ToArray();
    }
}
