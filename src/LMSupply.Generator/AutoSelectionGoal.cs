namespace LMSupply.Generator;

/// <summary>
/// What <c>"default"</c>, <c>"auto"</c> and <c>"gguf:auto"</c> optimize for. <see cref="Quality"/> and
/// <see cref="Responsive"/> differ only when no candidate fits the VRAM budget and the model will run on the CPU (or
/// partly offloaded): a candidate that fits VRAM is the largest that fits under both. <see cref="QuickStart"/> also
/// decides among candidates that fit VRAM.
/// </summary>
public enum AutoSelectionGoal
{
    /// <summary>
    /// The largest model the system RAM budget holds. Best answers; on the CPU a larger model generates
    /// proportionally slower.
    /// </summary>
    Quality = 0,

    /// <summary>
    /// The smallest model in the auto-selection pool. For interactive use where a person waits for the answer on
    /// a host whose GPU cannot hold the model.
    /// </summary>
    Responsive = 1,

    /// <summary>
    /// The model with the smallest download among those that fit the VRAM budget — or, when none fits, among those the
    /// system RAM budget holds. For a first run that should start working before it is the best it can be: the file
    /// is the least that fits this host, where <see cref="Quality"/> fetches the largest.
    /// </summary>
    QuickStart = 2,
}
