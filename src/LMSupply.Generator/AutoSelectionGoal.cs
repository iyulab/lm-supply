namespace LMSupply.Generator;

/// <summary>
/// What <c>"default"</c>, <c>"auto"</c> and <c>"gguf:auto"</c> optimize for when no candidate fits the VRAM budget
/// and the model will run on the CPU (or partly offloaded). A candidate that fits VRAM is chosen the same way under
/// every goal: the largest that fits.
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
}
