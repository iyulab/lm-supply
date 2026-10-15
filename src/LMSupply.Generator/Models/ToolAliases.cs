namespace LMSupply.Generator.Models;

/// <summary>
/// Alias for <see cref="ChatToolCall"/> for convenience.
/// </summary>
public sealed record ToolCall(
    string Id,
    string FunctionName,
    string Arguments)
{
    /// <summary>
    /// Converts to <see cref="ChatToolCall"/>.
    /// </summary>
    public static implicit operator ChatToolCall(ToolCall tc)
        => new(tc.Id, tc.FunctionName, tc.Arguments);

    /// <summary>
    /// Converts from <see cref="ChatToolCall"/>.
    /// </summary>
    public static implicit operator ToolCall(ChatToolCall tc)
        => new(tc.Id, tc.FunctionName, tc.Arguments);
}
