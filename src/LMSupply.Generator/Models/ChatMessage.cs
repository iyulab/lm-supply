namespace LMSupply.Generator.Models;

/// <summary>
/// Represents a message in a chat conversation.
/// </summary>
/// <param name="Role">The role of the message sender.</param>
/// <param name="Content">The content of the message.</param>
public readonly record struct ChatMessage(ChatRole Role, string Content)
{
    /// <summary>
    /// Creates a system message.
    /// </summary>
    public static ChatMessage System(string content) => new(ChatRole.System, content);

    /// <summary>
    /// Creates a user message.
    /// </summary>
    public static ChatMessage User(string content) => new(ChatRole.User, content);

    /// <summary>
    /// Creates an assistant message.
    /// </summary>
    public static ChatMessage Assistant(string content) => new(ChatRole.Assistant, content);
}

/// <summary>
/// Represents the role of a participant in a chat conversation.
/// </summary>
public enum ChatRole
{
    /// <summary>
    /// System message providing context or instructions.
    /// </summary>
    System,

    /// <summary>
    /// User message from the human participant.
    /// </summary>
    User,

    /// <summary>
    /// Assistant message from the AI model.
    /// </summary>
    Assistant
}
