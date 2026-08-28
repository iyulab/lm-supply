using AwesomeAssertions;
using LMSupply.Generator.Internal.Llama;
using LMSupply.Generator.Models;

namespace LMSupply.Generator.Tests;

public class SlidingWindowTrimTests
{
    // --- TryTrimOldestTurn ---

    [Fact]
    public void TryTrimOldestTurn_EmptyList_ReturnsFalse()
    {
        var messages = new List<ChatMessage>();
        LlamaServerGeneratorModel.TryTrimOldestTurn(messages).Should().BeFalse();
        messages.Should().BeEmpty();
    }

    [Fact]
    public void TryTrimOldestTurn_OnlySystemMessage_ReturnsFalse()
    {
        var messages = new List<ChatMessage> { ChatMessage.System("You are helpful.") };
        LlamaServerGeneratorModel.TryTrimOldestTurn(messages).Should().BeFalse();
        messages.Should().HaveCount(1);
    }

    [Fact]
    public void TryTrimOldestTurn_SingleUserMessage_ReturnsFalse()
    {
        // Last User message must never be removed
        var messages = new List<ChatMessage> { ChatMessage.User("Hello?") };
        LlamaServerGeneratorModel.TryTrimOldestTurn(messages).Should().BeFalse();
        messages.Should().HaveCount(1);
    }

    [Fact]
    public void TryTrimOldestTurn_SystemPlusSingleUserMessage_ReturnsFalse()
    {
        var messages = new List<ChatMessage>
        {
            ChatMessage.System("sys"),
            ChatMessage.User("current turn")
        };
        LlamaServerGeneratorModel.TryTrimOldestTurn(messages).Should().BeFalse();
        messages.Should().HaveCount(2);
    }

    [Fact]
    public void TryTrimOldestTurn_TwoUserMessages_RemovesFirstTurn()
    {
        // Oldest turn: User[0] + Assistant[1] — removed.
        // User[2] is the current turn — preserved.
        var messages = new List<ChatMessage>
        {
            ChatMessage.User("turn 1 question"),
            ChatMessage.Assistant("turn 1 answer"),
            ChatMessage.User("turn 2 question")
        };

        LlamaServerGeneratorModel.TryTrimOldestTurn(messages).Should().BeTrue();

        messages.Should().HaveCount(1);
        messages[0].Role.Should().Be(ChatRole.User);
        messages[0].Content.Should().Be("turn 2 question");
    }

    [Fact]
    public void TryTrimOldestTurn_PreservesSystemMessage()
    {
        var messages = new List<ChatMessage>
        {
            ChatMessage.System("system"),
            ChatMessage.User("turn 1"),
            ChatMessage.Assistant("answer 1"),
            ChatMessage.User("turn 2")
        };

        LlamaServerGeneratorModel.TryTrimOldestTurn(messages).Should().BeTrue();

        messages.Should().HaveCount(2);
        messages[0].Role.Should().Be(ChatRole.System);
        messages[0].Content.Should().Be("system");
        messages[1].Content.Should().Be("turn 2");
    }

    [Fact]
    public void TryTrimOldestTurn_ToolCallResultPairRemovedAtomically()
    {
        // Turn 1: User → Assistant(tool call) → Tool(result)
        // Turn 2: User (preserved)
        var toolCall = new ChatToolCall("tc-1", "search", "{}");
        var messages = new List<ChatMessage>
        {
            ChatMessage.User("find something"),
            ChatMessage.AssistantToolCalls([toolCall]),
            ChatMessage.ToolResult("tc-1", "result text"),
            ChatMessage.User("current question")
        };

        LlamaServerGeneratorModel.TryTrimOldestTurn(messages).Should().BeTrue();

        messages.Should().HaveCount(1);
        messages[0].Role.Should().Be(ChatRole.User);
        messages[0].Content.Should().Be("current question");
    }

    [Fact]
    public void TryTrimOldestTurn_MultipleToolResultsRemovedAtomically()
    {
        // Turn 1: User → Assistant(2 tool calls) → Tool × 2
        var tc1 = new ChatToolCall("tc-1", "search", "{}");
        var tc2 = new ChatToolCall("tc-2", "calc", "{}");
        var messages = new List<ChatMessage>
        {
            ChatMessage.User("multi-tool turn"),
            ChatMessage.AssistantToolCalls([tc1, tc2]),
            ChatMessage.ToolResult("tc-1", "r1"),
            ChatMessage.ToolResult("tc-2", "r2"),
            ChatMessage.User("next turn")
        };

        LlamaServerGeneratorModel.TryTrimOldestTurn(messages).Should().BeTrue();

        messages.Should().HaveCount(1);
        messages[0].Content.Should().Be("next turn");
    }

    [Fact]
    public void TryTrimOldestTurn_ProgressivelyReducesToMinimum()
    {
        var messages = new List<ChatMessage>
        {
            ChatMessage.System("sys"),
            ChatMessage.User("q1"),
            ChatMessage.Assistant("a1"),
            ChatMessage.User("q2"),
            ChatMessage.Assistant("a2"),
            ChatMessage.User("q3")
        };

        // First trim: removes q1+a1
        LlamaServerGeneratorModel.TryTrimOldestTurn(messages).Should().BeTrue();
        messages.Should().HaveCount(4);

        // Second trim: removes q2+a2
        LlamaServerGeneratorModel.TryTrimOldestTurn(messages).Should().BeTrue();
        messages.Should().HaveCount(2);
        messages[0].Role.Should().Be(ChatRole.System);
        messages[1].Content.Should().Be("q3");

        // Third trim: only system + last user remain — cannot trim
        LlamaServerGeneratorModel.TryTrimOldestTurn(messages).Should().BeFalse();
        messages.Should().HaveCount(2);
    }

    [Fact]
    public void TryTrimOldestTurn_EnableThinking_SystemNotRemovedAsNonSystem()
    {
        // Verify that injected system messages (thinking token, tool fragment)
        // treated the same as regular system: never trimmed directly.
        // Turn 1: System(injected) + User + Assistant + User(current)
        var messages = new List<ChatMessage>
        {
            ChatMessage.System("<|think|>\\nYou are helpful."),
            ChatMessage.User("q1"),
            ChatMessage.Assistant("a1"),
            ChatMessage.User("q2 current")
        };

        LlamaServerGeneratorModel.TryTrimOldestTurn(messages).Should().BeTrue();

        messages.Should().HaveCount(2);
        messages[0].Role.Should().Be(ChatRole.System);
        messages[1].Content.Should().Be("q2 current");
    }
}
