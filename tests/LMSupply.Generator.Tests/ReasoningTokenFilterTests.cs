using System.Text;
using AwesomeAssertions;
using LMSupply.Generator.Internal;

namespace LMSupply.Generator.Tests;

public class ReasoningTokenFilterTests
{
    [Fact]
    public void Process_WithoutReasoningTags_PassesThrough()
    {
        var filter = new ReasoningTokenFilter();

        var result = filter.Process("Hello, world!");

        result.Should().Be("Hello, world!");
    }

    [Fact]
    public void Process_WithThinkTag_FiltersContent()
    {
        var filter = new ReasoningTokenFilter();

        var output = new StringBuilder();
        output.Append(filter.Process("Hello "));
        output.Append(filter.Process("<think>"));
        output.Append(filter.Process("reasoning content"));
        output.Append(filter.Process("</think>"));
        output.Append(filter.Process(" world!"));
        output.Append(filter.Flush());

        output.ToString().Should().Be("Hello  world!");
    }

    [Fact]
    public void Process_WithPartialTags_HandlesCorrectly()
    {
        var filter = new ReasoningTokenFilter();

        var output = new StringBuilder();
        output.Append(filter.Process("Hello "));
        output.Append(filter.Process("<thi")); // Partial opening tag
        output.Append(filter.Process("nk>"));
        output.Append(filter.Process("reasoning"));
        output.Append(filter.Process("</thi")); // Partial closing tag
        output.Append(filter.Process("nk>"));
        output.Append(filter.Process(" done"));
        output.Append(filter.Flush());

        output.ToString().Should().Be("Hello  done");
    }

    [Fact]
    public void Process_WithExtractReasoning_CapturesContent()
    {
        var filter = new ReasoningTokenFilter(extractReasoning: true);

        var output = new StringBuilder();
        output.Append(filter.Process("Hello "));
        output.Append(filter.Process("<think>"));
        output.Append(filter.Process("this is my reasoning"));
        output.Append(filter.Process("</think>"));
        output.Append(filter.Process(" world!"));
        output.Append(filter.Flush());

        output.ToString().Should().Be("Hello  world!");
        filter.ReasoningContent.Should().Be("this is my reasoning");
    }

    [Fact]
    public void Process_WithDeepSeekFormat_FiltersCorrectly()
    {
        var filter = new ReasoningTokenFilter();

        var output = new StringBuilder();
        output.Append(filter.Process("Answer: "));
        output.Append(filter.Process("<｜begin▁of▁thinking｜>"));
        output.Append(filter.Process("let me think..."));
        output.Append(filter.Process("<｜end▁of▁thinking｜>"));
        output.Append(filter.Process(" 42"));
        output.Append(filter.Flush());

        output.ToString().Should().Be("Answer:  42");
    }

    [Fact]
    public void Process_MultipleReasoningBlocks_FiltersAll()
    {
        var filter = new ReasoningTokenFilter(extractReasoning: true);

        var output = new StringBuilder();
        output.Append(filter.Process("First "));
        output.Append(filter.Process("<think>block1</think>"));
        output.Append(filter.Process(" middle "));
        output.Append(filter.Process("<think>block2</think>"));
        output.Append(filter.Process(" end"));
        output.Append(filter.Flush());

        output.ToString().Should().Be("First  middle  end");
        filter.ReasoningContent.Should().Be("block1block2");
    }

    [Fact]
    public void Process_UnclosedTag_TreatsAsReasoning()
    {
        var filter = new ReasoningTokenFilter();

        var output = new StringBuilder();
        output.Append(filter.Process("Start "));
        output.Append(filter.Process("<think>"));
        output.Append(filter.Process("never closed"));
        output.Append(filter.Flush());

        // Unclosed reasoning should not appear in output
        output.ToString().Should().Be("Start ");
    }

    [Fact]
    public void Reset_ClearsState()
    {
        var filter = new ReasoningTokenFilter(extractReasoning: true);

        filter.Process("<think>reasoning</think>");
        filter.ReasoningContent.Should().Be("reasoning");

        filter.Reset();

        filter.ReasoningContent.Should().BeEmpty();
    }

    [Fact]
    public void Process_StreamingTokens_WorksCorrectly()
    {
        // Simulate streaming where tags are split across tokens
        var filter = new ReasoningTokenFilter();

        var tokens = new[] { "Hel", "lo ", "<", "think", ">", "rea", "son", "<", "/think", ">", " world" };
        var output = new StringBuilder();

        foreach (var token in tokens)
        {
            output.Append(filter.Process(token));
        }
        output.Append(filter.Flush());

        output.ToString().Should().Be("Hello  world");
    }

    [Fact]
    public void Process_EmptyInput_ReturnsEmpty()
    {
        var filter = new ReasoningTokenFilter();

        var result = filter.Process("");

        result.Should().BeEmpty();
    }

    [Fact]
    public void Flush_WithoutProcessing_ReturnsEmpty()
    {
        var filter = new ReasoningTokenFilter();

        var result = filter.Flush();

        result.Should().BeEmpty();
    }

    [Fact]
    public void Process_TagLikeButNotTag_PassesThrough()
    {
        var filter = new ReasoningTokenFilter();

        var result = filter.Process("Use <thinking> for thoughts"); // Not exact match
        result += filter.Flush();

        result.Should().Be("Use <thinking> for thoughts");
    }
}
