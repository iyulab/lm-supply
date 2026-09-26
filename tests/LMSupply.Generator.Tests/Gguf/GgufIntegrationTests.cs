using System.Text;
using AwesomeAssertions;
using LMSupply.Generator.Internal;
using LMSupply.Generator.Internal.Llama;
using LMSupply.Generator.Models;
using Xunit;

namespace LMSupply.Generator.Tests.Gguf;

/// <summary>
/// Integration tests for GGUF model support.
/// These tests require actual model downloads and inference,
/// so they are marked as Integration tests and skipped in CI.
/// Run with: dotnet test --filter "Category=Integration"
/// </summary>
[Trait("Category", "Integration")]
public class GgufIntegrationTests
{
    /// <summary>
    /// Tests that a GGUF model can be loaded from a registry alias.
    /// </summary>
    [Fact]
    public async Task LoadAsync_WithRegistryAlias_LoadsModel()
    {
        // Arrange
        var options = new GeneratorOptions
        {
            MaxContextLength = 2048 // Smaller context for faster testing
        };

        // Act
        await using var model = await LocalGenerator.LoadAsync("gguf:qwen3-fast", options, cancellationToken: TestContext.Current.CancellationToken);

        // Assert
        model.Should().NotBeNull();
        model.ModelId.Should().NotBeNullOrEmpty();
        model.MaxContextLength.Should().BeGreaterThan(0);
    }

    /// <summary>
    /// Tests text generation with a GGUF model.
    /// </summary>
    [Fact]
    public async Task GenerateAsync_WithGgufModel_GeneratesText()
    {
        // Arrange
        await using var model = await LocalGenerator.LoadAsync("gguf:qwen3-fast", cancellationToken: TestContext.Current.CancellationToken);

        // Act
        var result = new StringBuilder();
        await foreach (var token in model.GenerateAsync("Hello, my name is", new GenerationOptions { MaxTokens = 20 }, TestContext.Current.CancellationToken))
        {
            result.Append(token);
        }

        // Assert
        result.ToString().Should().NotBeNullOrEmpty();
        result.Length.Should().BeGreaterThan(0);
    }

    /// <summary>
    /// A raw prompt completion reports why it ended: <c>"length"</c> when it stopped at MaxTokens,
    /// <c>"stop"</c> when a stop sequence ended it (llama-server's <c>stop_type</c> = limit / word).
    /// </summary>
    [Fact]
    public async Task GenerateCompleteResultAsync_WithGgufModel_ReportsTheFinishReason()
    {
        await using var model = await LocalGenerator.LoadAsync("gguf:qwen3-fast", cancellationToken: TestContext.Current.CancellationToken);

        var cut = await model.GenerateCompleteResultAsync(
            "One, two, three, four, five, six, seven, eight, nine, ten, eleven,",
            new GenerationOptions { MaxTokens = 6, Temperature = 0f }, TestContext.Current.CancellationToken);
        cut.FinishReason.Should().Be("length");

        var stopped = await model.GenerateCompleteResultAsync(
            "One, two, three,",
            new GenerationOptions { MaxTokens = 64, Temperature = 0f, StopSequences = [","] }, TestContext.Current.CancellationToken);
        stopped.FinishReason.Should().Be("stop", "the stop sequence ended the completion");
    }

    /// <summary>
    /// A chat completion reports why it ended too — the string chat API cannot, so a consumer used to collect
    /// <c>GenerateChatStreamAsync</c> itself to learn it. llama-server's own <c>finish_reason</c> carries through.
    /// </summary>
    [Fact]
    public async Task GenerateChatCompleteResultAsync_WithGgufModel_ReportsTheFinishReason()
    {
        await using var model = await LocalGenerator.LoadAsync("gguf:qwen3-fast", cancellationToken: TestContext.Current.CancellationToken);

        var cut = await model.GenerateChatCompleteResultAsync(
            [ChatMessage.User("Count from one to fifty in words, separated by commas.")],
            new GenerationOptions { MaxTokens = 6, Temperature = 0f, Thinking = ThinkingMode.Off }, TestContext.Current.CancellationToken);
        cut.FinishReason.Should().Be("length", "six tokens cannot hold fifty numbers");
        cut.Content.Should().NotBeEmpty();

        var whole = await model.GenerateChatCompleteResultAsync(
            [ChatMessage.User("Reply with the single word: yes")],
            new GenerationOptions { MaxTokens = 64, Temperature = 0f, Thinking = ThinkingMode.Off }, TestContext.Current.CancellationToken);
        whole.FinishReason.Should().Be("stop", "a short answer ends on the model's end token");
        whole.Content.Should().Be(await model.GenerateChatCompleteAsync(
            [ChatMessage.User("Reply with the single word: yes")],
            new GenerationOptions { MaxTokens = 64, Temperature = 0f, Thinking = ThinkingMode.Off }, TestContext.Current.CancellationToken),
            "the result twin returns the same text as the string API");
    }

    /// <summary>
    /// A streamed chat turn ends with one chunk carrying both the finish reason and the server's token counts —
    /// reasoning included, which the visible text cannot show — the same counts the non-streamed call reports.
    /// </summary>
    [Fact]
    public async Task GenerateChatStreamAsync_WithGgufModel_ReportsServerUsageOnTheFinalChunk()
    {
        await using var model = await LocalGenerator.LoadAsync("gguf:qwen3-fast", cancellationToken: TestContext.Current.CancellationToken);
        var messages = new[] { ChatMessage.User("Is 17 a prime number? Answer yes or no.") };
        var options = new GenerationOptions { MaxTokens = 512, Temperature = 0f, Thinking = ThinkingMode.On };

        var chunks = new List<ChatStreamChunk>();
        await foreach (var chunk in model.GenerateChatStreamAsync(messages, options, TestContext.Current.CancellationToken))
            chunks.Add(chunk);

        chunks.Count(c => c.FinishReason is not null).Should().Be(1);
        var last = chunks[^1];
        last.FinishReason.Should().NotBeNull("the finish reason is on the last chunk, after any flushed text");
        last.Usage.Should().NotBeNull("llama-server reports usage on a streamed call");
        chunks.Take(chunks.Count - 1).Should().OnlyContain(c => c.Usage == null);

        var visible = string.Concat(chunks.Select(c => c.Text));
        var visibleTokens = await model.CountTokensAsync(visible, TestContext.Current.CancellationToken);
        last.Usage!.CompletionTokens.Should().BeGreaterThan(visibleTokens, "hidden reasoning is counted too");

        var whole = await model.GenerateChatWithToolsAsync(messages, options, TestContext.Current.CancellationToken);
        last.Usage.Should().BeEquivalentTo(whole.Usage, "greedy decoding of the same prompt generates the same tokens");
    }

    /// <summary>
    /// Tests chat generation with a GGUF model.
    /// </summary>
    [Fact]
    public async Task GenerateChatAsync_WithGgufModel_GeneratesResponse()
    {
        // Arrange
        await using var model = await LocalGenerator.LoadAsync("gguf:qwen3-fast", cancellationToken: TestContext.Current.CancellationToken);

        var messages = new[]
        {
            ChatMessage.System("You are a helpful assistant."),
            ChatMessage.User("What is 2+2?")
        };

        // Act
        var result = new StringBuilder();
        await foreach (var token in model.GenerateChatAsync(messages, new GenerationOptions { MaxTokens = 512 }, TestContext.Current.CancellationToken))
        {
            result.Append(token);
        }

        // Assert
        result.ToString().Should().NotBeNullOrEmpty();
    }

    /// <summary>
    /// Tests that GgufModelDownloader can list files from a repository.
    /// </summary>
    [Fact]
    public async Task GgufModelDownloader_ListGgufFilesAsync_ReturnsFiles()
    {
        // Arrange
        using var downloader = new GgufModelDownloader();

        // Act
        var files = await downloader.ListGgufFilesAsync("bartowski/Llama-3.2-1B-Instruct-GGUF", TestContext.Current.CancellationToken);

        // Assert
        files.Should().NotBeEmpty();
        files.Should().AllSatisfy(f =>
        {
            f.FileName.Should().EndWith(".gguf");
            f.SizeBytes.Should().BeGreaterThan(0);
        });
    }

    /// <summary>
    /// Tests that warmup works correctly for GGUF models.
    /// </summary>
    [Fact]
    public async Task WarmupAsync_WithGgufModel_CompletesSuccessfully()
    {
        // Arrange
        await using var model = await LocalGenerator.LoadAsync("gguf:qwen3-fast", cancellationToken: TestContext.Current.CancellationToken);

        // Act
        var warmupTask = model.WarmupAsync(TestContext.Current.CancellationToken);

        // Assert
        await warmupTask.Invoking(t => t).Should().NotThrowAsync();
    }

    /// <summary>
    /// Tests that model info is correctly populated for GGUF models.
    /// </summary>
    [Fact]
    public async Task GetModelInfo_WithGgufModel_ReturnsCorrectInfo()
    {
        // Arrange
        await using var model = await LocalGenerator.LoadAsync("gguf:qwen3-fast", cancellationToken: TestContext.Current.CancellationToken);

        // Act
        var info = model.GetModelInfo();

        // Assert
        info.ModelId.Should().NotBeNullOrEmpty();
        info.ModelPath.Should().EndWith(".gguf");
        info.MaxContextLength.Should().BeGreaterThan(0);
        info.ChatFormat.Should().NotBeNullOrEmpty();
        info.ExecutionProvider.Should().StartWith("llama-server-");
    }
}

/// <summary>
/// Unit tests for GGUF model format detection.
/// These tests don't require model downloads.
/// </summary>
public class GgufModelFormatTests
{
    /// <summary>
    /// Tests model format detection for GGUF files.
    /// </summary>
    [Theory]
    [InlineData("gguf:qwen3-default", true)]
    [InlineData("gguf:qwen3-fast", true)]
    [InlineData("gguf:quality", true)]
    [InlineData("gguf:korean", true)]
    [InlineData("bartowski/Llama-3.2-3B-Instruct-GGUF", true)]
    [InlineData("microsoft/Phi-4-mini-instruct-onnx", false)]
    public void ModelFormatDetector_DetectsGgufFormat(string modelId, bool expectedGguf)
    {
        // Act
        var format = ModelFormatDetector.Detect(modelId);

        // Assert
        if (expectedGguf)
        {
            format.Should().Be(ModelFormat.Gguf);
        }
        else
        {
            format.Should().NotBe(ModelFormat.Gguf);
        }
    }

    [Theory]
    [InlineData("/path/to/model.gguf", true)]
    [InlineData("/path/to/model.onnx", false)]
    [InlineData("C:\\models\\test.gguf", true)]
    public void ModelFormatDetector_DetectsFromFilePath(string path, bool expectedGguf)
    {
        // Act
        var format = ModelFormatDetector.Detect(path);

        // Assert
        if (expectedGguf)
        {
            format.Should().Be(ModelFormat.Gguf);
        }
        else
        {
            format.Should().NotBe(ModelFormat.Gguf);
        }
    }
}
