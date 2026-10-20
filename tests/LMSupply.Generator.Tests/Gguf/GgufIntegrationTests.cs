using System.Text;
using AwesomeAssertions;
using LMSupply.Generator.Internal;
using LMSupply.Generator.Gguf;
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
    /// Per-token log probabilities reach every llama-server result path, and they are the distribution before
    /// sampling: a near-greedy request reports the same values as a hot one. The label's id comes from
    /// <c>TokenizeAsync</c> — for this model's tokenizer <c>"B"</c> is the single token 33.
    /// </summary>
    [Fact]
    public async Task TopLogprobs_WithGgufModel_ReportTheDistributionBeforeSampling()
    {
        var ct = TestContext.Current.CancellationToken;
        await using var model = await LocalGenerator.LoadAsync("gguf:qwen3-fast", cancellationToken: ct);
        ChatMessage[] question =
        [
            ChatMessage.User("Which planet is known as the Red Planet?\nA) Venus\nB) Mars\nC) Jupiter\nD) Saturn\nAnswer with a single letter."),
        ];

        var b = await model.TokenizeAsync("B", ct);
        b.Should().Equal(33);

        var hot = await model.GenerateChatCompleteResultAsync(
            question, new GenerationOptions { MaxTokens = 1, Temperature = 1.5f, Thinking = ThinkingMode.Off, TopLogprobs = 5 }, ct);
        var token = hot.Logprobs.Should().ContainSingle().Subject;
        token.TopLogprobs.Should().HaveCount(5);
        token.TopLogprobs.Select(c => c.Logprob).Should().BeInDescendingOrder().And.OnlyContain(l => l <= 0);
        var answer = token.TopLogprobs.Should().ContainSingle(c => c.TokenId == 33).Subject;
        answer.Token.Should().Be("B");
        answer.Bytes.Should().Equal((byte)'B');

        var cold = await model.GenerateChatCompleteResultAsync(
            question, new GenerationOptions { MaxTokens = 1, Temperature = 0.1f, TopK = 1, Thinking = ThinkingMode.Off, TopLogprobs = 5 }, ct);
        cold.Logprobs!.Single().TopLogprobs.Select(c => c.TokenId).Should().Equal(token.TopLogprobs.Select(c => c.TokenId));
        cold.Logprobs!.Single().TopLogprobs[0].Logprob.Should().BeApproximately(token.TopLogprobs[0].Logprob, 1e-3,
            "sampling settings do not change the values");

        var tools = await model.GenerateChatWithToolsAsync(
            question, new GenerationOptions { MaxTokens = 1, Thinking = ThinkingMode.Off, TopLogprobs = 0 }, ct);
        tools.Logprobs.Should().ContainSingle().Which.TopLogprobs.Should().BeEmpty("0 asks for the sampled token alone");

        var streamed = new List<TokenLogprob>();
        await foreach (var chunk in model.GenerateChatStreamAsync(
            question, new GenerationOptions { MaxTokens = 1, Thinking = ThinkingMode.Off, TopLogprobs = 2 }, ct))
            streamed.AddRange(chunk.Logprobs ?? []);
        streamed.Should().ContainSingle().Which.TopLogprobs.Should().HaveCount(2);

        var raw = await model.GenerateCompleteResultAsync(
            "One, two, three,", new GenerationOptions { MaxTokens = 2, TopLogprobs = 2 }, ct);
        raw.Logprobs.Should().HaveCount(2);
        raw.Logprobs!.Should().OnlyContain(t => t.TopLogprobs.Count == 2 && t.TokenId != null);
    }

    /// <summary>
    /// Every llama-server result path carries the server's own timings: the raw prompt result, the chat result, the
    /// chat-with-tools result and the last chunk of the chat stream. A rate taken from the visible text cannot stand
    /// in for them, since a reasoning model generates tokens the text never shows.
    /// </summary>
    [Fact]
    public async Task EveryResultPath_WithGgufModel_CarriesTheServerTimings()
    {
        var ct = TestContext.Current.CancellationToken;
        await using var model = await LocalGenerator.LoadAsync("gguf:qwen3-fast", cancellationToken: ct);
        var options = new GenerationOptions { MaxTokens = 32, Temperature = 0f, Thinking = ThinkingMode.Off };
        ChatMessage[] messages = [ChatMessage.User("Name three colours.")];

        var raw = await model.GenerateCompleteResultAsync("The sky is blue because", options, ct);
        var chat = await model.GenerateChatCompleteResultAsync(messages, options, ct);
        var tools = await model.GenerateChatWithToolsAsync(messages, options, ct);
        ChatStreamChunk? last = null;
        await foreach (var chunk in model.GenerateChatStreamAsync(messages, options, ct))
            last = chunk;

        foreach (var (path, timings) in new[] { ("raw", raw.Timings), ("chat", chat.Timings), ("tools", tools.Timings), ("stream", last?.Timings) })
        {
            timings.Should().NotBeNull(path);
            timings!.CompletionTokensPerSecond.Should().BePositive(path);
            timings.CompletionDuration.Should().BeGreaterThan(TimeSpan.Zero, path);
            timings.PromptTokensEvaluated.Should().NotBeNull(path);
        }
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
    /// A raw prompt long enough to take several batches reports how far the server has read it, on chunks of their own
    /// before the first token; the text is the plain stream's, and the last chunk carries the ending and the usage.
    /// </summary>
    [Fact]
    public async Task GenerateStreamAsync_WithGgufModel_ReportsPromptProgressThenTheEnding()
    {
        var ct = TestContext.Current.CancellationToken;
        await using var model = await LocalGenerator.LoadAsync("gguf:qwen3-fast", cancellationToken: ct);
        var prompt = string.Concat(Enumerable.Repeat("The quick brown fox jumps over the lazy dog. ", 150));
        var options = new GenerationOptions { MaxTokens = 8, Temperature = 0f };

        var chunks = new List<TextStreamChunk>();
        await foreach (var chunk in model.GenerateStreamAsync(prompt, options, ct))
            chunks.Add(chunk);

        var progress = chunks.TakeWhile(c => c.Text is null && c.FinishReason is null).ToList();
        progress.Should().NotBeEmpty().And.OnlyContain(c => c.PromptProgress != null, "the server reads the prompt first");
        progress[^1].PromptProgress!.ProcessedTokens.Should().Be(progress[^1].PromptProgress!.TotalTokens);
        chunks.Skip(progress.Count).Should().OnlyContain(c => c.PromptProgress == null);

        var last = chunks[^1];
        last.FinishReason.Should().Be("length");
        last.Usage!.CompletionTokens.Should().Be(8);
        last.Timings.Should().NotBeNull();

        var plain = new List<string>();
        await foreach (var token in model.GenerateAsync(prompt, options, ct))
            plain.Add(token);
        string.Concat(chunks.Select(c => c.Text)).Should().Be(string.Concat(plain), "greedy decoding of the same prompt");
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
