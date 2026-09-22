using AwesomeAssertions;

namespace LMSupply.Embedder.Tests;

/// <summary>
/// <see cref="LocalEmbedder.GetVectorSpaceRevisionAsync"/> answers from the cached files alone — no session, no
/// download — or says it cannot (<c>null</c>). A consumer that names its vector store after the identity before
/// the model is loaded is what this exists for (0.72.0).
/// </summary>
public sealed class VectorSpaceRevisionPreloadTests : IDisposable
{
    private readonly string _dir = Path.Combine(Path.GetTempPath(), $"lmsupply-vs-preload-{Guid.NewGuid():N}");

    private static CancellationToken Ct => TestContext.Current.CancellationToken;

    public void Dispose()
    {
        if (Directory.Exists(_dir))
            Directory.Delete(_dir, recursive: true);
    }

    private string WriteLocalModel(bool withConfig = true, int hiddenSize = 384, bool clsPooling = false)
    {
        Directory.CreateDirectory(_dir);
        var modelPath = Path.Combine(_dir, "model.onnx");
        File.WriteAllBytes(modelPath, [0x00]); // File.Exists is all the pre-load read needs; no session is opened
        File.WriteAllText(Path.Combine(_dir, "vocab.txt"), "[PAD]\n[UNK]\n[CLS]\n[SEP]\nhello\nworld\n");
        if (withConfig)
            File.WriteAllText(Path.Combine(_dir, "config.json"), $"{{\"hidden_size\": {hiddenSize}}}");
        if (clsPooling)
        {
            Directory.CreateDirectory(Path.Combine(_dir, "1_Pooling"));
            File.WriteAllText(Path.Combine(_dir, "1_Pooling", "config.json"), "{\"pooling_mode_cls_token\": true, \"pooling_mode_mean_tokens\": false}");
        }
        return modelPath;
    }

    [Fact]
    public async Task A_cached_local_model_answers_from_its_files_without_a_session()
    {
        var modelPath = WriteLocalModel();

        var revision = await LocalEmbedder.GetVectorSpaceRevisionAsync(modelPath, cancellationToken: Ct);

        revision.Should().NotBeNull().And.MatchRegex("^[0-9a-f]{16}$");
        (await LocalEmbedder.GetVectorSpaceRevisionAsync(modelPath, cancellationToken: Ct)).Should().Be(revision, "deterministic");
    }

    [Fact]
    public async Task The_declared_dimension_and_pooling_are_part_of_the_answer()
    {
        var modelPath = WriteLocalModel(hiddenSize: 384);
        var at384 = await LocalEmbedder.GetVectorSpaceRevisionAsync(modelPath, cancellationToken: Ct);

        File.WriteAllText(Path.Combine(_dir, "config.json"), "{\"hidden_size\": 768}");
        var at768 = await LocalEmbedder.GetVectorSpaceRevisionAsync(modelPath, cancellationToken: Ct);

        Directory.CreateDirectory(Path.Combine(_dir, "1_Pooling"));
        File.WriteAllText(Path.Combine(_dir, "1_Pooling", "config.json"), "{\"pooling_mode_cls_token\": true, \"pooling_mode_mean_tokens\": false}");
        var cls = await LocalEmbedder.GetVectorSpaceRevisionAsync(modelPath, cancellationToken: Ct);

        at768.Should().NotBe(at384, "the dimension is a vector-space decision");
        cls.Should().NotBe(at768, "pooling read from 1_Pooling/config.json is a vector-space decision");
    }

    [Fact]
    public async Task Without_a_declared_dimension_it_says_it_cannot_know()
    {
        var modelPath = WriteLocalModel(withConfig: false);

        (await LocalEmbedder.GetVectorSpaceRevisionAsync(modelPath, cancellationToken: Ct))
            .Should().BeNull("only the ONNX graph knows the width, and that needs a session");
    }

    [Fact]
    public async Task The_callers_options_are_not_mutated()
    {
        var modelPath = WriteLocalModel();
        var options = new EmbedderOptions();

        await LocalEmbedder.GetVectorSpaceRevisionAsync(modelPath, options, Ct);

        options.MaxSequenceLength.Should().BeNull();
        options.PoolingMode.Should().BeNull();
        options.DisableAutoDownload.Should().BeFalse();
    }

    [Theory]
    [InlineData("default")]
    [InlineData("sentence-transformers/all-MiniLM-L6-v2")]
    [InlineData("no-such-alias")]
    public async Task A_model_not_in_the_cache_is_null_and_downloads_nothing(string modelId)
    {
        var options = new EmbedderOptions { CacheDirectory = _dir };

        var revision = await LocalEmbedder.GetVectorSpaceRevisionAsync(modelId, options, Ct);

        revision.Should().BeNull();
        Directory.Exists(_dir).Should().BeFalse("a miss writes nothing to the cache and fetches nothing");
    }

    [Fact]
    public async Task A_gguf_model_is_null_because_the_server_decides_its_dimension()
    {
        (await LocalEmbedder.GetVectorSpaceRevisionAsync("gguf:nomic-ai/nomic-embed-text-v1.5-GGUF", new EmbedderOptions { CacheDirectory = _dir }, Ct))
            .Should().BeNull();
    }
}
