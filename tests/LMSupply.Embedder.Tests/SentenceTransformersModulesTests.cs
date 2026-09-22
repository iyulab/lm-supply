using AwesomeAssertions;
using LMSupply.Embedder.Utils;

namespace LMSupply.Embedder.Tests;

/// <summary>
/// A model loaded by repository id declares its pooling in <c>1_Pooling/config.json</c> and, sometimes,
/// its prompts in <c>config_sentence_transformers.json</c>. The shapes below are the files as published
/// (BAAI/bge-m3 pools on CLS; intfloat/multilingual-e5-small on the mean) — read, not guessed.
/// </summary>
public sealed class SentenceTransformersModulesTests : IDisposable
{
    private readonly string _dir = Path.Combine(Path.GetTempPath(), $"st-modules-{Guid.NewGuid():N}");

    public SentenceTransformersModulesTests() => Directory.CreateDirectory(_dir);

    public void Dispose()
    {
        if (Directory.Exists(_dir))
        {
            Directory.Delete(_dir, true);
        }
    }

    private void WritePooling(string json)
    {
        Directory.CreateDirectory(Path.Combine(_dir, "1_Pooling"));
        File.WriteAllText(Path.Combine(_dir, "1_Pooling", "config.json"), json);
    }

    private void WriteStConfig(string json) =>
        File.WriteAllText(Path.Combine(_dir, SentenceTransformersModules.SentenceTransformersConfigFileName), json);

    [Fact]
    public void ClsPooling_AsBgeM3Declares_IsRead()
    {
        WritePooling("""
            {
              "word_embedding_dimension": 1024,
              "pooling_mode_cls_token": true,
              "pooling_mode_mean_tokens": false,
              "pooling_mode_max_tokens": false,
              "pooling_mode_mean_sqrt_len_tokens": false
            }
            """);

        SentenceTransformersModules.TryReadPoolingMode(_dir).Should().Be(PoolingMode.Cls);
    }

    [Fact]
    public void MeanPooling_AsE5Declares_IsRead()
    {
        WritePooling("""
            {
                "word_embedding_dimension": 384,
                "pooling_mode_cls_token": false,
                "pooling_mode_mean_tokens": true,
                "pooling_mode_max_tokens": false,
                "pooling_mode_mean_sqrt_len_tokens": false
            }
            """);

        SentenceTransformersModules.TryReadPoolingMode(_dir).Should().Be(PoolingMode.Mean);
    }

    [Fact]
    public void MaxPooling_IsRead()
    {
        WritePooling("""{ "pooling_mode_max_tokens": true }""");

        SentenceTransformersModules.TryReadPoolingMode(_dir).Should().Be(PoolingMode.Max);
    }

    [Theory]
    [InlineData("""{ "pooling_mode_mean_sqrt_len_tokens": true }""")]
    [InlineData("""{ "pooling_mode_lasttoken": true }""")]
    [InlineData("""{ "pooling_mode_weightedmean_tokens": true, "pooling_mode_mean_tokens": false }""")]
    [InlineData("""{ "pooling_mode_cls_token": true, "pooling_mode_mean_tokens": true }""")]
    [InlineData("""{ "word_embedding_dimension": 384 }""")]
    [InlineData("""[1, 2]""")]
    [InlineData("""not json""")]
    public void APoolingThisLibraryDoesNotImplement_OrAnAmbiguousFile_IsUnknown_NotAGuess(string json)
    {
        WritePooling(json);

        SentenceTransformersModules.TryReadPoolingMode(_dir).Should().BeNull();
    }

    [Fact]
    public void NoPoolingFile_IsUnknown() => SentenceTransformersModules.TryReadPoolingMode(_dir, null).Should().BeNull();

    [Fact]
    public void TheFirstRootThatHasTheFile_Decides()
    {
        WritePooling("""{ "pooling_mode_cls_token": true }""");

        SentenceTransformersModules.TryReadPoolingMode(null, Path.Combine(_dir, "onnx"), _dir).Should().Be(PoolingMode.Cls);
    }

    [Fact]
    public void DeclaredPrompts_AreRead_WithDocumentAsTheFallbackNameForPassage()
    {
        WriteStConfig("""
            {
              "__version__": { "sentence_transformers": "3.0.0" },
              "prompts": { "query": "query: ", "document": "passage: " },
              "default_prompt_name": null
            }
            """);

        SentenceTransformersModules.TryReadPrompts(_dir).Should().Be(("query: ", "passage: "));
    }

    [Fact]
    public void AConfigWithoutPrompts_AsMostModelsPublish_DeclaresNone()
    {
        WriteStConfig("""{ "__version__": { "sentence_transformers": "2.0.0", "transformers": "4.6.1", "pytorch": "1.8.1" } }""");

        SentenceTransformersModules.TryReadPrompts(_dir).Should().Be(((string?)null, (string?)null));
    }

    [Fact]
    public void NoConfigFile_DeclaresNoPrompts() =>
        SentenceTransformersModules.TryReadPrompts(_dir).Should().Be(((string?)null, (string?)null));
}
