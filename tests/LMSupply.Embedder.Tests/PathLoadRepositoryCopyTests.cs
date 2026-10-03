using AwesomeAssertions;

namespace LMSupply.Embedder.Tests;

/// <summary>
/// A model shipped as a copied repository (<c>&lt;root&gt;/onnx/model.onnx</c> plus the repository's root files) and
/// loaded by path is the same model as its catalog alias: root files are read from the root, and a model the catalog
/// knows keeps the catalog's declarations — for E5, the query/passage prefixes its repository never declares.
/// </summary>
public class PathLoadRepositoryCopyTests : IDisposable
{
    private readonly string _root = Path.Combine(Path.GetTempPath(), "lmsupply-pathload-" + Guid.NewGuid().ToString("N"));

    public PathLoadRepositoryCopyTests() => Directory.CreateDirectory(Path.Combine(_root, "onnx"));

    public void Dispose()
    {
        try { Directory.Delete(_root, recursive: true); } catch (IOException) { } catch (UnauthorizedAccessException) { }
        GC.SuppressFinalize(this);
    }

    [Fact]
    public void FindRepositoryRoot_OnnxFolderOfARepositoryCopy_IsTheParent()
    {
        File.WriteAllText(Path.Combine(_root, "modules.json"), "[]");

        LocalEmbedder.FindRepositoryRoot(Path.Combine(_root, "onnx")).Should().Be(_root);
    }

    [Fact]
    public void FindRepositoryRoot_FolderWithoutRootFilesAbove_IsItself()
    {
        var onnx = Path.Combine(_root, "onnx");

        LocalEmbedder.FindRepositoryRoot(onnx).Should().Be(onnx);
    }

    [Theory]
    [InlineData("intfloat/multilingual-e5-small", "intfloat/multilingual-e5-small")]
    [InlineData("/home/me/models/e5", null)]
    [InlineData("C:\\\\models\\\\e5", null)]
    [InlineData("./e5", null)]
    [InlineData("e5-small", null)]
    public void ReadDeclaredRepositoryId_ReadsNameOrPathOnlyWhenItIsARepositoryId(string nameOrPath, string? expected)
    {
        File.WriteAllText(Path.Combine(_root, "config.json"), $$"""{"_name_or_path":"{{nameOrPath}}"}""");

        LocalEmbedder.ReadDeclaredRepositoryId(Path.Combine(_root, "onnx"), _root).Should().Be(expected);
    }

    [Fact]
    public void ReadDeclaredRepositoryId_TheDownloadManifestWins()
    {
        File.WriteAllText(Path.Combine(_root, "config.json"), """{"_name_or_path":"someone/else"}""");
        File.WriteAllText(Path.Combine(_root, "onnx", ".lmsupply-manifest.json"), """{"version":2,"repoId":"intfloat/multilingual-e5-small","files":[]}""");

        LocalEmbedder.ReadDeclaredRepositoryId(Path.Combine(_root, "onnx"), _root).Should().Be("intfloat/multilingual-e5-small");
    }
}

/// <summary>
/// Against the real multilingual-e5-small files from the local Hugging Face cache, laid out as a copied repository.
/// Excluded from CI (needs the cached model).
/// </summary>
[Trait("Category", "Integration")]
public sealed class PathLoadRepositoryCopyLiveTests : IDisposable
{
    private static readonly string CachedOnnx = Path.Combine(
        Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
        ".cache", "huggingface", "hub", "models--intfloat--multilingual-e5-small", "snapshots", "main", "onnx");

    private readonly string _root = Path.Combine(Path.GetTempPath(), "lmsupply-e5copy-" + Guid.NewGuid().ToString("N"));

    public void Dispose()
    {
        try { Directory.Delete(_root, recursive: true); } catch (IOException) { } catch (UnauthorizedAccessException) { }
    }

    [Fact]
    public async Task CopiedE5Repository_LoadedByPath_EmbedsQueriesLikeTheCatalogAlias()
    {
        Assert.SkipUnless(File.Exists(Path.Combine(CachedOnnx, "model.onnx")), $"model not cached: {CachedOnnx}");
        var ct = TestContext.Current.CancellationToken;
        CopyAsRepository(declareRepository: true);

        await using var byPath = await LocalEmbedder.LoadAsync(Path.Combine(_root, "onnx", "model.onnx"), cancellationToken: ct);
        await using var byAlias = await LocalEmbedder.LoadAsync("intfloat/multilingual-e5-small",
            new EmbedderOptions { DisableAutoDownload = true }, cancellationToken: ct);

        byPath.GetModelInfo()!.QueryPrefix.Should().Be("query: ");
        byPath.GetModelInfo()!.PassagePrefix.Should().Be("passage: ");
        (await byPath.EmbedQueryAsync("서울 날씨", ct)).Should().Equal(await byAlias.EmbedQueryAsync("서울 날씨", ct));
    }

    [Fact]
    public async Task CopiedRepositoryThatDoesNotSayWhatItIs_LoadsWithoutCatalogPrefixes()
    {
        // Positive control for the fact above: the same files, with nothing naming the repository, get no prefixes —
        // so it is the declared identity, not the layout, that carries them.
        Assert.SkipUnless(File.Exists(Path.Combine(CachedOnnx, "model.onnx")), $"model not cached: {CachedOnnx}");
        CopyAsRepository(declareRepository: false);

        await using var byPath = await LocalEmbedder.LoadAsync(Path.Combine(_root, "onnx", "model.onnx"),
            cancellationToken: TestContext.Current.CancellationToken);

        byPath.GetModelInfo()?.QueryPrefix.Should().BeNull();
    }

    private void CopyAsRepository(bool declareRepository)
    {
        var onnx = Directory.CreateDirectory(Path.Combine(_root, "onnx")).FullName;
        foreach (var file in Directory.GetFiles(CachedOnnx))
        {
            var name = Path.GetFileName(file);
            if (name.StartsWith(".lmsupply", StringComparison.Ordinal) || name == "config.json")
                continue;
            File.Copy(file, Path.Combine(onnx, name));
        }

        File.WriteAllText(Path.Combine(_root, "modules.json"), """[{"idx":0,"name":"0","path":"","type":"sentence_transformers.models.Transformer"}]""");
        var config = File.ReadAllText(Path.Combine(CachedOnnx, "config.json"));
        if (!declareRepository)
            config = config.Replace("\"intfloat/multilingual-e5-small\"", "\"/local/copy\"", StringComparison.Ordinal);
        File.WriteAllText(Path.Combine(_root, "config.json"), config);
    }
}
