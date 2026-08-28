using AwesomeAssertions;
using LMSupply.Generator.Internal.Llama;

namespace LMSupply.Generator.Tests;

public class GgufModelDownloaderLocalCacheTests : IDisposable
{
    private readonly string _tempDir = Path.Combine(Path.GetTempPath(), "lmsupply_cache_test_" + Guid.NewGuid().ToString("N"));

    private GgufModelDownloader CreateDownloader() => new(_tempDir);

    private string CreateCacheFile(string repoId, string filename)
    {
        // Mirrors GgufModelDownloader.GetCachedPath: replace '/' with OS separator, then replace separator with '--'
        var safeRepo = "models--" + repoId.Replace('/', Path.DirectorySeparatorChar)
            .Replace(Path.DirectorySeparatorChar.ToString(), "--");
        var dir = Path.Combine(_tempDir, safeRepo, "snapshots", "main");
        Directory.CreateDirectory(dir);
        var path = Path.Combine(dir, filename);
        File.WriteAllText(path, "placeholder");
        return path;
    }

    [Fact]
    public void TrySelectFromLocalCache_EmptyCache_ReturnsNull()
    {
        using var dl = CreateDownloader();
        dl.TrySelectFromLocalCache("bartowski/TestModel-GGUF", null).Should().BeNull();
    }

    [Fact]
    public void TrySelectFromLocalCache_CachedFile_ReturnsFilename()
    {
        CreateCacheFile("bartowski/TestModel-GGUF", "TestModel-Q4_K_M.gguf");
        using var dl = CreateDownloader();

        var result = dl.TrySelectFromLocalCache("bartowski/TestModel-GGUF", null);

        result.Should().Be("TestModel-Q4_K_M.gguf");
    }

    [Fact]
    public void TrySelectFromLocalCache_PreferredQuantization_ReturnsMatchingFile()
    {
        CreateCacheFile("bartowski/TestModel-GGUF", "TestModel-Q4_K_M.gguf");
        CreateCacheFile("bartowski/TestModel-GGUF", "TestModel-Q8_0.gguf");
        using var dl = CreateDownloader();

        var result = dl.TrySelectFromLocalCache("bartowski/TestModel-GGUF", "Q8_0");

        result.Should().Be("TestModel-Q8_0.gguf");
    }

    [Fact]
    public void TrySelectFromLocalCache_SkipsMmprojFiles()
    {
        CreateCacheFile("bartowski/TestModel-GGUF", "mmproj-TestModel-bf16.gguf");
        CreateCacheFile("bartowski/TestModel-GGUF", "TestModel-Q4_K_M.gguf");
        using var dl = CreateDownloader();

        var result = dl.TrySelectFromLocalCache("bartowski/TestModel-GGUF", null);

        result.Should().Be("TestModel-Q4_K_M.gguf");
    }

    [Fact]
    public void TrySelectFromLocalCache_OnlyMmprojFiles_ReturnsNull()
    {
        CreateCacheFile("bartowski/TestModel-GGUF", "mmproj-TestModel-bf16.gguf");
        using var dl = CreateDownloader();

        var result = dl.TrySelectFromLocalCache("bartowski/TestModel-GGUF", null);

        result.Should().BeNull();
    }

    [Fact]
    public void TrySelectFromLocalCache_FallsBackToDefaultPriority_WhenNoPreferred()
    {
        CreateCacheFile("bartowski/TestModel-GGUF", "TestModel-Q8_0.gguf");
        CreateCacheFile("bartowski/TestModel-GGUF", "TestModel-Q4_K_M.gguf");
        using var dl = CreateDownloader();

        // Q4_K_M has higher default priority than Q8_0
        var result = dl.TrySelectFromLocalCache("bartowski/TestModel-GGUF", null);

        result.Should().Be("TestModel-Q4_K_M.gguf");
    }

    public void Dispose()
    {
        if (Directory.Exists(_tempDir))
            Directory.Delete(_tempDir, recursive: true);
        GC.SuppressFinalize(this);
    }
}

public class GgufModelDownloaderTests
{
    [Fact]
    public void GenerateShardFilenames_ThreeShards_GeneratesCorrectNames()
    {
        var first = "Q4_K_M/Qwen3.5-122B-A10B-Q4_K_M-00001-of-00003.gguf";

        var result = GgufModelDownloader.GenerateShardFilenames(first, 3);

        result.Should().HaveCount(3);
        result[0].Should().Be("Q4_K_M/Qwen3.5-122B-A10B-Q4_K_M-00001-of-00003.gguf");
        result[1].Should().Be("Q4_K_M/Qwen3.5-122B-A10B-Q4_K_M-00002-of-00003.gguf");
        result[2].Should().Be("Q4_K_M/Qwen3.5-122B-A10B-Q4_K_M-00003-of-00003.gguf");
    }

    [Fact]
    public void GenerateShardFilenames_TwoShards_GeneratesCorrectNames()
    {
        var first = "model-Q4_K_M-00001-of-00002.gguf";

        var result = GgufModelDownloader.GenerateShardFilenames(first, 2);

        result.Should().HaveCount(2);
        result[0].Should().Be("model-Q4_K_M-00001-of-00002.gguf");
        result[1].Should().Be("model-Q4_K_M-00002-of-00002.gguf");
    }

    [Fact]
    public void GenerateShardFilenames_NonSplitFile_ReturnsSingle()
    {
        var filename = "model-Q4_K_M.gguf";

        var result = GgufModelDownloader.GenerateShardFilenames(filename, 1);

        result.Should().HaveCount(1);
        result[0].Should().Be(filename);
    }

    [Fact]
    public void GenerateShardFilenames_PreservesSubfolder()
    {
        var first = "subfolder/deep/model-00001-of-00005.gguf";

        var result = GgufModelDownloader.GenerateShardFilenames(first, 5);

        result.Should().HaveCount(5);
        result.Should().AllSatisfy(f => f.Should().StartWith("subfolder/deep/model-"));
        result[4].Should().Be("subfolder/deep/model-00005-of-00005.gguf");
    }

    [Theory]
    [InlineData("mmproj-gemma-4-E4B-it-bf16.gguf", true)]
    [InlineData("mmproj-gemma-4-E4B-it-Q8_0.gguf", true)]
    [InlineData("MMPROJ-model.gguf", true)]
    [InlineData("MmProj-mixed-case.gguf", true)]
    [InlineData("gemma-4-E4B-it-Q4_K_M.gguf", false)]
    [InlineData("gemma-4-E4B-it-Q8_0.gguf", false)]
    [InlineData("model-mmproj-suffix.gguf", false)]
    public void IsMmprojFile_DetectsMmprojPrefix(string filename, bool expected)
    {
        GgufModelDownloader.IsMmprojFile(filename).Should().Be(expected);
    }

    // ggml-org/llama.cpp#24343 — an mtp-* file loaded as the main model crashes llama-server.
    [Theory]
    [InlineData("mtp-gemma-4-E4B-it-BF16.gguf", true)]
    [InlineData("mtp-gemma-4-E4B-it-Q4_0.gguf", true)]
    [InlineData("MTP-model.gguf", true)]
    [InlineData("Mtp-mixed-case.gguf", true)]
    [InlineData("gemma-4-E4B-it-Q4_0.gguf", false)]
    [InlineData("gemma-4-E4B-it-Q8_0.gguf", false)]
    [InlineData("model-mtp-suffix.gguf", false)]
    public void IsMtpFile_DetectsMtpPrefix(string filename, bool expected)
    {
        GgufModelDownloader.IsMtpFile(filename).Should().Be(expected);
    }

    // ggml-org/llama.cpp#22105 — DFlash block-diffusion draft companion, same class as mtp-*.
    [Theory]
    [InlineData("dflash-gemma-4-26B-A4B-it-BF16.gguf", true)]
    [InlineData("dflash-gemma-4-31B-it-Q8_0.gguf", true)]
    [InlineData("DFLASH-model.gguf", true)]
    [InlineData("Dflash-mixed-case.gguf", true)]
    [InlineData("gemma-4-26B-A4B-it-Q4_0.gguf", false)]
    [InlineData("model-dflash-suffix.gguf", false)]
    public void IsDflashFile_DetectsDflashPrefix(string filename, bool expected)
    {
        GgufModelDownloader.IsDflashFile(filename).Should().Be(expected);
    }

    [Theory]
    [InlineData("mmproj-model.gguf", true)]
    [InlineData("mtp-model.gguf", true)]
    [InlineData("dflash-model.gguf", true)]
    [InlineData("gemma-4-E4B-it-Q4_0.gguf", false)]
    public void IsCompanionFile_MatchesAnyKnownCompanionPrefix(string filename, bool expected)
    {
        GgufModelDownloader.IsCompanionFile(filename).Should().Be(expected);
    }
}
