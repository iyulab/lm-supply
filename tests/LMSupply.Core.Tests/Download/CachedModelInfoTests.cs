using AwesomeAssertions;
using LMSupply.Download;

namespace LMSupply.Core.Tests.Download;

public class CachedModelInfoTests
{
    [Fact]
    public void DefaultValues_ShouldBeCorrect()
    {
        var info = new CachedModelInfo
        {
            RepoId = "test/model",
            LocalPath = "/models/test"
        };

        info.SizeBytes.Should().Be(0);
        info.FileCount.Should().Be(0);
        info.DetectedType.Should().Be(ModelType.Unknown);
        info.Files.Should().BeEmpty();
        info.Metadata.Should().BeNull();
    }

    [Fact]
    public void SizeMB_ShouldCalculateCorrectly()
    {
        var info = new CachedModelInfo
        {
            RepoId = "test/model",
            LocalPath = "/models/test",
            SizeBytes = 10 * 1024 * 1024
        };

        info.SizeMB.Should().Be(10.0);
    }

    [Theory]
    [InlineData(0, 0.0)]
    [InlineData(1024, 1024.0 / (1024.0 * 1024.0))]
    [InlineData(1024 * 1024, 1.0)]
    [InlineData(5368709120L, 5120.0)] // 5 GB
    public void SizeMB_ShouldHandleVariousSizes(long sizeBytes, double expectedMB)
    {
        var info = new CachedModelInfo
        {
            RepoId = "test/model",
            LocalPath = "/path",
            SizeBytes = sizeBytes
        };

        info.SizeMB.Should().BeApproximately(expectedMB, 0.001);
    }

    [Theory]
    [InlineData(0, false)]
    [InlineData(1024 * 1024 - 1, false)]
    [InlineData(1024 * 1024, true)]
    [InlineData(1024 * 1024 + 1, true)]
    [InlineData(5368709120L, true)]
    public void IsComplete_ShouldCheckMinimumSize(long sizeBytes, bool expected)
    {
        var info = new CachedModelInfo
        {
            RepoId = "test/model",
            LocalPath = "/path",
            SizeBytes = sizeBytes
        };

        info.IsComplete.Should().Be(expected);
    }

    [Fact]
    public void MinimumCompleteSize_ShouldBeOneMB()
    {
        CachedModelInfo.MinimumCompleteSize.Should().Be(1024 * 1024);
    }

    [Fact]
    public void HasMetadata_ShouldBeFalse_WhenMetadataIsNull()
    {
        var info = new CachedModelInfo
        {
            RepoId = "test/model",
            LocalPath = "/path"
        };

        info.HasMetadata.Should().BeFalse();
    }

    [Fact]
    public void HasMetadata_ShouldBeTrue_WhenMetadataIsSet()
    {
        var info = new CachedModelInfo
        {
            RepoId = "test/model",
            LocalPath = "/path",
            Metadata = new ModelMetadata()
        };

        info.HasMetadata.Should().BeTrue();
    }

    [Fact]
    public void Files_ShouldPreserveList()
    {
        var files = new[] { "model.onnx", "config.json", "tokenizer.json" };
        var info = new CachedModelInfo
        {
            RepoId = "test/model",
            LocalPath = "/path",
            Files = files
        };

        info.Files.Should().HaveCount(3);
        info.Files.Should().Contain("model.onnx");
    }

    [Fact]
    public void RecordEquality_SameValues_ShouldBeEqual()
    {
        var a = new CachedModelInfo { RepoId = "test/model", LocalPath = "/path", SizeBytes = 100 };
        var b = new CachedModelInfo { RepoId = "test/model", LocalPath = "/path", SizeBytes = 100 };

        a.Should().Be(b);
    }

    [Fact]
    public void RecordEquality_DifferentValues_ShouldNotBeEqual()
    {
        var a = new CachedModelInfo { RepoId = "test/model", LocalPath = "/path", SizeBytes = 100 };
        var b = new CachedModelInfo { RepoId = "test/model", LocalPath = "/path", SizeBytes = 200 };

        a.Should().NotBe(b);
    }
}

public class ModelMetadataTests
{
    [Fact]
    public void DefaultValues_ShouldBeCorrect()
    {
        var metadata = new ModelMetadata();

        metadata.Description.Should().BeNull();
        metadata.License.Should().BeNull();
        metadata.Author.Should().BeNull();
        metadata.Downloads.Should().Be(0);
        metadata.Likes.Should().Be(0);
        metadata.Tags.Should().BeEmpty();
        metadata.PipelineTag.Should().BeNull();
        metadata.LibraryName.Should().BeNull();
        metadata.LastModifiedOnHub.Should().BeNull();
        metadata.IsGated.Should().BeFalse();
        metadata.IsPrivate.Should().BeFalse();
    }

    [Fact]
    public void FetchedAt_ShouldDefaultToUtcNow()
    {
        var before = DateTime.UtcNow;
        var metadata = new ModelMetadata();
        var after = DateTime.UtcNow;

        metadata.FetchedAt.Should().BeOnOrAfter(before);
        metadata.FetchedAt.Should().BeOnOrBefore(after);
    }

    [Fact]
    public void ShouldInitialize_WithAllFields()
    {
        var tags = new[] { "pytorch", "en", "text-generation" };
        var metadata = new ModelMetadata
        {
            Description = "A test model",
            License = "mit",
            Author = "iyulab",
            Downloads = 50000,
            Likes = 1200,
            Tags = tags,
            PipelineTag = "text-generation",
            LibraryName = "transformers",
            IsGated = true,
            IsPrivate = false
        };

        metadata.Description.Should().Be("A test model");
        metadata.License.Should().Be("mit");
        metadata.Author.Should().Be("iyulab");
        metadata.Downloads.Should().Be(50000);
        metadata.Likes.Should().Be(1200);
        metadata.Tags.Should().HaveCount(3);
        metadata.PipelineTag.Should().Be("text-generation");
        metadata.LibraryName.Should().Be("transformers");
        metadata.IsGated.Should().BeTrue();
        metadata.IsPrivate.Should().BeFalse();
    }

    [Fact]
    public void RecordEquality_ShouldWork()
    {
        var now = DateTime.UtcNow;
        var a = new ModelMetadata { License = "mit", Downloads = 100, FetchedAt = now };
        var b = new ModelMetadata { License = "mit", Downloads = 100, FetchedAt = now };

        a.Should().Be(b); // Same values → equal

        var c = new ModelMetadata { License = "mit", Downloads = 200, FetchedAt = now };
        a.Should().NotBe(c); // Different Downloads → not equal
    }
}

public class ModelTypeEnumTests
{
    [Fact]
    public void ShouldHaveExpectedValues()
    {
        Enum.GetValues<ModelType>().Should().HaveCount(12);
    }

    [Theory]
    [InlineData(ModelType.Unknown, 0)]
    [InlineData(ModelType.Generator, 1)]
    [InlineData(ModelType.Embedder, 2)]
    [InlineData(ModelType.Reranker, 3)]
    [InlineData(ModelType.Transcriber, 4)]
    [InlineData(ModelType.Synthesizer, 5)]
    [InlineData(ModelType.Detector, 6)]
    [InlineData(ModelType.Captioner, 7)]
    [InlineData(ModelType.Translator, 8)]
    [InlineData(ModelType.Ocr, 9)]
    [InlineData(ModelType.Segmenter, 10)]
    [InlineData(ModelType.ImageGenerator, 11)]
    public void ShouldHaveExpectedIntValues(ModelType type, int expected)
    {
        ((int)type).Should().Be(expected);
    }
}
