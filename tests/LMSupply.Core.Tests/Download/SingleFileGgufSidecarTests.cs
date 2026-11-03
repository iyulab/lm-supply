using AwesomeAssertions;
using LMSupply.Core.Download;

namespace LMSupply.Core.Tests.Download;

/// <summary>
/// A repository of a multimodal model lists a projector (<c>mmproj-*.gguf</c>) beside the model, and the projector can be
/// the largest file of a small model. A loader that opens one file must never choose it.
/// </summary>
public class SingleFileGgufSidecarTests
{
    private static RepoFile File(string path, long size) => new() { Path = path, Type = "file", Size = size };

    [Theory]
    [InlineData("mmproj-model-BF16.gguf", true)]
    [InlineData("MMPROJ-model-Q8_0.gguf", true)]
    [InlineData("sub/mmproj-f16.gguf", true)]
    [InlineData("model-Q8_0.gguf", false)]
    [InlineData("model-mmproj-like-name.gguf", false)]
    public void AProjectorFile_IsRecognizedByItsName(string path, bool expected) =>
        SingleFileGgufDownloader.IsProjectorSidecar(path).Should().Be(expected);

    [Fact]
    public void TheCandidates_LeaveOutTheProjector_EvenWhenItIsTheLargestFile()
    {
        IReadOnlyList<RepoFile> files =
        [
            File("model-BF16.gguf", 558_000_000),
            File("model-Q8_0.gguf", 310_000_000),
            File("mmproj-model-BF16.gguf", 982_000_000),
            File("mmproj-model-Q8_0.gguf", 554_000_000),
            File("README.md", 4_000),
        ];

        SingleFileGgufDownloader.GgufFilesOf(files, "org/model-GGUF").Select(f => f.Path)
            .Should().Equal("model-BF16.gguf", "model-Q8_0.gguf");
    }

    [Fact]
    public void ARepositoryOfAProjectorOnly_HasNoModel() =>
        FluentActions.Invoking(() => SingleFileGgufDownloader.GgufFilesOf([File("mmproj-only.gguf", 1_000)], "org/model-GGUF"))
            .Should().Throw<LMSupply.Exceptions.ModelNotFoundException>();
}
