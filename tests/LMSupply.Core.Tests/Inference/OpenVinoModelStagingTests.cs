using AwesomeAssertions;
using LMSupply.Inference;

namespace LMSupply.Core.Tests.Inference;

/// <summary>
/// The OpenVINO plugin refuses an external-data file that resolves outside the model's directory (the Hugging Face cache
/// links <c>snapshots/&lt;rev&gt;/model.onnx_data</c> to <c>blobs/&lt;sha&gt;</c>). <see cref="OpenVinoModelStaging"/> hands it
/// the files inside a directory instead.
/// </summary>
public sealed class OpenVinoModelStagingTests : IDisposable
{
    private readonly DirectoryInfo _root = Directory.CreateTempSubdirectory("lmsupply-ovstage-");

    public void Dispose() => _root.Delete(recursive: true);

    private string Blob(string name, string content)
    {
        var blobs = Directory.CreateDirectory(Path.Combine(_root.FullName, "blobs")).FullName;
        var path = Path.Combine(blobs, name);
        File.WriteAllText(path, content);
        return path;
    }

    private string Snapshot() => Directory.CreateDirectory(Path.Combine(_root.FullName, "snapshots", "rev", "onnx")).FullName;

    private static void Link(string linkPath, string target)
    {
        try
        {
            File.CreateSymbolicLink(linkPath, target);
        }
        catch (Exception ex) when (ex is UnauthorizedAccessException or IOException or PlatformNotSupportedException)
        {
            Assert.Skip($"Symbolic links cannot be created here ({ex.GetType().Name}).");
        }
    }

    [Fact]
    public void LinkedExternalData_IsStagedAsFilesInsideTheModelDirectory()
    {
        var snapshot = Snapshot();
        Link(Path.Combine(snapshot, "model.onnx"), Blob("aaa", "graph"));
        Link(Path.Combine(snapshot, "model.onnx_data"), Blob("bbb", "weights-weights"));
        Link(Path.Combine(snapshot, "tokenizer.json"), Blob("ccc", "{}"));
        var modelPath = Path.Combine(snapshot, "model.onnx");

        var staged = OpenVinoModelStaging.Resolve(modelPath);

        staged.Should().NotBe(modelPath);
        Path.GetDirectoryName(staged).Should().Be(Path.Combine(snapshot, OpenVinoModelStaging.StagingDirectoryName));
        Path.GetFileName(staged).Should().Be("model.onnx");
        var data = Path.Combine(Path.GetDirectoryName(staged)!, "model.onnx_data");
        File.ReadAllText(data).Should().Be("weights-weights");
        File.ReadAllText(staged).Should().Be("graph");
        new FileInfo(data).LinkTarget.Should().BeNull("a staged file is a hard link or a copy, never a link out of the directory");
        new FileInfo(staged).LinkTarget.Should().BeNull();
        File.Exists(Path.Combine(Path.GetDirectoryName(staged)!, "tokenizer.json")).Should().BeFalse(
            "only the files that carry the model's name are staged");
    }

    [Fact]
    public void RelativeLinks_AsTheHuggingFaceCacheWritesThem_AreStagedAsFilesInsideTheModelDirectory()
    {
        // The real cache links with relative targets (../../blobs/<sha>), not absolute ones.
        var snapshot = Snapshot();
        Blob("aaa", "graph");
        Blob("bbb", "weights-weights");
        Link(Path.Combine(snapshot, "model.onnx"), Path.Combine("..", "..", "..", "blobs", "aaa"));
        Link(Path.Combine(snapshot, "model.onnx_data"), Path.Combine("..", "..", "..", "blobs", "bbb"));
        var modelPath = Path.Combine(snapshot, "model.onnx");

        var staged = OpenVinoModelStaging.Resolve(modelPath);

        staged.Should().NotBe(modelPath);
        var data = Path.Combine(Path.GetDirectoryName(staged)!, "model.onnx_data");
        File.ReadAllText(data).Should().Be("weights-weights");
        File.ReadAllText(staged).Should().Be("graph");
        new FileInfo(data).LinkTarget.Should().BeNull();
    }

    [Fact]
    public void StagedFiles_AreHardLinks_NotCopies_WhereTheFileSystemAllowsIt()
    {
        // A copy of multi-gigabyte weights would work but is exactly what the hard link avoids: a write through the blob
        // is visible through the staged file only when both names are one file.
        var snapshot = Snapshot();
        var blob = Blob("bbb", "weights");
        Link(Path.Combine(snapshot, "model.onnx"), Blob("aaa", "graph"));
        Link(Path.Combine(snapshot, "model.onnx_data"), blob);

        var staged = OpenVinoModelStaging.Resolve(Path.Combine(snapshot, "model.onnx"));
        File.AppendAllText(blob, "-and-more");

        File.ReadAllText(Path.Combine(Path.GetDirectoryName(staged)!, "model.onnx_data")).Should().Be("weights-and-more");
    }

    [Fact]
    public void Staging_IsRepeatable_AndKeepsTheSamePath()
    {
        var snapshot = Snapshot();
        Link(Path.Combine(snapshot, "model.onnx"), Blob("aaa", "graph"));
        Link(Path.Combine(snapshot, "model.onnx_data"), Blob("bbb", "weights"));
        var modelPath = Path.Combine(snapshot, "model.onnx");

        var first = OpenVinoModelStaging.Resolve(modelPath);
        var second = OpenVinoModelStaging.Resolve(modelPath);

        second.Should().Be(first);
        File.ReadAllText(Path.Combine(Path.GetDirectoryName(first)!, "model.onnx_data")).Should().Be("weights");
    }

    [Fact]
    public void ExternalDataInsideTheModelDirectory_NeedsNoStaging()
    {
        var snapshot = Snapshot();
        File.WriteAllText(Path.Combine(snapshot, "model.onnx"), "graph");
        File.WriteAllText(Path.Combine(snapshot, "model.onnx_data"), "weights");
        var modelPath = Path.Combine(snapshot, "model.onnx");

        OpenVinoModelStaging.Resolve(modelPath).Should().Be(modelPath);
        Directory.Exists(Path.Combine(snapshot, OpenVinoModelStaging.StagingDirectoryName)).Should().BeFalse();
    }

    [Fact]
    public void ALinkedModelFileWithoutExternalData_NeedsNoStaging()
    {
        var snapshot = Snapshot();
        Link(Path.Combine(snapshot, "model.onnx"), Blob("aaa", "graph"));
        var modelPath = Path.Combine(snapshot, "model.onnx");

        OpenVinoModelStaging.Resolve(modelPath).Should().Be(modelPath);
    }

    [Fact]
    public void AMissingModelFile_IsReturnedUnchanged()
    {
        var modelPath = Path.Combine(_root.FullName, "nope", "model.onnx");

        OpenVinoModelStaging.Resolve(modelPath).Should().Be(modelPath);
    }
}
