using AwesomeAssertions;
using LMSupply.Core.Download;

namespace LMSupply.Core.Tests.Download;

/// <summary>
/// The containment check every cache lookup and download makes: the same directory written in any form contains the
/// same files, and nothing outside it — a sibling sharing its name as a prefix, a path climbing out — is let through.
/// </summary>
public sealed class PathContainmentTests
{
    private static readonly string Root = Path.Combine(Path.GetTempPath(), "lmsupply-contain", "cache");

    [Fact]
    public void AFileBelowTheDirectory_IsInside_WhicheverWayTheDirectoryIsWritten()
    {
        var file = Path.Combine(Root, "models--a--b", "snapshots", "main", "model.onnx");

        PathContainment.IsInside(Root, file).Should().BeTrue();
        PathContainment.IsInside(Root + Path.DirectorySeparatorChar, file).Should().BeTrue();
        PathContainment.IsInside(Path.Combine(Path.GetDirectoryName(Root)!, ".", "cache"), file).Should().BeTrue();
        PathContainment.IsInside(Root.Replace(Path.DirectorySeparatorChar, '/'), file).Should().BeTrue();
    }

    [Fact]
    public void ASiblingSharingTheDirectorysNameAsAPrefix_IsNotInside()
    {
        PathContainment.IsInside(Root, Root + "-old" + Path.DirectorySeparatorChar + "model.onnx").Should().BeFalse();
    }

    [Fact]
    public void TheDirectoryItself_IsNotInsideItself()
    {
        PathContainment.IsInside(Root, Root).Should().BeFalse();
        PathContainment.IsInside(Root, Root + Path.DirectorySeparatorChar).Should().BeFalse();
    }

    [Fact]
    public void Resolve_KeepsARepositoryPath_AndRefusesOneThatClimbsOut()
    {
        PathContainment.Resolve(Root, "onnx/model.onnx")
            .Should().Be(Path.Combine(Root, "onnx", "model.onnx"));
        PathContainment.Resolve(Root, "onnx/../tokenizer.json")
            .Should().Be(Path.Combine(Root, "tokenizer.json"));
        PathContainment.Resolve(Root, "../../etc/passwd").Should().BeNull();
        PathContainment.Resolve(Root, "../cache-old/model.onnx").Should().BeNull();
    }

    [Fact]
    public void ADifferentlyCasedDirectory_IsInsideOnlyWhereTheFileSystemIgnoresCase()
    {
        var climbed = PathContainment.Resolve(Root, "../CACHE/model.onnx");

        if (OperatingSystem.IsWindows())
            climbed.Should().NotBeNull();
        else
            climbed.Should().BeNull("outside Windows, CACHE and cache are two directories");
    }

    [Fact]
    public void AVolumeRoot_ContainsTheFilesOnIt()
    {
        var volume = Path.GetPathRoot(Root)!;

        PathContainment.IsInside(volume, Root).Should().BeTrue();
        PathContainment.IsInside(volume, volume).Should().BeFalse();
    }
}
