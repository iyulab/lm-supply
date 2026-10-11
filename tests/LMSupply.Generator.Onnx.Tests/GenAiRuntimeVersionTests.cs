using AwesomeAssertions;

namespace LMSupply.Generator.Onnx.Tests;

/// <summary>
/// The native GenAI runtime is provisioned at the version of the managed GenAI package the application resolved — not
/// ONNX Runtime's version and not the newest on the feed. The managed assembly itself carries no version (0.0.0.0).
/// </summary>
public class GenAiRuntimeVersionTests
{
    [Fact]
    public void The_managed_GenAI_assembly_carries_no_version_of_its_own()
    {
        // Positive control for why the version comes from deps.json and the build stamp, not from the assembly
        typeof(Microsoft.ML.OnnxRuntimeGenAI.Model).Assembly.GetName().Version!.Major.Should().Be(0);
    }

    [Fact]
    public void The_resolved_version_is_read_from_this_applications_deps_json()
    {
        var resolved = OnnxGeneratorBackend.ReadResolvedGenAiVersion();

        resolved.Should().NotBeNull();
        resolved.Should().StartWith("0.", "GenAI is 0.x; ONNX Runtime's 1.x is a different version line");
        OnnxGeneratorBackend.GenAiVersion.Should().Be(resolved);
    }

    [Fact]
    public void The_build_stamp_matches_what_this_build_resolved()
    {
        OnnxGeneratorBackend.ReadBuildGenAiVersion().Should().Be(OnnxGeneratorBackend.ReadResolvedGenAiVersion());
    }
}
