using AwesomeAssertions;
using LMSupply.Llama.Server;

namespace LMSupply.Llama.Tests.Server;

/// <summary>
/// A consumer-supplied llama-server binary (<see cref="LlamaServerUpdateOptions.ServerBinaryPath"/>)
/// is asked for its build with <c>--version</c>, because every renamed flag is gated on the build
/// number. Both output forms below are copied from real binaries.
/// </summary>
public class LlamaServerBinaryVersionTests
{
    [Theory]
    // b11146 (llama.cpp v0.5.0, 2026-09-23), preceded by the server's own init line.
    [InlineData("0.00.000.708 I srv  llama_server: initializing ...\nversion: 0.5.0-dev (build 11146, commit 7fe450e19)\nbuilt with Clang 20.1.8 for Windows x86_64", "b11146")]
    // b10298 and every build before the versioned-release scheme.
    [InlineData("version: 10298 (15586e2d7)\nbuilt with Clang 20.1.8 for Windows x86_64", "b10298")]
    public void ParseBuildTag_ReadsBothOutputForms(string output, string expected)
    {
        LlamaServerBinaryVersion.ParseBuildTag(output).Should().Be(expected);
    }

    [Theory]
    [InlineData(null)]
    [InlineData("")]
    [InlineData("version: 0.5.0-dev")]
    [InlineData("usage: llama-server [options]")]
    public void ParseBuildTag_NoBuildNumber_IsNull(string? output)
    {
        // "version: 0.5.0-dev" must not be read as build 0.
        LlamaServerBinaryVersion.ParseBuildTag(output).Should().BeNull();
    }

    [Fact]
    public async Task ProbeAsync_FileThatIsNotAnExecutable_IsNull_DoesNotThrow()
    {
        var path = Path.Combine(Path.GetTempPath(), "lmsupply-probe-" + Guid.NewGuid().ToString("N") + (OperatingSystem.IsWindows() ? ".exe" : ""));
        await File.WriteAllBytesAsync(path, "not a program"u8.ToArray(), TestContext.Current.CancellationToken);
        try
        {
            var tag = await LlamaServerBinaryVersion.ProbeAsync(path, TestContext.Current.CancellationToken);
            tag.Should().BeNull();
        }
        finally
        {
            File.Delete(path);
        }
    }
}
