using AwesomeAssertions;
using LMSupply.Llama.Server;

namespace LMSupply.Llama.Tests.Server;

/// <summary>
/// llama.cpp replaced <c>--mmap</c>/<c>--no-mmap</c>/<c>--mlock</c> with a single <c>--load-mode</c>
/// in build b10105 (b10103 is the last release without it; b10104 was never tagged). The old flags
/// still parse on newer builds but log <c>DEPRECATED</c> on every start, and the day they are
/// removed a start that passes them dies on "unknown argument" — the same failure class as the
/// 2026-08 release-layout change that made the binary un-acquirable for weeks. So the translation
/// is gated on the build: new builds get <c>--load-mode</c>, older or unknown builds keep the flags
/// they understand.
/// </summary>
public class LlamaServerLoadModeArgsTests
{
    private const string Modern = "b10105";
    private const string Legacy = "b10103";

    [Fact]
    public void GateBuild_IsTheFirstReleaseThatParsesLoadMode()
    {
        LlamaServerVersionRequirements.GetMinimumBuild("load-mode").Should().Be(10105);
    }

    [Theory]
    [InlineData(null, null, new string[0])]
    [InlineData(true, null, new[] { "--load-mode", "mmap" })]
    [InlineData(true, false, new[] { "--load-mode", "mmap" })]
    [InlineData(false, null, new[] { "--load-mode", "none" })]
    [InlineData(false, false, new[] { "--load-mode", "none" })]
    [InlineData(null, true, new[] { "--load-mode", "mlock" })]
    [InlineData(true, true, new[] { "--load-mode", "mlock" })]
    [InlineData(null, false, new string[0])]
    public void ModernBuild_TranslatesToLoadMode(bool? mmap, bool? mlock, string[] expected)
    {
        LlamaServerProcess.ResolveLoadModeArgs(mmap, mlock, Modern).Should().Equal(expected);
    }

    [Fact]
    public void ModernBuild_MlockWithoutMmap_StillLocks()
    {
        // The new vocabulary has no "lock but do not map" mode: `mlock` means mmap + mlock. Locking
        // is the intent the caller expressed, so that is what survives the translation.
        LlamaServerProcess.ResolveLoadModeArgs(useMemoryMap: false, useMemoryLock: true, Modern)
            .Should().Equal("--load-mode", "mlock");
    }

    [Theory]
    [InlineData(null, null, new string[0])]
    [InlineData(true, null, new[] { "--mmap" })]
    [InlineData(false, null, new[] { "--no-mmap" })]
    [InlineData(null, true, new[] { "--mlock" })]
    [InlineData(true, true, new[] { "--mmap", "--mlock" })]
    [InlineData(false, true, new[] { "--no-mmap", "--mlock" })]
    public void LegacyBuild_KeepsTheFlagsItUnderstands(bool? mmap, bool? mlock, string[] expected)
    {
        LlamaServerProcess.ResolveLoadModeArgs(mmap, mlock, Legacy).Should().Equal(expected);
    }

    [Theory]
    [InlineData(null)]
    [InlineData("")]
    [InlineData("custom-build")]
    public void UnknownBuild_KeepsTheLegacyFlags(string? version)
    {
        // A consumer-provisioned binary of unknown build: the deprecated flags are the only ones
        // guaranteed to parse on both sides of the gate today, and a wrong guess the other way is
        // a fatal "unknown argument" before the port is even open.
        LlamaServerProcess.ResolveLoadModeArgs(true, null, version).Should().Equal("--mmap");
    }

    [Fact]
    public void Build_AtTheGate_UsesLoadMode_AndOneBelowDoesNot()
    {
        LlamaServerProcess.ResolveLoadModeArgs(true, null, "b10105").Should().Equal("--load-mode", "mmap");
        LlamaServerProcess.ResolveLoadModeArgs(true, null, "b10104").Should().Equal("--mmap");
    }
}
