using AwesomeAssertions;
using LMSupply.Runtime;

namespace LMSupply.Core.Tests.Runtime;

/// <summary>
/// A signed (hardened-runtime) macOS app refuses the ONNX Runtime LMSupply downloads at first use (another team signed it), and
/// every model load then failed with a text about «required system libraries». The message now carries the loader's own
/// explanation and, for that one refusal, the remedy. The macOS load itself cannot run on a Windows or Linux machine, so the
/// loader's texts below are the ones dlopen produces for library validation, and the message is asserted per platform.
/// </summary>
public class NativeLoadFailureMessageTests
{
    private const string SigningRefusal =
        "Unable to load shared library '/Users/app/Library/Caches/LMSupply/runtimes/onnxruntime/cpu/1.31.0/osx-arm64/libonnxruntime.dylib' "
        + "or one of its dependencies. dlopen(/Users/app/Library/Caches/LMSupply/runtimes/onnxruntime/cpu/1.31.0/osx-arm64/libonnxruntime.dylib, 0x0001): "
        + "tried: '/Users/app/Library/Caches/LMSupply/runtimes/onnxruntime/cpu/1.31.0/osx-arm64/libonnxruntime.dylib' "
        + "(code signature in <5C1A0E5E-0B7B-3F54-9F4E-6E0D5D1C7A10> '/Users/app/Library/Caches/LMSupply/runtimes/onnxruntime/cpu/1.31.0/osx-arm64/libonnxruntime.dylib' "
        + "not valid for use in process: mapping process and mapped file (non-platform) have different Team IDs)";

    [Theory]
    [InlineData("dlopen(x, 0x0001): tried: 'x' (code signature in <UUID> 'x' not valid for use in process: library load disallowed by system policy)")]
    [InlineData("mapping process and mapped file (non-platform) have different Team IDs")]
    [InlineData("... library load disallowed by system policy")]
    [InlineData(SigningRefusal)]
    public void CodeSigningRefusals_AreRecognised(string loaderText) =>
        NativeLoadFailureMessage.IsCodeSigningRefusal(loaderText).Should().BeTrue();

    [Theory]
    [InlineData(null)]
    [InlineData("")]
    [InlineData("dlopen(x, 0x0001): tried: 'x' (no such file)")]
    [InlineData("libgomp.so.1: cannot open shared object file: No such file or directory")]
    [InlineData("An attempt was made to load a program with an incorrect format.")]
    public void OtherLoaderTexts_AreNotCodeSigningRefusals(string? loaderText) =>
        NativeLoadFailureMessage.IsCodeSigningRefusal(loaderText).Should().BeFalse();

    [Fact]
    public void MacOs_SigningRefusal_QuotesTheLoaderAndNamesTheRemedy()
    {
        var message = NativeLoadFailureMessage.ForOnnxRuntime(NativeLoadPlatform.MacOS, SigningRefusal);

        message.Should().StartWith("ONNX Runtime native library failed to load.");
        message.Should().Contain("The loader said:").And.Contain("have different Team IDs");
        message.Should().Contain("RuntimeManagerOptions.RuntimeDirectory").And.Contain("ship the native ONNX Runtime with the application");
        message.Should().Contain("com.apple.security.cs.disable-library-validation");
        message.Should().NotContain("Ensure required system libraries", "the generic hint points at the wrong fix for a signing refusal");
    }

    [Fact]
    public void MacOs_AnotherReason_KeepsTheLoaderTextButNoSigningRemedy()
    {
        var message = NativeLoadFailureMessage.ForOnnxRuntime(NativeLoadPlatform.MacOS, "dlopen(x, 0x0001): tried: 'x' (no such file)");

        message.Should().Contain("The loader said: dlopen(x, 0x0001)");
        message.Should().NotContain("RuntimeDirectory", "the signing remedy belongs to the signing refusal only");
        message.Should().Contain("Ensure required system libraries are installed.");
    }

    [Theory]
    [InlineData("Linux")]
    [InlineData("Windows")]
    public void OffMacOs_ASigningText_GetsNoSigningRemedy(string platform)
    {
        // The remedy is macOS library validation's; the same words on another platform are only the loader's text.
        NativeLoadFailureMessage.ForOnnxRuntime(Enum.Parse<NativeLoadPlatform>(platform), SigningRefusal).Should().NotContain("RuntimeDirectory");
    }

    [Theory]
    [InlineData("Windows", "This usually means Visual C++ Redistributable 2015-2022 is not installed. Download from: https://aka.ms/vs/17/release/vc_redist.x64.exe")]
    [InlineData("Linux", "Ensure required shared libraries are installed (libstdc++, libgomp).")]
    [InlineData("MacOS", "Ensure required system libraries are installed.")]
    [InlineData("Other", "Ensure required system libraries are installed.")]
    public void WithoutALoaderText_TheMessageIsTheOneItAlwaysWas(string platform, string hint) =>
        NativeLoadFailureMessage.ForOnnxRuntime(Enum.Parse<NativeLoadPlatform>(platform), null)
            .Should().Be($"ONNX Runtime native library failed to load. {hint}");

    [Fact]
    public void ARefusedLoad_IsRecordedByTheLoader_AsOneLine()
    {
        // A file that is not a native library: the loader refuses it on every platform, and its explanation must be kept.
        var name = $"nativeloader_refused_{Guid.NewGuid():N}";
        var dir = Directory.CreateTempSubdirectory().FullName;
        try
        {
            var path = Path.Combine(dir, $"{name}.bin");
            File.WriteAllText(path, "this is not a native library");

            NativeLoader.Instance.GetLastLoadFailure(name).Should().BeNull("nothing was refused yet");

            NativeLoader.Instance.RegisterLibrary(name, path);
            NativeLoader.Instance.TryLoad(name, out _).Should().BeFalse();

            var reason = NativeLoader.Instance.GetLastLoadFailure(name);
            reason.Should().NotBeNullOrWhiteSpace();
            reason.Should().NotContain("\n").And.NotContain("\r", "one line is what an error message wants");
        }
        finally
        {
            Directory.Delete(dir, recursive: true);
        }
    }
}
