using AwesomeAssertions;
using LMSupply.Runtime;

namespace LMSupply.Core.Tests.Runtime;

public class NativeLoaderTests
{
    [Fact]
    public void IsLoaded_UnregisteredLibrary_ReturnsFalse()
    {
        var result = NativeLoader.Instance.IsLoaded("nonexistent_library_xyz");
        result.Should().BeFalse();
    }

    [Fact]
    public void IsLoaded_RegisteredButNotPreloaded_ReturnsFalse()
    {
        NativeLoader.Instance.RegisterLibrary("fake_test_lib", "/nonexistent/path/fake.dll");
        var result = NativeLoader.Instance.IsLoaded("fake_test_lib");
        result.Should().BeFalse();
    }

    [Fact]
    public void GetLoadedPath_UnloadedLibrary_ReturnsNull()
    {
        var result = NativeLoader.Instance.GetLoadedPath($"nonexistent_library_{Guid.NewGuid():N}");
        result.Should().BeNull();
    }

    [Fact]
    public void RegisterDirectory_SameLibraryNameDifferentPath_FirstLoadedBinaryStaysResident()
    {
        // Reproduces docket iyulab/lm-supply#151's cause #2: two different directories each
        // register a native library under the same name (e.g. two different onnxruntime
        // providers' "onnxruntime.dll" both preloaded during the same process's lifetime).
        // Only the first-loaded binary should ever become resident, and GetLoadedPath should
        // report exactly which one won -- not just whether *a* binary is loaded.
        if (!OperatingSystem.IsWindows())
            return; // Native DLL used for the real load is Windows-only.

        var libraryName = $"nativeloader_test_{Guid.NewGuid():N}";
        var dirA = Directory.CreateTempSubdirectory().FullName;
        var dirB = Directory.CreateTempSubdirectory().FullName;
        try
        {
            var systemDll = Path.Combine(Environment.SystemDirectory, "kernel32.dll");
            var pathA = Path.Combine(dirA, $"{libraryName}.dll");
            var pathB = Path.Combine(dirB, $"{libraryName}.dll");
            File.Copy(systemDll, pathA);
            File.Copy(systemDll, pathB);

            NativeLoader.Instance.RegisterDirectory(dirA, preload: true, primaryLibrary: libraryName);
            NativeLoader.Instance.RegisterDirectory(dirB, preload: true, primaryLibrary: libraryName);

            NativeLoader.Instance.GetLoadedPath(libraryName).Should().Be(pathA,
                "the first-loaded binary for a given library name stays resident; the second " +
                "directory's copy must not silently replace it");
            NativeLoader.Instance.IsLoaded(libraryName).Should().BeTrue();
        }
        finally
        {
            // dirA's copy was actually loaded via NativeLibrary.TryLoad and is never unloaded
            // (NativeLoader.Instance is a process-wide singleton with no unload path for
            // individual libraries), so Windows keeps the file locked for the rest of the
            // process's lifetime -- best-effort cleanup only, same as production behavior.
            try { Directory.Delete(dirA, recursive: true); } catch (IOException) { }
            catch (UnauthorizedAccessException) { }
            Directory.Delete(dirB, recursive: true);
        }
    }
}
