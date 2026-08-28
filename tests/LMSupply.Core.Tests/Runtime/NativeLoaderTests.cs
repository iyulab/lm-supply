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
}
