using System.Text.Json;
using AwesomeAssertions;
using Xunit;

namespace LMSupply.Generator.Tests;

/// <summary>
/// The csproj runs this suite with reflection-based JSON off, like a trimmed/AOT host. This fact is the control: if the
/// switch stopped reaching the test host, every other fact would pass on reflection and prove nothing about AOT safety.
/// </summary>
public class ReflectionFreeJsonGuardTests
{
    [Fact]
    public void Suite_runs_with_reflection_based_serialization_disabled()
        => JsonSerializer.IsReflectionEnabledByDefault.Should().BeFalse();
}
