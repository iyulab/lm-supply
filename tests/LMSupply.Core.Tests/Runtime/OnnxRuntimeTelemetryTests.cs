using System.Runtime.CompilerServices;
using AwesomeAssertions;
using LMSupply.Inference;
using LMSupply.Runtime;

namespace LMSupply.Core.Tests.Runtime;

/// <summary>ONNX Runtime's own telemetry is off in a process that uses LMSupply, unless the process chose otherwise.</summary>
public class OnnxRuntimeTelemetryTests
{
    [Fact]
    public void BeforeAnySessionIsMade_TheVariableOnnxRuntimeReadsIsSet_WhereNativeCodeReadsIt()
    {
        RuntimeHelpers.RunClassConstructor(typeof(OnnxSessionFactory).TypeHandle);

        // A process that set the variable itself keeps its value; otherwise LMSupply has set it to 1.
        OnnxRuntimeTelemetry.NativeValue(OnnxRuntimeTelemetry.Variable).Should().NotBeNull("ONNX Runtime reads it from the native environment, not from .NET's copy");
        if (!OnnxRuntimeTelemetry.EnsureOff())
        {
            return;
        }

        OnnxRuntimeTelemetry.NativeValue(OnnxRuntimeTelemetry.Variable).Should().Be("1");
    }

    [Fact]
    public void AnUnsetVariable_IsSetTo1_BothForDotNetAndForNativeCode()
    {
        var variable = Unique();

        OnnxRuntimeTelemetry.TurnOffUnlessSet(variable).Should().BeTrue();

        Environment.GetEnvironmentVariable(variable).Should().Be("1");
        OnnxRuntimeTelemetry.NativeValue(variable).Should().Be("1");
        Clear(variable);
    }

    [Theory]
    [InlineData("0")]
    [InlineData("false")]
    [InlineData("1")]
    public void AVariableTheProcessSet_IsLeftAsItIs(string value)
    {
        var variable = Unique();
        Environment.SetEnvironmentVariable(variable, value);

        OnnxRuntimeTelemetry.TurnOffUnlessSet(variable).Should().BeFalse("the consumer chose");

        Environment.GetEnvironmentVariable(variable).Should().Be(value);
        Clear(variable);
    }

    private static string Unique() => "LMSUPPLY_TELEMETRY_TEST_" + Guid.NewGuid().ToString("N");

    private static void Clear(string variable) => Environment.SetEnvironmentVariable(variable, null);
}
