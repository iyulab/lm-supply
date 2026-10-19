using System.Runtime.InteropServices;

namespace LMSupply.Runtime;

/// <summary>
/// Keeps ONNX Runtime from sending its own telemetry from the consumer's process.
/// </summary>
/// <remarks>
/// <para>
/// The official ONNX Runtime builds for Linux, macOS and Windows outside Windows apps upload trace
/// events and a persistent device identifier to Microsoft over HTTPS, on by default. LMSupply runs
/// inference in its consumer's process, and that process must not reach out to a third party it never
/// chose. ONNX Runtime reads <c>ORT_DISABLE_TELEMETRY</c> once, when its environment is created, and with
/// it set creates no uploader, events or identifier at all — so it is set before LMSupply first
/// creates a session or touches the environment, and left alone when the process has already set it
/// (to any value, <c>0</c> included: that is the consumer's choice).
/// </para>
/// <para>
/// The variable has to reach the native environment. On Windows ONNX Runtime reads it with
/// <c>GetEnvironmentVariableW</c>, which sees what .NET sets; elsewhere it reads <c>getenv</c>, which
/// does not, so the variable is also set with the C library's <c>setenv</c>.
/// </para>
/// </remarks>
internal static class OnnxRuntimeTelemetry
{
    internal const string Variable = "ORT_DISABLE_TELEMETRY";

    private static readonly Lazy<bool> s_turnedOff = new(() => TurnOffUnlessSet(Variable), LazyThreadSafetyMode.ExecutionAndPublication);

    /// <summary>
    /// Sets <c>ORT_DISABLE_TELEMETRY=1</c> for the process unless it is set already. Call before creating
    /// an ONNX Runtime session or environment; later calls do nothing.
    /// </summary>
    /// <returns>Whether this process's first call set the variable (rather than finding it set).</returns>
    public static bool EnsureOff() => s_turnedOff.Value;

    /// <summary>The work of <see cref="EnsureOff"/> for <paramref name="variable"/>, run on every call — for tests.</summary>
    internal static bool TurnOffUnlessSet(string variable)
    {
        if (Environment.GetEnvironmentVariable(variable) is not null || NativeValue(variable) is not null)
        {
            return false;
        }

        Environment.SetEnvironmentVariable(variable, "1");
        if (!OperatingSystem.IsWindows())
        {
            _ = setenv(Utf8z(variable), Utf8z("1"), 0);
        }

        return true;
    }

    /// <summary>The variable as native code reads it — <c>getenv</c> on Unix, the process block on Windows.</summary>
    internal static string? NativeValue(string variable) =>
        OperatingSystem.IsWindows() ? Environment.GetEnvironmentVariable(variable) : Marshal.PtrToStringUTF8(getenv(Utf8z(variable)));

    // The C strings are passed as NUL-terminated UTF-8 bytes.
    private static byte[] Utf8z(string text) => System.Text.Encoding.UTF8.GetBytes(text + "\0");

    [DllImport("libc", EntryPoint = "setenv")]
    private static extern int setenv(byte[] name, byte[] value, int overwrite);

    [DllImport("libc", EntryPoint = "getenv")]
    private static extern nint getenv(byte[] name);
}
