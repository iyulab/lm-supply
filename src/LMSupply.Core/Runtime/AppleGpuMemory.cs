using System.Runtime.InteropServices;
using System.Text;

namespace LMSupply.Runtime;

/// <summary>
/// How much memory the GPU of an Apple Silicon Mac may use. Its memory is the system's own (unified),
/// so there is no dedicated amount to read; Metal states the budget the GPU should keep its working set
/// within, which is what llama.cpp and other runtimes size models against.
/// </summary>
internal static class AppleGpuMemory
{
    private const string Metal = "/System/Library/Frameworks/Metal.framework/Metal";
    private const string ObjC = "/usr/lib/libobjc.A.dylib";

    /// <summary>
    /// Metal's <c>recommendedMaxWorkingSetSize</c> for the default device, or, where it cannot be read,
    /// the share of system memory macOS lets the GPU wire by default (<see cref="DefaultShare"/>).
    /// <see langword="null"/> off macOS or when neither is known.
    /// </summary>
    public static long? WorkingSetBytes()
    {
        if (!OperatingSystem.IsMacOS())
        {
            return null;
        }

        return RecommendedMaxWorkingSetSize() ?? DefaultShare(GC.GetGCMemoryInfo().TotalAvailableMemoryBytes);
    }

    /// <summary>
    /// The part of <paramref name="systemMemoryBytes"/> macOS lets the GPU use when nothing has changed its
    /// limit: two thirds on a Mac with up to 36 GB, three quarters above. <see langword="null"/> for an
    /// unknown amount.
    /// </summary>
    internal static long? DefaultShare(long systemMemoryBytes)
    {
        if (systemMemoryBytes <= 0)
        {
            return null;
        }

        const long ThirtySixGiB = 36L * 1024 * 1024 * 1024;
        return systemMemoryBytes <= ThirtySixGiB ? systemMemoryBytes / 3 * 2 : systemMemoryBytes / 4 * 3;
    }

    private static long? RecommendedMaxWorkingSetSize()
    {
        try
        {
            var device = MTLCreateSystemDefaultDevice();
            if (device == 0)
            {
                return null;
            }

            try
            {
                var bytes = objc_msgSend_UInt64(device, sel_registerName(Utf8z("recommendedMaxWorkingSetSize")));
                return bytes is > 0 and <= long.MaxValue ? (long)bytes : null;
            }
            finally
            {
                objc_msgSend_Void(device, sel_registerName(Utf8z("release")));
            }
        }
        catch (Exception e) when (e is DllNotFoundException or EntryPointNotFoundException)
        {
            return null;
        }
    }

    private static byte[] Utf8z(string text) => Encoding.UTF8.GetBytes(text + "\0");

    [DllImport(Metal)]
    private static extern nint MTLCreateSystemDefaultDevice();

    [DllImport(ObjC)]
    private static extern nint sel_registerName(byte[] name);

    [DllImport(ObjC, EntryPoint = "objc_msgSend")]
    private static extern ulong objc_msgSend_UInt64(nint receiver, nint selector);

    [DllImport(ObjC, EntryPoint = "objc_msgSend")]
    private static extern void objc_msgSend_Void(nint receiver, nint selector);
}
