using System.Diagnostics;
using System.Runtime.InteropServices;
using LMSupply.Download;

namespace LMSupply.Inference;

/// <summary>
/// Makes a model's external-data files readable by the OpenVINO execution provider plugin. The plugin refuses an
/// external-data file whose real path lies outside the directory of the model it compiles (a link out of that directory
/// fails with «resolves to ... which is outside the base directory»), and the session then ends on the CPU provider.
/// The Hugging Face cache layout is exactly that: a file is stored once under <c>blobs/</c> and shown in
/// <c>snapshots/&lt;revision&gt;/...</c> as a link. This helper puts the model file and the files that carry its name
/// (<c>model.onnx_data</c>, <c>model.onnx.data</c>) into a directory next to them as hard links (a copy when the file
/// system cannot link), and the session is created from there.
/// </summary>
internal static class OpenVinoModelStaging
{
    /// <summary>The directory, inside the model's own directory, that holds the staged files.</summary>
    internal const string StagingDirectoryName = ".lmsupply-openvino";

    /// <summary>
    /// The path to create an OpenVINO session from: <paramref name="modelPath"/> itself unless a file that carries the
    /// model's name resolves outside the model's directory, otherwise the staged copy of the model file. Never throws for
    /// an I/O problem: the caller keeps the original path (and the plugin's own refusal) when staging is not possible.
    /// </summary>
    /// <param name="modelPath">Path to the ONNX model file.</param>
    internal static string Resolve(string modelPath)
    {
        try
        {
            return ResolveCore(modelPath);
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException or NotSupportedException)
        {
            Trace.TraceWarning(
                $"[OpenVINO] Could not stage the external-data files of '{modelPath}' inside the model directory: {ex.Message}. " +
                "The OpenVINO plugin may refuse the model and the session may run on the CPU provider.");
            return modelPath;
        }
    }

    private static string ResolveCore(string modelPath)
    {
        var fullPath = Path.GetFullPath(modelPath);
        var directory = Path.GetDirectoryName(fullPath);
        if (directory is null || !File.Exists(fullPath))
            return modelPath;

        var modelName = Path.GetFileName(fullPath);
        var files = new List<(string Name, string Source)>();
        var escapes = false;
        foreach (var file in Directory.EnumerateFiles(directory))
        {
            var name = Path.GetFileName(file);
            if (!name.StartsWith(modelName, StringComparison.OrdinalIgnoreCase))
                continue;

            var source = RealPath(file);
            files.Add((name, source));
            // The model file itself may be a link too; only a linked external-data file makes the plugin refuse.
            if (!string.Equals(name, modelName, StringComparison.OrdinalIgnoreCase) && !IsInside(directory, source))
                escapes = true;
        }

        if (!escapes)
            return modelPath;

        var stagingDirectory = Path.Combine(directory, StagingDirectoryName);
        Directory.CreateDirectory(stagingDirectory);
        foreach (var (name, source) in files)
            Stage(source, Path.Combine(stagingDirectory, name));

        return Path.Combine(stagingDirectory, modelName);
    }

    private static string RealPath(string file)
        => File.ResolveLinkTarget(file, returnFinalTarget: true)?.FullName ?? Path.GetFullPath(file);

    private static bool IsInside(string directory, string path)
    {
        var prefix = directory.EndsWith(Path.DirectorySeparatorChar) ? directory : directory + Path.DirectorySeparatorChar;
        var comparison = OperatingSystem.IsWindows() ? StringComparison.OrdinalIgnoreCase : StringComparison.Ordinal;
        return Path.GetFullPath(path).StartsWith(prefix, comparison);
    }

    private static void Stage(string source, string destination)
    {
        var sourceLength = CacheManager.GetContentLength(source);
        if (File.Exists(destination))
        {
            // Already staged by an earlier load (a hard link shares the size; a copy has it too).
            if (CacheManager.GetContentLength(destination) == sourceLength)
                return;
            File.Delete(destination);
        }

        if (TryHardLink(source, destination) || File.Exists(destination))
            return;

        Trace.TraceInformation($"[OpenVINO] Hard link not possible for '{source}'; copying it next to the model ({sourceLength:N0} bytes).");
        File.Copy(source, destination, overwrite: true);
    }

    private static bool TryHardLink(string existing, string link)
    {
        try
        {
            if (OperatingSystem.IsWindows())
                return CreateHardLinkW(link, existing, IntPtr.Zero);
            if (OperatingSystem.IsLinux() || OperatingSystem.IsMacOS())
                return LinkUnix(existing, link) == 0;
        }
        catch (Exception ex) when (ex is DllNotFoundException or EntryPointNotFoundException)
        {
            // No such entry point on this platform: the copy fallback applies.
        }

        return false;
    }

    [DllImport("kernel32.dll", SetLastError = true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    private static extern bool CreateHardLinkW(
        [MarshalAs(UnmanagedType.LPWStr)] string lpFileName,
        [MarshalAs(UnmanagedType.LPWStr)] string lpExistingFileName,
        IntPtr lpSecurityAttributes);

    // CharSet.Ansi marshals as UTF-8 on Unix; best-fit mapping off so a path is never silently altered.
    [DllImport("libc", EntryPoint = "link", SetLastError = true, CharSet = CharSet.Ansi, BestFitMapping = false, ThrowOnUnmappableChar = true)]
    private static extern int LinkUnix(string existingPath, string newPath);
}
