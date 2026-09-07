using System.Formats.Tar;
using System.IO.Compression;
using System.Runtime.InteropServices;

namespace LMSupply.Llama.Tests.Server;

/// <summary>
/// Builds the platform-appropriate fake llama.cpp release asset (name + archive bytes) for the
/// machine the tests run on, so acquisition tests exercise the real asset-pattern and extraction
/// paths without depending on the network or on a specific OS.
/// </summary>
internal static class FakeReleaseAssets
{
    public static (string os, string arch, string ext) PlatformAssetParts()
    {
        var os = OperatingSystem.IsWindows() ? "win" : OperatingSystem.IsMacOS() ? "macos" : "ubuntu";
        var arch = RuntimeInformation.OSArchitecture == Architecture.Arm64 ? "arm64" : "x64";
        var ext = OperatingSystem.IsWindows() ? "zip" : "tar.gz";
        return (os, arch, ext);
    }

    /// <summary>The CPU-backend asset name llama.cpp publishes for the given build on this platform.</summary>
    public static string CpuAssetName(string build)
    {
        var (os, arch, ext) = PlatformAssetParts();
        // Windows CPU assets carry "cpu" in the name; the Linux/macOS CPU asset omits it.
        return OperatingSystem.IsWindows()
            ? $"llama-{build}-bin-{os}-cpu-{arch}.{ext}"
            : $"llama-{build}-bin-{os}-{arch}.{ext}";
    }

    public static string ServerExecutableName => OperatingSystem.IsWindows() ? "llama-server.exe" : "llama-server";

    /// <summary>An archive whose single entry is the server executable, in this platform's archive format.</summary>
    public static byte[] ServerArchive(byte[]? content = null)
    {
        content ??= "not a real binary, just test bytes"u8.ToArray();
        return PlatformAssetParts().ext == "zip"
            ? BuildZip(ServerExecutableName, content)
            : BuildTarGz(ServerExecutableName, content);
    }

    private static byte[] BuildZip(string entryName, byte[] content)
    {
        using var ms = new MemoryStream();
        using (var zip = new ZipArchive(ms, ZipArchiveMode.Create, leaveOpen: true))
        {
            using var entryStream = zip.CreateEntry(entryName).Open();
            entryStream.Write(content);
        }
        return ms.ToArray();
    }

    private static byte[] BuildTarGz(string entryName, byte[] content)
    {
        using var ms = new MemoryStream();
        using (var gzip = new GZipStream(ms, CompressionLevel.Fastest, leaveOpen: true))
        using (var writer = new TarWriter(gzip, TarEntryFormat.Gnu, leaveOpen: true))
        {
            var entry = new GnuTarEntry(TarEntryType.RegularFile, entryName)
            {
                DataStream = new MemoryStream(content)
            };
            writer.WriteEntry(entry);
        }
        return ms.ToArray();
    }
}
