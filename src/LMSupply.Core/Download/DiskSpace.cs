using LMSupply.Exceptions;

namespace LMSupply.Download;

/// <summary>What a download needs to know of the disk it writes to: how much is free, and whether a write failed for want of it.</summary>
internal static class DiskSpace
{
    // errno values of "no space left on device" and "disk quota exceeded" on Linux and macOS; the Windows
    // HRESULTs of ERROR_DISK_FULL and ERROR_HANDLE_DISK_FULL.
    private static readonly int[] s_fullHResults = [28, 69, 122, unchecked((int)0x80070070), unchecked((int)0x80070027)];

    /// <summary>
    /// The free bytes of the volume that holds <paramref name="path"/> — for the user the process runs as, so a quota
    /// counts — or null when it cannot be told (a path the platform gives no volume for): nothing is then refused.
    /// </summary>
    public static long? FreeBytes(string path)
    {
        try
        {
            var directory = Directory.Exists(path) ? path : Path.GetDirectoryName(Path.GetFullPath(path));
            if (string.IsNullOrEmpty(directory))
                return null;

            // Windows names a drive by its root; elsewhere the mount that holds the directory answers for it.
            var name = OperatingSystem.IsWindows() ? Path.GetPathRoot(Path.GetFullPath(directory)) : directory;
            return string.IsNullOrEmpty(name) ? null : new DriveInfo(name).AvailableFreeSpace;
        }
        catch (Exception e) when (e is ArgumentException or IOException or UnauthorizedAccessException or NotSupportedException)
        {
            return null;
        }
    }

    /// <summary>Whether a write failed because the disk, or the user's share of it, is full.</summary>
    public static bool IsFull(IOException exception) => Array.IndexOf(s_fullHResults, exception.HResult) >= 0;

    /// <summary>Throws when the volume of the destination has fewer free bytes than <paramref name="remainingBytes"/>.</summary>
    public static void EnsureRoom(ResumableFileDownload.Request request, long remainingBytes)
    {
        if (request.FreeBytes(request.DestinationPath) is { } free && free < remainingBytes)
            throw Insufficient(request, remainingBytes, inner: null, free);
    }

    public static InsufficientDiskSpaceException Insufficient(ResumableFileDownload.Request request, long? remainingBytes, Exception? inner, long? free = null)
    {
        free ??= request.FreeBytes(request.DestinationPath);
        var needed = remainingBytes is { } bytes ? $"{bytes:N0} more bytes are needed" : "the file does not fit";
        var available = free is { } left ? $"{left:N0} are free" : "no room is left";
        return new InsufficientDiskSpaceException(
            $"Not enough free disk space for '{request.FileName}': {needed} and {available} on the volume of '{Path.GetDirectoryName(Path.GetFullPath(request.DestinationPath))}'. " +
            "Free space on it, or put the model cache on a larger volume.",
            request.ModelId,
            Path.GetFullPath(request.DestinationPath),
            remainingBytes,
            free,
            inner);
    }
}
