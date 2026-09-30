using System.Diagnostics;

namespace LMSupply.Download;

/// <summary>
/// Centralized model directory validation.
/// Uses manifest-based validation (primary) or fallback heuristics (legacy/local).
/// </summary>
public static class ModelDirectoryValidator
{
    private const uint GgufMagic = 0x46554747; // "GGUF"

    public record ValidationResult(
        bool IsValid,
        string? Reason = null,
        string[]? MissingFiles = null)
    {
        public static readonly ValidationResult Valid = new(true);
    }

    /// <summary>
    /// Validates a model directory for completeness and integrity.
    /// </summary>
    public static ValidationResult Validate(string directoryPath)
    {
        if (!Directory.Exists(directoryPath))
            return new ValidationResult(false, $"Directory does not exist: {directoryPath}");

        // Check for incomplete downloads (.part files)
        var partFiles = Directory.GetFiles(directoryPath, "*.part");
        if (partFiles.Length > 0)
        {
            var names = partFiles.Select(Path.GetFileName).ToArray();
            return new ValidationResult(false,
                $"Incomplete download detected. Partial files: {string.Join(", ", names)}. " +
                "Delete the cache directory and retry.");
        }

        // Check for LFS pointer files among model files
        foreach (var file in Directory.GetFiles(directoryPath))
        {
            var ext = Path.GetExtension(file);
            if (ext is ".onnx" or ".gguf" || file.EndsWith(".onnx.data", StringComparison.OrdinalIgnoreCase))
            {
                if (CacheManager.IsLfsPointerFile(file))
                    return new ValidationResult(false,
                        $"LFS pointer detected: {Path.GetFileName(file)}. The actual model was not downloaded.");
            }
        }

        // Try manifest-based validation first
        var manifest = ReadManifestSync(directoryPath);
        if (manifest != null)
            return ValidateWithManifest(directoryPath, manifest);

        // Fallback: heuristic validation
        return ValidateFallback(directoryPath);
    }

    private static ValidationResult ValidateWithManifest(string directoryPath, DownloadManifest manifest)
    {
        var missing = new List<string>();
        var sizeMismatches = new List<string>();

        foreach (var entry in manifest.Files)
        {
            // Normalize path separators for cross-platform compatibility
            var normalizedPath = entry.Path.Replace('/', Path.DirectorySeparatorChar);
            var fullPath = Path.Combine(directoryPath, normalizedPath);
            if (!CacheManager.ContentExists(fullPath))
            {
                missing.Add(entry.Path);
                continue;
            }

            var actualSize = CacheManager.GetContentLength(fullPath);
            if (entry.Size > 0 && actualSize != entry.Size)
            {
                sizeMismatches.Add($"{entry.Path} (expected {entry.Size}, actual {actualSize})");
            }
        }

        if (missing.Count > 0)
            return new ValidationResult(false,
                $"Missing files: {string.Join(", ", missing)}",
                missing.ToArray());

        if (sizeMismatches.Count > 0)
            return new ValidationResult(false,
                $"File size mismatch: {string.Join("; ", sizeMismatches)}");

        return ValidationResult.Valid;
    }

    private static ValidationResult ValidateFallback(string directoryPath)
    {
        var files = Directory.GetFiles(directoryPath);

        // Check GGUF files
        var ggufFiles = files.Where(f => f.EndsWith(".gguf", StringComparison.OrdinalIgnoreCase)).ToArray();
        if (ggufFiles.Length > 0)
            return ValidateGgufFiles(ggufFiles);

        // Check ONNX GenAI (genai_config.json + model files)
        var hasGenaiConfig = File.Exists(Path.Combine(directoryPath, "genai_config.json"));
        if (hasGenaiConfig)
        {
            var hasModelOnnx = File.Exists(Path.Combine(directoryPath, "model.onnx"));
            var hasModelData = File.Exists(Path.Combine(directoryPath, "model.onnx.data"));
            if (!hasModelOnnx && !hasModelData)
                return new ValidationResult(false,
                    "ONNX GenAI model incomplete: genai_config.json exists but model.onnx and model.onnx.data are missing.");
            return ValidationResult.Valid;
        }

        // Check general ONNX
        var onnxFiles = files.Where(f => f.EndsWith(".onnx", StringComparison.OrdinalIgnoreCase)).ToArray();
        if (onnxFiles.Length > 0)
            return ValidationResult.Valid;

        return new ValidationResult(false,
            "No model files found (.onnx, .gguf, or genai_config.json).");
    }

    private static ValidationResult ValidateGgufFiles(string[] ggufFiles)
    {
        foreach (var ggufFile in ggufFiles)
        {
            try
            {
                using var stream = File.OpenRead(ggufFile);
                if (stream.Length < 8)
                    return new ValidationResult(false,
                        $"GGUF file too small: {Path.GetFileName(ggufFile)}");

                var buffer = new byte[4];
                stream.ReadExactly(buffer);
                var magic = BitConverter.ToUInt32(buffer);
                if (magic != GgufMagic)
                    return new ValidationResult(false,
                        $"Invalid GGUF magic number in {Path.GetFileName(ggufFile)}. File may be corrupt.");
            }
            catch (Exception ex)
            {
                Trace.TraceWarning($"[ModelDirectoryValidator] Failed to read GGUF: {ex.Message}");
                return new ValidationResult(false,
                    $"Cannot read GGUF file: {Path.GetFileName(ggufFile)}");
            }
        }

        return ValidationResult.Valid;
    }

    private static DownloadManifest? ReadManifestSync(string directoryPath)
    {
        return DownloadManifest.Read(directoryPath);
    }
}
