using LMSupply.Download;

namespace LMSupply.Generator.Internal;

/// <summary>
/// Finds the ONNX Runtime GenAI model directory (the one holding <c>genai_config.json</c>) inside a downloaded model
/// repository whose variants live in subfolders: one level (<c>cpu-int4/</c>) or two
/// (<c>cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4/</c>). The hub load and the from-path load share it.
/// </summary>
internal static class OnnxVariantLocator
{
    public static string? Find(string basePath, ExecutionProvider provider)
    {
        if (!Directory.Exists(basePath))
            return null;

        // Provider-specific variant prefixes in priority order
        var variantPatterns = provider switch
        {
            ExecutionProvider.Cuda => new[] { "cuda", "gpu", "cpu" },
            _ => new[] { "cpu", "gpu" }
        };

        // Level 1: direct prefix match on an immediate subdirectory
        foreach (var pattern in variantPatterns)
        {
            foreach (var subdir in Directory.GetDirectories(basePath))
            {
                var dirName = Path.GetFileName(subdir);
                if (dirName.StartsWith(pattern, StringComparison.OrdinalIgnoreCase)
                    && IsValidModelDirectory(subdir))
                    return subdir;
            }
        }

        // Level 2: nested variant layout (e.g., cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4)
        // Walk into any level-1 directory whose name contains the provider pattern,
        // then check each of its immediate children for a valid model directory.
        foreach (var pattern in variantPatterns)
        {
            foreach (var subdir in Directory.GetDirectories(basePath))
            {
                var dirName = Path.GetFileName(subdir);
                if (!dirName.Contains(pattern, StringComparison.OrdinalIgnoreCase))
                    continue;

                foreach (var nested in Directory.GetDirectories(subdir))
                {
                    if (IsValidModelDirectory(nested))
                        return nested;
                }
            }
        }

        // Fallback: any valid model at level 1
        foreach (var subdir in Directory.GetDirectories(basePath))
        {
            if (IsValidModelDirectory(subdir))
                return subdir;
        }

        // Fallback: any valid model at level 2
        foreach (var subdir in Directory.GetDirectories(basePath))
        {
            foreach (var nested in Directory.GetDirectories(subdir))
            {
                if (IsValidModelDirectory(nested))
                    return nested;
            }
        }

        return null;
    }

    /// <summary>Whether <paramref name="path"/> is itself a loadable GenAI model directory.</summary>
    public static bool IsValidModelDirectory(string path)
    {
        var result = ModelDirectoryValidator.Validate(path);
        return result.IsValid;
    }
}
