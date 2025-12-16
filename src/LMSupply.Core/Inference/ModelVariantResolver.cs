namespace LMSupply.Inference;

/// <summary>
/// Resolves the best available model variant based on precision requirements and available files.
/// </summary>
public static class ModelVariantResolver
{
    /// <summary>
    /// Standard model file naming patterns for different precisions.
    /// </summary>
    public static class FilePatterns
    {
        public const string DefaultModel = "model.onnx";
        public const string FP32Model = "model_fp32.onnx";
        public const string FP16Model = "model_fp16.onnx";
        public const string INT8Model = "model_int8.onnx";
        public const string INT4Model = "model_int4.onnx";
        public const string QuantizedModel = "model_quantized.onnx";

        /// <summary>
        /// Alternative patterns for HuggingFace ONNX models.
        /// </summary>
        public static readonly string[] OnnxSubdirectoryPatterns =
        [
            "onnx/model.onnx",
            "onnx/model_fp32.onnx",
            "onnx/model_fp16.onnx",
            "onnx/model_int8.onnx",
            "onnx/model_quantized.onnx"
        ];
    }

    /// <summary>
    /// Resolves the best available model file for the specified precision.
    /// </summary>
    /// <param name="modelDirectory">Directory containing model files.</param>
    /// <param name="requestedPrecision">The requested model precision.</param>
    /// <param name="fallbackToDefault">Whether to fall back to default model if precision-specific not found.</param>
    /// <returns>Path to the best available model file, or null if not found.</returns>
    public static string? ResolveModelPath(
        string modelDirectory,
        ModelPrecision requestedPrecision,
        bool fallbackToDefault = true)
    {
        if (!Directory.Exists(modelDirectory))
            return null;

        // Get explicit precision-specific variants only (not model.onnx)
        var availableVariants = DiscoverModelVariants(modelDirectory, includeDefaultFallback: false);

        // Try to find exact match first
        if (availableVariants.Count > 0 &&
            requestedPrecision != ModelPrecision.Auto &&
            availableVariants.TryGetValue(requestedPrecision, out var exactMatch))
        {
            return exactMatch;
        }

        // For Auto or if exact match not found, use best available from explicit variants
        if (availableVariants.Count > 0)
        {
            var bestPrecision = SelectBestAvailablePrecision(
                availableVariants.Keys.ToList(),
                requestedPrecision);

            if (bestPrecision.HasValue && availableVariants.TryGetValue(bestPrecision.Value, out var bestMatch))
            {
                return bestMatch;
            }
        }

        // Fall back to default model.onnx only if allowed
        if (fallbackToDefault)
        {
            var defaultPath = Path.Combine(modelDirectory, FilePatterns.DefaultModel);
            if (File.Exists(defaultPath))
                return defaultPath;

            // Try onnx subdirectory
            var onnxSubPath = Path.Combine(modelDirectory, "onnx", "model.onnx");
            if (File.Exists(onnxSubPath))
                return onnxSubPath;
        }

        return null;
    }

    /// <summary>
    /// Discovers all available model variants in a directory.
    /// </summary>
    /// <param name="modelDirectory">Directory to scan.</param>
    /// <param name="includeDefaultFallback">Whether to include model.onnx when no specific variants found.</param>
    /// <returns>Dictionary mapping precision to model file path.</returns>
    public static Dictionary<ModelPrecision, string> DiscoverModelVariants(
        string modelDirectory,
        bool includeDefaultFallback = true)
    {
        var variants = new Dictionary<ModelPrecision, string>();

        // Check main directory for precision-specific variants
        CheckAndAddVariant(variants, modelDirectory, FilePatterns.FP32Model, ModelPrecision.FP32);
        CheckAndAddVariant(variants, modelDirectory, FilePatterns.FP16Model, ModelPrecision.FP16);
        CheckAndAddVariant(variants, modelDirectory, FilePatterns.INT8Model, ModelPrecision.INT8);
        CheckAndAddVariant(variants, modelDirectory, FilePatterns.INT4Model, ModelPrecision.INT4);
        CheckAndAddVariant(variants, modelDirectory, FilePatterns.QuantizedModel, ModelPrecision.INT8); // Assume quantized is INT8

        // Check onnx subdirectory for precision-specific variants
        var onnxSubDir = Path.Combine(modelDirectory, "onnx");
        if (Directory.Exists(onnxSubDir))
        {
            CheckAndAddVariant(variants, onnxSubDir, "model_fp32.onnx", ModelPrecision.FP32);
            CheckAndAddVariant(variants, onnxSubDir, "model_fp16.onnx", ModelPrecision.FP16);
            CheckAndAddVariant(variants, onnxSubDir, "model_int8.onnx", ModelPrecision.INT8);
            CheckAndAddVariant(variants, onnxSubDir, "model_int4.onnx", ModelPrecision.INT4);
            CheckAndAddVariant(variants, onnxSubDir, "model_quantized.onnx", ModelPrecision.INT8);
        }

        // If no specific variants found and fallback enabled, check for default model
        if (variants.Count == 0 && includeDefaultFallback)
        {
            var defaultModel = Path.Combine(modelDirectory, FilePatterns.DefaultModel);
            if (File.Exists(defaultModel))
            {
                var inferredPrecision = InferPrecisionFromFileName(defaultModel);
                variants[inferredPrecision] = defaultModel;
            }

            var onnxDefault = Path.Combine(modelDirectory, "onnx", "model.onnx");
            if (File.Exists(onnxDefault) && variants.Count == 0)
            {
                var inferredPrecision = InferPrecisionFromFileName(onnxDefault);
                variants[inferredPrecision] = onnxDefault;
            }
        }

        return variants;
    }

    /// <summary>
    /// Gets information about available model variants.
    /// </summary>
    public static ModelVariantInfo GetVariantInfo(string modelDirectory)
    {
        var variants = DiscoverModelVariants(modelDirectory);

        return new ModelVariantInfo
        {
            ModelDirectory = modelDirectory,
            AvailableVariants = variants,
            HasFP32 = variants.ContainsKey(ModelPrecision.FP32),
            HasFP16 = variants.ContainsKey(ModelPrecision.FP16),
            HasINT8 = variants.ContainsKey(ModelPrecision.INT8),
            HasINT4 = variants.ContainsKey(ModelPrecision.INT4),
            RecommendedPrecision = SelectBestAvailablePrecision(
                variants.Keys.ToList(), ModelPrecision.Auto) ?? ModelPrecision.FP16
        };
    }

    private static void CheckAndAddVariant(
        Dictionary<ModelPrecision, string> variants,
        string directory,
        string fileName,
        ModelPrecision precision)
    {
        var path = Path.Combine(directory, fileName);
        if (File.Exists(path) && !variants.ContainsKey(precision))
        {
            variants[precision] = path;
        }
    }

    private static ModelPrecision? SelectBestAvailablePrecision(
        IList<ModelPrecision> available,
        ModelPrecision requested)
    {
        if (available.Count == 0)
            return null;

        if (requested == ModelPrecision.Auto)
        {
            // Priority: FP16 > INT8 > FP32 > INT4 (balance of speed/accuracy)
            if (available.Contains(ModelPrecision.FP16)) return ModelPrecision.FP16;
            if (available.Contains(ModelPrecision.INT8)) return ModelPrecision.INT8;
            if (available.Contains(ModelPrecision.FP32)) return ModelPrecision.FP32;
            if (available.Contains(ModelPrecision.INT4)) return ModelPrecision.INT4;
        }
        else
        {
            // Try to get as close to requested as possible
            if (available.Contains(requested)) return requested;

            // Fall back in order of quality degradation
            var fallbackOrder = GetFallbackOrder(requested);
            foreach (var fallback in fallbackOrder)
            {
                if (available.Contains(fallback))
                    return fallback;
            }
        }

        return available.FirstOrDefault();
    }

    private static IEnumerable<ModelPrecision> GetFallbackOrder(ModelPrecision requested) => requested switch
    {
        ModelPrecision.FP32 => [ModelPrecision.FP16, ModelPrecision.INT8, ModelPrecision.INT4],
        ModelPrecision.FP16 => [ModelPrecision.FP32, ModelPrecision.INT8, ModelPrecision.INT4],
        ModelPrecision.INT8 => [ModelPrecision.FP16, ModelPrecision.FP32, ModelPrecision.INT4],
        ModelPrecision.INT4 => [ModelPrecision.INT8, ModelPrecision.FP16, ModelPrecision.FP32],
        _ => [ModelPrecision.FP16, ModelPrecision.INT8, ModelPrecision.FP32, ModelPrecision.INT4]
    };

    private static ModelPrecision InferPrecisionFromFileName(string path)
    {
        var fileName = Path.GetFileName(path).ToLowerInvariant();

        if (fileName.Contains("fp32") || fileName.Contains("float32"))
            return ModelPrecision.FP32;
        if (fileName.Contains("fp16") || fileName.Contains("float16") || fileName.Contains("half"))
            return ModelPrecision.FP16;
        if (fileName.Contains("int8") || fileName.Contains("quantized") || fileName.Contains("qint8"))
            return ModelPrecision.INT8;
        if (fileName.Contains("int4") || fileName.Contains("qint4"))
            return ModelPrecision.INT4;

        // Default to FP32 for standard model.onnx
        return ModelPrecision.FP32;
    }
}

/// <summary>
/// Information about available model variants.
/// </summary>
public sealed class ModelVariantInfo
{
    /// <summary>
    /// The model directory path.
    /// </summary>
    public required string ModelDirectory { get; init; }

    /// <summary>
    /// Dictionary of available variants and their paths.
    /// </summary>
    public required Dictionary<ModelPrecision, string> AvailableVariants { get; init; }

    /// <summary>
    /// Whether FP32 variant is available.
    /// </summary>
    public bool HasFP32 { get; init; }

    /// <summary>
    /// Whether FP16 variant is available.
    /// </summary>
    public bool HasFP16 { get; init; }

    /// <summary>
    /// Whether INT8 variant is available.
    /// </summary>
    public bool HasINT8 { get; init; }

    /// <summary>
    /// Whether INT4 variant is available.
    /// </summary>
    public bool HasINT4 { get; init; }

    /// <summary>
    /// Recommended precision based on available variants.
    /// </summary>
    public ModelPrecision RecommendedPrecision { get; init; }

    /// <summary>
    /// Gets the total number of available variants.
    /// </summary>
    public int VariantCount => AvailableVariants.Count;

    public override string ToString()
    {
        var variants = string.Join(", ", AvailableVariants.Keys.Select(k => k.ToString()));
        return $"ModelVariants({VariantCount}): {variants}, Recommended: {RecommendedPrecision}";
    }
}
