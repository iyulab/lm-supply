namespace LMSupply.Segmenter.Models;

/// <summary>
/// Registry for looking up segmenter model information by ID or alias.
/// </summary>
public sealed class SegmenterModelRegistry
{
    private readonly Dictionary<string, SegmenterModelInfo> _modelsByAlias;
    private readonly Dictionary<string, SegmenterModelInfo> _modelsById;

    /// <summary>
    /// Gets the default registry instance with built-in models.
    /// </summary>
    public static SegmenterModelRegistry Default { get; } = new(DefaultModels.All);

    /// <summary>
    /// Initializes a new registry with the specified models.
    /// </summary>
    /// <param name="models">Models to register.</param>
    public SegmenterModelRegistry(IEnumerable<SegmenterModelInfo> models)
    {
        ArgumentNullException.ThrowIfNull(models);

        var modelList = models.ToList();
        _modelsByAlias = new Dictionary<string, SegmenterModelInfo>(StringComparer.OrdinalIgnoreCase);
        _modelsById = new Dictionary<string, SegmenterModelInfo>(StringComparer.OrdinalIgnoreCase);

        foreach (var model in modelList)
        {
            _modelsByAlias[model.Alias] = model;
            _modelsById[model.Id] = model;
        }
    }

    /// <summary>
    /// Resolves a model identifier to its full information.
    /// </summary>
    /// <param name="modelIdOrAlias">Model ID, alias, or local path.</param>
    /// <returns>The model information.</returns>
    public SegmenterModelInfo Resolve(string modelIdOrAlias)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(modelIdOrAlias);

        // Check if it's a local path
        if (IsLocalPath(modelIdOrAlias))
        {
            return CreateLocalModelInfo(modelIdOrAlias);
        }

        // Try alias first
        if (_modelsByAlias.TryGetValue(modelIdOrAlias, out var modelByAlias))
        {
            return modelByAlias;
        }

        // Try full ID
        if (_modelsById.TryGetValue(modelIdOrAlias, out var modelById))
        {
            return modelById;
        }

        // Assume it's a HuggingFace ID not in our registry
        if (modelIdOrAlias.Contains('/'))
        {
            return CreateHuggingFaceModelInfo(modelIdOrAlias);
        }

        throw new ModelNotFoundException(
            $"Model '{modelIdOrAlias}' not found. Use a built-in alias (default, fast, quality, large, interactive), " +
            "a HuggingFace model ID (org/model), or a local file path.",
            modelIdOrAlias);
    }

    /// <summary>
    /// Tries to resolve a model identifier.
    /// </summary>
    /// <param name="modelIdOrAlias">Model ID, alias, or local path.</param>
    /// <param name="modelInfo">The resolved model information.</param>
    /// <returns>True if resolved successfully.</returns>
    public bool TryResolve(string modelIdOrAlias, out SegmenterModelInfo? modelInfo)
    {
        try
        {
            modelInfo = Resolve(modelIdOrAlias);
            return true;
        }
        catch
        {
            modelInfo = null;
            return false;
        }
    }

    /// <summary>
    /// Gets all registered models.
    /// </summary>
    public IEnumerable<SegmenterModelInfo> GetAll() => _modelsById.Values;

    /// <summary>
    /// Gets all available aliases.
    /// </summary>
    public IEnumerable<string> GetAliases() => _modelsByAlias.Keys;

    private static bool IsLocalPath(string path)
    {
        return path.EndsWith(".onnx", StringComparison.OrdinalIgnoreCase) ||
               Path.IsPathRooted(path) ||
               path.StartsWith("./", StringComparison.Ordinal) ||
               path.StartsWith("../", StringComparison.Ordinal) ||
               path.StartsWith(".\\", StringComparison.Ordinal) ||
               path.StartsWith("..\\", StringComparison.Ordinal);
    }

    private static SegmenterModelInfo CreateLocalModelInfo(string path)
    {
        var fullPath = Path.GetFullPath(path);
        var directory = Path.GetDirectoryName(fullPath) ?? ".";
        var fileName = Path.GetFileName(fullPath);

        return new SegmenterModelInfo
        {
            Id = fullPath,
            Alias = "local",
            DisplayName = $"Local: {fileName}",
            Architecture = "Unknown",
            ParametersM = 0,
            SizeBytes = File.Exists(fullPath) ? new FileInfo(fullPath).Length : 0,
            MIoU = 0,
            InputSize = 512,
            NumClasses = 150,
            OnnxFile = fileName,
            Dataset = "Unknown",
            Description = $"Local model from {directory}",
            License = "Unknown"
        };
    }

    private static SegmenterModelInfo CreateHuggingFaceModelInfo(string modelId)
    {
        var parts = modelId.Split('/');
        var name = parts.Length > 1 ? parts[1] : modelId;

        // Detect architecture from model ID
        var architecture = name.ToLowerInvariant() switch
        {
            var n when n.Contains("segformer") => "SegFormer",
            var n when n.Contains("deeplabv3") || n.Contains("deeplab") => "DeepLabV3+",
            var n when n.Contains("mask2former") => "Mask2Former",
            var n when n.Contains("sam") => "SAM",
            _ => "Unknown"
        };

        // Try to detect number of classes from model name
        var numClasses = name.ToLowerInvariant() switch
        {
            var n when n.Contains("ade") => 150,
            var n when n.Contains("cityscapes") => 19,
            var n when n.Contains("coco") => 171,
            _ => 150
        };

        // Detect input size from model name
        var inputSize = 512;
        if (name.Contains("640"))
            inputSize = 640;
        else if (name.Contains("1024"))
            inputSize = 1024;

        return new SegmenterModelInfo
        {
            Id = modelId,
            Alias = modelId,
            DisplayName = name,
            Architecture = architecture,
            ParametersM = 0,
            SizeBytes = 0,
            MIoU = 0,
            InputSize = inputSize,
            NumClasses = numClasses,
            OnnxFile = "model.onnx",
            Dataset = "Unknown",
            Description = $"HuggingFace model: {modelId}",
            License = "Unknown"
        };
    }
}

/// <summary>
/// Exception thrown when a model cannot be found.
/// </summary>
public class ModelNotFoundException : Exception
{
    /// <summary>
    /// The model identifier that was not found.
    /// </summary>
    public string ModelId { get; }

    /// <summary>
    /// Initializes a new instance.
    /// </summary>
    public ModelNotFoundException(string message, string modelId) : base(message)
    {
        ModelId = modelId;
    }
}
