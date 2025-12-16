namespace LMSupply.Translator.Models;

/// <summary>
/// Registry for looking up translator model information by ID or alias.
/// </summary>
public sealed class TranslatorModelRegistry
{
    private readonly Dictionary<string, TranslatorModelInfo> _modelsByAlias;
    private readonly Dictionary<string, TranslatorModelInfo> _modelsById;

    /// <summary>
    /// Gets the default registry instance with built-in models.
    /// </summary>
    public static TranslatorModelRegistry Default { get; } = new(DefaultModels.All);

    /// <summary>
    /// Initializes a new registry with the specified models.
    /// </summary>
    /// <param name="models">Models to register.</param>
    public TranslatorModelRegistry(IEnumerable<TranslatorModelInfo> models)
    {
        ArgumentNullException.ThrowIfNull(models);

        var modelList = models.ToList();
        _modelsByAlias = new Dictionary<string, TranslatorModelInfo>(StringComparer.OrdinalIgnoreCase);
        _modelsById = new Dictionary<string, TranslatorModelInfo>(StringComparer.OrdinalIgnoreCase);

        foreach (var model in modelList)
        {
            _modelsByAlias[model.Alias] = model;
            if (!_modelsById.ContainsKey(model.Id))
            {
                _modelsById[model.Id] = model;
            }
        }
    }

    /// <summary>
    /// Resolves a model identifier to its full information.
    /// </summary>
    /// <param name="modelIdOrAlias">Model ID, alias, or local path.</param>
    /// <returns>The model information.</returns>
    public TranslatorModelInfo Resolve(string modelIdOrAlias)
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
            $"Model '{modelIdOrAlias}' not found. Use a built-in alias (default, ko-en, en-ko), " +
            "a HuggingFace model ID (org/model), or a local file path.",
            modelIdOrAlias);
    }

    /// <summary>
    /// Tries to resolve a model identifier.
    /// </summary>
    /// <param name="modelIdOrAlias">Model ID, alias, or local path.</param>
    /// <param name="modelInfo">The resolved model information.</param>
    /// <returns>True if resolved successfully.</returns>
    public bool TryResolve(string modelIdOrAlias, out TranslatorModelInfo? modelInfo)
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
    public IEnumerable<TranslatorModelInfo> GetAll() => _modelsById.Values;

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

    private static TranslatorModelInfo CreateLocalModelInfo(string path)
    {
        var fullPath = Path.GetFullPath(path);
        var directory = Path.GetDirectoryName(fullPath) ?? ".";
        var fileName = Path.GetFileName(fullPath);

        return new TranslatorModelInfo
        {
            Id = fullPath,
            Alias = "local",
            DisplayName = $"Local: {fileName}",
            Architecture = "Unknown",
            SourceLanguage = "auto",
            TargetLanguage = "auto",
            ParametersM = 0,
            SizeBytes = File.Exists(fullPath) ? new FileInfo(fullPath).Length : 0,
            BleuScore = 0,
            MaxLength = 512,
            VocabSize = 0,
            EncoderFile = fileName.Contains("encoder") ? fileName : "encoder_model.onnx",
            DecoderFile = fileName.Contains("decoder") ? fileName : "decoder_model.onnx",
            Description = $"Local model from {directory}",
            License = "Unknown"
        };
    }

    private static TranslatorModelInfo CreateHuggingFaceModelInfo(string modelId)
    {
        var parts = modelId.Split('/');
        var name = parts.Length > 1 ? parts[1] : modelId;

        // Try to detect language pair from model name
        var (sourceLang, targetLang) = DetectLanguagePair(name);

        // Detect architecture from model ID
        var architecture = name.ToLowerInvariant() switch
        {
            var n when n.Contains("opus-mt") => "MarianMT",
            var n when n.Contains("marian") => "MarianMT",
            var n when n.Contains("nllb") => "NLLB",
            var n when n.Contains("m2m") => "M2M-100",
            _ => "Unknown"
        };

        return new TranslatorModelInfo
        {
            Id = modelId,
            Alias = modelId,
            DisplayName = name,
            Architecture = architecture,
            SourceLanguage = sourceLang,
            TargetLanguage = targetLang,
            ParametersM = 0,
            SizeBytes = 0,
            BleuScore = 0,
            MaxLength = 512,
            VocabSize = 0,
            EncoderFile = "encoder_model.onnx",
            DecoderFile = "decoder_model.onnx",
            Description = $"HuggingFace model: {modelId}",
            License = "Unknown"
        };
    }

    private static (string source, string target) DetectLanguagePair(string name)
    {
        // Try to parse opus-mt-{src}-{tgt} format
        var lowerName = name.ToLowerInvariant();
        if (lowerName.Contains("opus-mt-"))
        {
            var langPart = lowerName.Replace("opus-mt-", "");
            var langs = langPart.Split('-');
            if (langs.Length >= 2)
            {
                return (langs[0], langs[1]);
            }
        }

        return ("auto", "auto");
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
