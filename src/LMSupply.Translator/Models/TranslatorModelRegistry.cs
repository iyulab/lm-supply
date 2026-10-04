using System.Diagnostics;

namespace LMSupply.Translator.Models;

/// <summary>
/// Model registry for the Translator domain.
/// </summary>
public sealed class TranslatorModelRegistry : ModelRegistryBase<TranslatorModelInfo>
{
    /// <summary>
    /// Gets the default registry instance with built-in models.
    /// </summary>
    public static TranslatorModelRegistry Default { get; } = CreateDefault();

    /// <summary>
    /// Builds the default registry: built-in models plus the user's
    /// ~/.lmsupply/aliases.json "translator" section (fail-soft).
    /// </summary>
    internal static TranslatorModelRegistry CreateDefault()
        => AliasConfiguration.ApplyDomain(new TranslatorModelRegistry(DefaultModels.All), AliasConfiguration.Domains.Translator);

    /// <summary>
    /// Initializes a new registry with the specified system models.
    /// </summary>
    /// <param name="systemModels">Models to register as system defaults.</param>
    public TranslatorModelRegistry(IEnumerable<TranslatorModelInfo> systemModels)
        : base(systemModels) { }

    /// <summary>
    /// Gets the auto-selected model.
    /// Translation models are language-pair specific so auto returns the default (ko-en).
    /// </summary>
    protected override TranslatorModelInfo GetAutoModel()
    {
        Trace.TraceInformation("[TranslatorModelRegistry] Auto-selecting default translation model");

        var model = DefaultModels.OpusMtKoEn;

        return new TranslatorModelInfo
        {
            Id = model.Id,
            AliasName = "auto",
            DisplayName = model.DisplayName,
            Architecture = model.Architecture,
            SourceLanguage = model.SourceLanguage,
            TargetLanguage = model.TargetLanguage,
            ParametersM = model.ParametersM,
            SizeBytes = model.SizeBytes,
            BleuScore = model.BleuScore,
            MaxLength = model.MaxLength,
            VocabSize = model.VocabSize,
            Subfolder = model.Subfolder,
            UseAutoDiscovery = model.UseAutoDiscovery,
            PreferredDecoderVariant = model.PreferredDecoderVariant,
            EncoderFile = model.EncoderFile,
            DecoderFile = model.DecoderFile,
            TokenizerFile = model.TokenizerFile,
            Description = model.Description,
            License = model.License
        };
    }

    /// <summary>
    /// Creates a fallback model info for unknown model IDs (HuggingFace repos or local paths).
    /// </summary>
    protected override TranslatorModelInfo CreateFallbackModelInfo(string modelId)
    {
        Trace.TraceInformation($"[TranslatorModelRegistry] Creating fallback model info for: {modelId}");

        // Check if it's a local path with an ONNX file
        if (modelId.EndsWith(".onnx", StringComparison.OrdinalIgnoreCase) ||
            Path.IsPathRooted(modelId) ||
            modelId.StartsWith("./", StringComparison.Ordinal) ||
            modelId.StartsWith("../", StringComparison.Ordinal) ||
            modelId.StartsWith(".\\", StringComparison.Ordinal) ||
            modelId.StartsWith("..\\", StringComparison.Ordinal))
        {
            return CreateLocalModelInfo(modelId);
        }

        return CreateHuggingFaceModelInfo(modelId);
    }

    private static TranslatorModelInfo CreateLocalModelInfo(string path)
    {
        var fullPath = Path.GetFullPath(path);
        var directory = Path.GetDirectoryName(fullPath) ?? ".";
        var fileName = Path.GetFileName(fullPath);

        return new TranslatorModelInfo
        {
            Id = fullPath,
            AliasName = "local",
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
            Description = $"Local model from {directory}"
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
            AliasName = modelId,
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
            Description = $"HuggingFace model: {modelId}"
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
