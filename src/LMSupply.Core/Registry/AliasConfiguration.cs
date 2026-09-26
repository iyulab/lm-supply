using System.Diagnostics;
using System.Text.Json;
using LMSupply.Json;

namespace LMSupply;

/// <summary>
/// Loads user alias definitions from the well-known configuration file and applies them
/// to module registries. This is the filesystem half of user aliasing — the programmatic
/// half is <see cref="IAliasRegistry.RegisterAlias"/>, which always wins over file entries
/// because it runs after registry initialization and overwrites by name.
/// </summary>
/// <remarks>
/// File location: <c>LMSUPPLY_ALIASES_FILE</c> environment variable, or
/// <c>~/.lmsupply/aliases.json</c>. Schema — domain-scoped alias maps:
/// <code>
/// {
///   "generator": { "my-writer": "gguf:qwen3-quality" },
///   "embedder":  { "my-embed": "BAAI/bge-m3" }
/// }
/// </code>
/// Applying is fail-soft by contract: a broken file or a bad entry (system-alias conflict,
/// alias chain, ':' in the name — which collides with the variant-qualifier syntax) is
/// skipped with a Trace warning and never crashes application startup.
/// </remarks>
public static class AliasConfiguration
{
    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        TypeInfoResolver = CoreJsonContext.Default,
        PropertyNameCaseInsensitive = true,
        ReadCommentHandling = JsonCommentHandling.Skip,
        AllowTrailingCommas = true
    };

    /// <summary>
    /// Environment variable that relocates the alias configuration file.
    /// </summary>
    public const string FileEnvironmentVariable = "LMSUPPLY_ALIASES_FILE";

    /// <summary>
    /// Canonical domain keys — one per module registry. The file's top-level keys
    /// must use these values.
    /// </summary>
    public static class Domains
    {
        public const string Generator = "generator";
        public const string Embedder = "embedder";
        public const string Reranker = "reranker";
        public const string Captioner = "captioner";
        public const string Transcriber = "transcriber";
        public const string Translator = "translator";
        public const string Synthesizer = "synthesizer";
        public const string Segmenter = "segmenter";
        public const string Detector = "detector";
        public const string ImageGenerator = "imagegenerator";
        public const string OcrDetection = "ocr-detection";
        public const string OcrRecognition = "ocr-recognition";
    }

    /// <summary>
    /// Alias configuration file path: <c>LMSUPPLY_ALIASES_FILE</c> env override,
    /// or <c>~/.lmsupply/aliases.json</c>.
    /// </summary>
    public static string DefaultFilePath
    {
        get
        {
            var overridden = Environment.GetEnvironmentVariable(FileEnvironmentVariable);
            if (!string.IsNullOrWhiteSpace(overridden))
                return overridden;

            return Path.Combine(
                Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
                ".lmsupply",
                "aliases.json");
        }
    }

    /// <summary>
    /// Parses a JSON string containing domain-scoped alias definitions.
    /// </summary>
    public static Dictionary<string, Dictionary<string, string>> ParseAliasJson(string json)
    {
        var doc = JsonSerializer.Deserialize<Dictionary<string, Dictionary<string, string>>>(
            json, JsonOptions);
        return doc ?? [];
    }

    /// <summary>
    /// Loads alias definitions from a JSON file. Returns empty if the file doesn't exist.
    /// </summary>
    public static Dictionary<string, Dictionary<string, string>> LoadFromFile(string filePath)
    {
        if (!File.Exists(filePath))
            return [];

        var json = File.ReadAllText(filePath);
        return ParseAliasJson(json);
    }

    /// <summary>
    /// Loads alias definitions from the default file path.
    /// </summary>
    public static Dictionary<string, Dictionary<string, string>> LoadFromDefault()
        => LoadFromFile(DefaultFilePath);

    /// <summary>
    /// Applies the given domain's section of the user alias configuration file to a registry,
    /// fail-soft, and returns the registry (fluent — module registries call this in their
    /// <c>Default</c> initializer). No-op when the file or the domain section is absent.
    /// </summary>
    public static TRegistry ApplyDomain<TRegistry>(TRegistry registry, string domain)
        where TRegistry : IAliasRegistry
    {
        Dictionary<string, Dictionary<string, string>> config;
        try
        {
            config = LoadFromDefault();
        }
        catch (Exception ex)
        {
            Trace.TraceWarning(
                $"[AliasConfiguration] Ignoring unreadable alias config '{DefaultFilePath}': {ex.Message}");
            return registry;
        }

        if (!config.TryGetValue(domain, out var aliases))
            return registry;

        foreach (var (alias, target) in aliases)
        {
            if (alias.Contains(':'))
            {
                Trace.TraceWarning(
                    $"[AliasConfiguration] Skipping alias '{alias}' ({domain}): ':' collides with the variant-qualifier syntax");
                continue;
            }

            try
            {
                registry.RegisterAlias(alias, target);
            }
            catch (Exception ex)
            {
                Trace.TraceWarning(
                    $"[AliasConfiguration] Skipping alias '{alias}' -> '{target}' ({domain}): {ex.Message}");
            }
        }

        return registry;
    }

    /// <summary>
    /// Applies a set of parsed aliases to a registry for a specific domain.
    /// </summary>
    public static void ApplyToRegistry<TModelInfo>(
        IModelRegistry<TModelInfo> registry,
        Dictionary<string, string> aliases)
        where TModelInfo : IModelInfoBase
    {
        foreach (var (alias, target) in aliases)
        {
            registry.RegisterAlias(alias, target);
        }
    }
}
