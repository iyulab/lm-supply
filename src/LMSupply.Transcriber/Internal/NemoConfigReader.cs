using System.Diagnostics;
using System.Text.Json;

namespace LMSupply.Transcriber.Internal;

/// <summary>onnx-asr 형식 NeMo 내보내기의 <c>config.json</c>(<c>model_type</c>, <c>features_size</c>, <c>subsampling_factor</c>).</summary>
internal sealed record NemoModelConfig(string ModelType, int? FeaturesSize, int? SubsamplingFactor)
{
    public const string ConformerTdt = "nemo-conformer-tdt";

    public bool IsTdt => string.Equals(ModelType, ConformerTdt, StringComparison.OrdinalIgnoreCase);
}

/// <summary>
/// 모델 디렉터리의 <c>config.json</c>이 NeMo 내보내기인지 읽는다. <see cref="WhisperConfigReader"/>와 같은 자리, 같은 규칙 —
/// <c>model_type</c>이 없거나 NeMo 계열이 아니면 <c>null</c>.
/// </summary>
internal static class NemoConfigReader
{
    private const string ConfigFileName = "config.json";

    public static NemoModelConfig? ReadConfig(string modelDirectory)
    {
        var configPath = Path.Combine(modelDirectory, ConfigFileName);
        if (!File.Exists(configPath))
            return null;

        try
        {
            return ParseConfig(File.ReadAllText(configPath));
        }
        catch (Exception ex) when (ex is JsonException or IOException)
        {
            Trace.TraceInformation($"[NemoConfigReader] Failed to read config from {configPath}: {ex.Message}");
            return null;
        }
    }

    internal static NemoModelConfig? ParseConfig(string json)
    {
        using var doc = JsonDocument.Parse(json);
        var root = doc.RootElement;
        if (!root.TryGetProperty("model_type", out var modelType) || modelType.ValueKind != JsonValueKind.String)
            return null;

        var type = modelType.GetString();
        if (type is null || !type.StartsWith("nemo-", StringComparison.OrdinalIgnoreCase))
            return null;

        return new NemoModelConfig(type, ReadInt(root, "features_size"), ReadInt(root, "subsampling_factor"));
    }

    private static int? ReadInt(JsonElement root, string name)
        => root.TryGetProperty(name, out var el) && el.ValueKind == JsonValueKind.Number && el.TryGetInt32(out var v) ? v : null;
}
