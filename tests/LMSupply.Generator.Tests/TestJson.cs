using System.Text.Json;

namespace LMSupply.Generator.Tests;

/// <summary>
/// JSON helpers for a suite that runs with reflection-based serialization off (the csproj sets
/// <c>JsonSerializerIsReflectionEnabledByDefault=false</c>, like a trimmed/AOT host).
/// </summary>
internal static class TestJson
{
    /// <summary>Parses a JSON value without <c>JsonSerializer</c> (which would need reflection metadata).</summary>
    public static JsonElement ParseElement(string json)
    {
        using var doc = JsonDocument.Parse(json);
        return doc.RootElement.Clone();
    }
}
