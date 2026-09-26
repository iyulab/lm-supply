using System.Text.Json;
using System.Text.Json.Serialization.Metadata;

namespace LMSupply.Json;

/// <summary>
/// The typed metadata an options instance resolves for a type — what the <c>JsonTypeInfo</c> overloads of
/// <see cref="JsonSerializer"/> take. Those overloads are the ones the trimming/AOT analyzers accept; the
/// <c>JsonSerializerOptions</c> overloads are marked as needing reflection even when the options resolve through a
/// source-generated context. Taking the metadata from the same options keeps every option (naming policy,
/// converters, case-insensitivity) as it was.
/// </summary>
internal static class JsonTypeInfoExtensions
{
    /// <summary>The metadata for <typeparamref name="T"/> under these options.</summary>
    public static JsonTypeInfo<T> TypeInfo<T>(this JsonSerializerOptions options) => (JsonTypeInfo<T>)options.GetTypeInfo(typeof(T));

    /// <summary>The metadata for the static type of <paramref name="value"/> (as <c>Serialize&lt;T&gt;</c> would infer it).</summary>
    public static JsonTypeInfo<T> TypeInfoOf<T>(this JsonSerializerOptions options, T value) => options.TypeInfo<T>();
}
