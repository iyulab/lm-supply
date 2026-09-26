using System.Text.Json;
using System.Text.Json.Serialization;
using LMSupply.Core.Download;
using LMSupply.Download;
using LMSupply.Runtime;

namespace LMSupply.Json;

/// <summary>
/// Source-generated metadata for every type LMSupply.Core reads or writes as JSON (state files, manifests, hub and
/// NuGet API responses). Each serializer options instance in this assembly resolves through it, so a host that disables
/// reflection-based serialization (trimmed/AOT, the default for file-based <c>dotnet run app.cs</c>) still loads models.
/// </summary>
[JsonSerializable(typeof(ModelMetadata))]
[JsonSerializable(typeof(DownloadManifest))]
[JsonSerializable(typeof(List<RepoFile>))]
[JsonSerializable(typeof(ModelMetadataService.HfApiResponse))]
[JsonSerializable(typeof(Dictionary<string, Dictionary<string, string>>))]
[JsonSerializable(typeof(NuGetPackageResolver.VersionsResponse))]
[JsonSerializable(typeof(RuntimeVersionStateFile))]
internal sealed partial class CoreJsonContext : JsonSerializerContext;

/// <summary>
/// Options for call sites that passed none, resolved through <see cref="CoreJsonContext"/>. A separate class, not
/// static fields of the context: the generator's own statics live in another part of that partial class, and the
/// initialization order across parts is unspecified — a field there read <c>Default</c> as null and fell back to
/// reflection.
/// </summary>
internal static class CoreJsonOptions
{
    /// <summary>The defaults of a call that passed no options.</summary>
    internal static readonly JsonSerializerOptions Plain = new() { TypeInfoResolver = CoreJsonContext.Default };

    /// <summary>The defaults of <c>ReadFromJsonAsync</c>/<c>GetFromJsonAsync</c> (web).</summary>
    internal static readonly JsonSerializerOptions Web = new(JsonSerializerDefaults.Web) { TypeInfoResolver = CoreJsonContext.Default };
}
