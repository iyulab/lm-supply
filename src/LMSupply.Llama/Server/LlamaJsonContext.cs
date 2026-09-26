using System.Text.Json.Serialization;

namespace LMSupply.Llama.Server;

/// <summary>
/// Source-generated metadata for the llama-server wire types and the server state file, so GGUF loading and generation
/// work in a host that disables reflection-based serialization (trimmed/AOT, the default for file-based
/// <c>dotnet run app.cs</c>). The serializer options in this assembly resolve through it.
/// </summary>
[JsonSerializable(typeof(ChatCompletionRequest))]
[JsonSerializable(typeof(ChatCompletionChunk))]
[JsonSerializable(typeof(ChatCompletionFullResponse))]
[JsonSerializable(typeof(CompletionRequest))]
[JsonSerializable(typeof(CompletionChunk))]
[JsonSerializable(typeof(TokenizeRequest))]
[JsonSerializable(typeof(TokenizeResponse))]
[JsonSerializable(typeof(EmbeddingRequest))]
[JsonSerializable(typeof(EmbeddingResponse))]
[JsonSerializable(typeof(RerankRequest))]
[JsonSerializable(typeof(RerankResponse))]
[JsonSerializable(typeof(LlamaServerStateFile))]
// Values of ChatCompletionRequest.ChatTemplateKwargs (Dictionary<string, object>).
[JsonSerializable(typeof(bool))]
internal sealed partial class LlamaJsonContext : JsonSerializerContext;
