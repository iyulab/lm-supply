namespace LMSupply;

/// <summary>
/// Base class for all LMSupply model options.
/// Provides common configuration properties shared across all packages.
/// </summary>
public abstract class LMSupplyOptionsBase
{
    /// <summary>
    /// Gets or sets the custom cache directory for <b>model files</b>.
    /// <para>Default: null (uses HuggingFace standard cache location: ~/.cache/huggingface/hub)</para>
    /// </summary>
    /// <remarks>
    /// <b>This scopes model weights only — not runtime binaries.</b> Backend executables such as
    /// <c>llama-server</c> are shared across every consumer on the machine and resolve through the
    /// global LMSupply cache (<c>LMSUPPLY_CACHE_DIR</c>, defaulting under the user's local
    /// application data), regardless of what this is set to. An application that isolates its data
    /// per installation will otherwise read a per-app directory that never contains a server binary
    /// and conclude that acquisition has never worked, when in fact it works and lands elsewhere.
    /// </remarks>
    public string? CacheDirectory { get; set; }

    /// <summary>
    /// Gets or sets the execution provider for inference.
    /// <para>Default: <see cref="ExecutionProvider.Auto"/> (automatically selects the best available provider)</para>
    /// </summary>
    public ExecutionProvider Provider { get; set; } = ExecutionProvider.Auto;

    /// <summary>
    /// Gets or sets the number of threads to use for inference.
    /// <para>Default: null (uses ONNX Runtime default, typically all available cores)</para>
    /// <para>Set to a specific value to limit CPU usage, useful for:</para>
    /// <list type="bullet">
    ///   <item>Running multiple models concurrently</item>
    ///   <item>Leaving CPU headroom for other tasks</item>
    ///   <item>Reducing power consumption</item>
    /// </list>
    /// </summary>
    /// <remarks>
    /// This setting primarily affects CPU inference. GPU inference may have different threading behavior
    /// controlled by the GPU driver.
    /// </remarks>
    public int? ThreadCount { get; set; }

    /// <summary>
    /// Gets or sets the ONNX Runtime log severity level for inference sessions.
    /// <para>Default: <see cref="OrtLogLevel.Error"/> (suppresses provider fallback warnings)</para>
    /// <para>Set to <see cref="OrtLogLevel.Warning"/> to see ONNX Runtime's default diagnostics,
    /// or <see cref="OrtLogLevel.Verbose"/> for full debugging output.</para>
    /// </summary>
    public OrtLogLevel LogLevel { get; set; } = OrtLogLevel.Error;

    /// <summary>
    /// Gets or sets the quantization hint for model variant selection during download.
    /// Overrides hardware-adaptive default when explicitly specified.
    /// <para>Set automatically when using the qualifier syntax (e.g., "large:fp16").</para>
    /// <para>Examples: "fp16", "int8", "q4", "bnb4", "q4f16"</para>
    /// <para>Default: null (hardware-adaptive selection)</para>
    /// </summary>
    public string? QuantizationHint { get; set; }

    /// <summary>
    /// Splits a model identifier into base ID and optional variant qualifier.
    /// Supports "alias:qualifier" syntax (e.g., "large:fp16", "default:q4").
    /// Does not split HuggingFace repo IDs (containing /) or local paths.
    /// </summary>
    /// <param name="modelIdOrAlias">The model identifier, possibly with qualifier.</param>
    /// <returns>Base model ID and optional qualifier string.</returns>
    public static (string BaseId, string? Qualifier) SplitQualifier(string modelIdOrAlias)
    {
        if (string.IsNullOrWhiteSpace(modelIdOrAlias))
            return (modelIdOrAlias, null);

        // Don't split paths or HF repo IDs
        if (modelIdOrAlias.Contains('/') || modelIdOrAlias.Contains('\\'))
            return (modelIdOrAlias, null);

        var colonIdx = modelIdOrAlias.LastIndexOf(':');
        if (colonIdx <= 0 || colonIdx == modelIdOrAlias.Length - 1)
            return (modelIdOrAlias, null);

        // Don't split Windows drive paths like "C:\..."
        if (colonIdx == 1 && char.IsLetter(modelIdOrAlias[0]))
            return (modelIdOrAlias, null);

        return (modelIdOrAlias[..colonIdx], modelIdOrAlias[(colonIdx + 1)..]);
    }
}
