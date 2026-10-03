using System.Diagnostics;
using LMSupply.Hardware;
using LMSupply.Vision;

namespace LMSupply.Captioner.Models;

/// <summary>
/// Model registry for the Captioner domain.
/// </summary>
public sealed class CaptionerModelRegistry : ModelRegistryBase<ModelInfo>
{
    /// <summary>
    /// Gets the default registry instance with built-in models.
    /// </summary>
    public static CaptionerModelRegistry Default { get; } = CreateDefault();

    /// <summary>
    /// Builds the default registry: built-in models plus the user's
    /// ~/.lmsupply/aliases.json "captioner" section (fail-soft).
    /// </summary>
    internal static CaptionerModelRegistry CreateDefault()
        => AliasConfiguration.ApplyDomain(new CaptionerModelRegistry(DefaultModels.All), AliasConfiguration.Domains.Captioner);

    /// <summary>
    /// Initializes a new registry with the specified system models.
    /// </summary>
    /// <param name="systemModels">Models to register as system defaults.</param>
    public CaptionerModelRegistry(IEnumerable<ModelInfo> systemModels)
        : base(systemModels) { }

    /// <summary>
    /// The model <c>auto</c> loads: Florence-2 base on every hardware tier — its int8 variant (chosen on CPU-only tiers)
    /// is smaller than ViT-GPT2 and names the subject far more often.
    /// </summary>
    protected override ModelInfo GetAutoModel()
    {
        Trace.TraceInformation("[CaptionerModelRegistry] Auto-selecting default captioning model");
        return DefaultModels.Florence2Base with { AliasName = "auto" };
    }

    /// <summary>
    /// Creates a fallback model info for unknown model IDs (HuggingFace repos or local paths).
    /// </summary>
    protected override ModelInfo CreateFallbackModelInfo(string modelId)
    {
        Trace.TraceInformation($"[CaptionerModelRegistry] Creating fallback model info for: {modelId}");

        var parts = modelId.Split('/');
        var name = parts.Length > 1 ? parts[1] : modelId;

        // Default to ViT-GPT2-compatible settings for unknown models
        return new ModelInfo(
            RepoId: modelId,
            AliasName: modelId,
            DisplayName: name,
            EncoderFile: "encoder_model.onnx",
            DecoderFile: "decoder_model_merged.onnx",
            TokenizerType: TokenizerType.Gpt2,
            PreprocessProfile: PreprocessProfile.ViTGpt2,
            SupportsVqa: false,
            VocabSize: 50257,
            BosTokenId: 50256,
            EosTokenId: 50256,
            PadTokenId: 50256)
        {
            Subfolder = "onnx",
            Description = $"Custom model: {modelId}"
        };
    }
}
