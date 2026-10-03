using LMSupply.Vision;

namespace LMSupply.Captioner.Models;

/// <summary>
/// Provides definitions for built-in supported captioning models.
/// </summary>
public static class DefaultModels
{
    /// <summary>
    /// Gets the default model: Florence-2 base, registered under <c>default</c> (and chosen by <c>auto</c>).
    /// </summary>
    public static ModelInfo Default => Florence2Default;

    /// <summary>
    /// ViT-GPT2 Image Captioning, alias <c>fast</c> — the smallest decoder, but it tends to name a familiar scene in
    /// place of the photo's subject. It was the <c>default</c> alias before 0.104.0. GPT-2 tokenizer, 224x224 ViT
    /// preprocessing.
    /// </summary>
    public static ModelInfo VitGpt2 { get; } = new(
        RepoId: "Xenova/vit-gpt2-image-captioning",
        AliasName: "fast",
        DisplayName: "ViT-GPT2 Image Captioning",
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
        License = "Apache-2.0", // the converted model, nlpconnect/vit-gpt2-image-captioning
        Description = "Fast: ViT-GPT2, the smallest decoder (the original GIT-Base COCO model is no longer accessible)"
    };

    /// <summary>
    /// Florence-2 base (fine-tuned), alias <c>quality</c> — a current vision-language model (MIT) that names the main
    /// subjects of everyday photos where ViT-GPT2 tends to substitute a familiar scene. Captions at three levels of
    /// detail (<see cref="CaptionerOptions.Detail"/>). Every ONNX file is published in quantized variants; a load takes
    /// the one <see cref="LMSupplyOptionsBase.QuantizationHint"/> or the hardware tier picks (int8 ≈ 275 MB on most
    /// machines, full precision ≈ 1.1 GB) — <see cref="LocalCaptioner.GetDownloadSizeBytesAsync"/> answers for a load.
    /// </summary>
    public static ModelInfo Florence2Base { get; } = new(
        RepoId: "onnx-community/Florence-2-base-ft",
        AliasName: "quality",
        DisplayName: "Florence-2 Base (fine-tuned)",
        EncoderFile: "vision_encoder.onnx",
        DecoderFile: "decoder_model_merged.onnx",
        TokenizerType: TokenizerType.Gpt2,
        PreprocessProfile: PreprocessProfile.Florence2,
        SupportsVqa: false,
        VocabSize: 51289,
        BosTokenId: 0,
        EosTokenId: 2,
        PadTokenId: 1)
    {
        Subfolder = "onnx",
        License = "MIT",
        AdditionalFiles = ["embed_tokens.onnx", "encoder_model.onnx"],
        Architecture = CaptionerArchitecture.Florence2,
        DecoderStartTokenId = 2,
        HasQuantizationVariants = true,
        Description = "Quality: Florence-2 base, names everyday subjects; brief, detailed or paragraph captions"
    };

    /// <summary>
    /// The same Florence-2 base model registered under <c>default</c> (since 0.104.0; ViT-GPT2 before). On 12 everyday
    /// CC0 photos ViT-GPT2 named 4 subjects wrongly and Florence-2 none, and its int8 download (≈ 275 MB) is smaller
    /// than ViT-GPT2's (≈ 960 MB).
    /// </summary>
    public static ModelInfo Florence2Default { get; } = Florence2Base with
    {
        AliasName = "default",
        Description = "Default: Florence-2 base (same model as quality)"
    };

    /// <summary>
    /// Gets all built-in models.
    /// Note: Xenova/blip-image-captioning-base (the former quality alias) and Xenova/blip-image-captioning-large (large)
    /// were removed because their HuggingFace repos became inaccessible (401 Unauthorized) circa 2026-03.
    /// </summary>
    public static IReadOnlyList<ModelInfo> All { get; } =
    [
        Florence2Default, // default
        VitGpt2,          // fast
        Florence2Base,    // quality
    ];
}
