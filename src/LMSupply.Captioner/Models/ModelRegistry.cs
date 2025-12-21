using System.Diagnostics.CodeAnalysis;
using LMSupply.Vision;

namespace LMSupply.Captioner.Models;

/// <summary>
/// Registry of known captioning models and their configurations.
/// </summary>
public static class ModelRegistry
{
    private static readonly Dictionary<string, ModelInfo> Models = new(StringComparer.OrdinalIgnoreCase);

    static ModelRegistry()
    {
        // ViT-GPT2 (default model - most stable ONNX conversion)
        RegisterModel(new ModelInfo(
            RepoId: "Xenova/vit-gpt2-image-captioning",
            Alias: "vit-gpt2",
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
            Subfolder = "onnx"
        });

        // BLIP Base (quality model - 384x384 resolution)
        RegisterModel(new ModelInfo(
            RepoId: "Xenova/blip-image-captioning-base",
            Alias: "blip-base",
            DisplayName: "BLIP Image Captioning Base",
            EncoderFile: "vision_model.onnx",
            DecoderFile: "text_decoder_model_merged.onnx",
            TokenizerType: TokenizerType.Bert,
            PreprocessProfile: PreprocessProfile.Blip,
            SupportsVqa: false,
            VocabSize: 30524,
            BosTokenId: 30522,
            EosTokenId: 102,
            PadTokenId: 0)
        {
            Subfolder = "onnx"
        });

        // BLIP Large (large model - higher quality)
        RegisterModel(new ModelInfo(
            RepoId: "Xenova/blip-image-captioning-large",
            Alias: "blip-large",
            DisplayName: "BLIP Image Captioning Large",
            EncoderFile: "vision_model.onnx",
            DecoderFile: "text_decoder_model_merged.onnx",
            TokenizerType: TokenizerType.Bert,
            PreprocessProfile: PreprocessProfile.Blip,
            SupportsVqa: false,
            VocabSize: 30524,
            BosTokenId: 30522,
            EosTokenId: 102,
            PadTokenId: 0)
        {
            Subfolder = "onnx"
        });

        // GIT Base (fast model - efficient architecture)
        RegisterModel(new ModelInfo(
            RepoId: "Xenova/git-base-coco",
            Alias: "git-base",
            DisplayName: "GIT Base COCO",
            EncoderFile: "encoder_model.onnx",
            DecoderFile: "decoder_model_merged.onnx",
            TokenizerType: TokenizerType.Bert,
            PreprocessProfile: PreprocessProfile.ViTGpt2,
            SupportsVqa: false,
            VocabSize: 30522,
            BosTokenId: 101,
            EosTokenId: 102,
            PadTokenId: 0)
        {
            Subfolder = "onnx"
        });

        // Register standard aliases
        RegisterAlias("default", "vit-gpt2");
        RegisterAlias("fast", "git-base");
        RegisterAlias("quality", "blip-base");
        RegisterAlias("large", "blip-large");
    }

    /// <summary>
    /// Registers a model with the registry.
    /// </summary>
    public static void RegisterModel(ModelInfo model)
    {
        ArgumentNullException.ThrowIfNull(model);
        Models[model.Alias] = model;
        Models[model.RepoId] = model;
    }

    /// <summary>
    /// Registers an alias for an existing model.
    /// </summary>
    public static void RegisterAlias(string alias, string existingAlias)
    {
        if (!Models.TryGetValue(existingAlias, out var model))
        {
            throw new ArgumentException($"Model '{existingAlias}' not found in registry", nameof(existingAlias));
        }
        Models[alias] = model;
    }

    /// <summary>
    /// Tries to get model info by alias or repo ID.
    /// </summary>
    public static bool TryGetModel(string modelIdOrAlias, [NotNullWhen(true)] out ModelInfo? modelInfo)
    {
        return Models.TryGetValue(modelIdOrAlias, out modelInfo);
    }

    /// <summary>
    /// Gets model info by alias or repo ID.
    /// </summary>
    /// <exception cref="KeyNotFoundException">If the model is not found.</exception>
    public static ModelInfo GetModel(string modelIdOrAlias)
    {
        if (!TryGetModel(modelIdOrAlias, out var model))
        {
            throw new KeyNotFoundException($"Model '{modelIdOrAlias}' not found in registry. Use GetAvailableModels() to list available models.");
        }
        return model;
    }

    /// <summary>
    /// Gets a list of available model aliases.
    /// </summary>
    public static IEnumerable<string> GetAvailableModels()
    {
        return Models.Values
            .Select(m => m.Alias)
            .Distinct()
            .Order();
    }

    /// <summary>
    /// Gets all registered model information.
    /// </summary>
    public static IEnumerable<ModelInfo> GetAllModels()
    {
        return Models.Values.DistinctBy(m => m.Alias);
    }
}
