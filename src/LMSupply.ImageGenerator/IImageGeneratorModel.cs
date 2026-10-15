using LMSupply.ImageGenerator.Models;

namespace LMSupply.ImageGenerator;

/// <summary>
/// Interface for text-to-image generation models.
/// </summary>
public interface IImageGeneratorModel : IAsyncDisposable
{
    /// <summary>
    /// Gets the model identifier.
    /// </summary>
    string ModelId { get; }

    /// <summary>
    /// Gets the estimated memory usage of this model in bytes.
    /// Based on ONNX model file sizes with overhead factor.
    /// </summary>
    long? EstimatedMemoryBytes { get; }

    /// <summary>
    /// Pre-loads the model to avoid cold start latency.
    /// </summary>
    /// <param name="cancellationToken">Cancellation token.</param>
    Task WarmupAsync(CancellationToken cancellationToken = default);

    /// <summary>
    /// Gets information about the loaded model.
    /// </summary>
    /// <returns>Model information or null if not available.</returns>
    ImageGeneratorModelInfo? GetModelInfo();

    /// <summary>
    /// Generates an image from a text prompt.
    /// </summary>
    /// <param name="prompt">Text prompt describing the image to generate.</param>
    /// <param name="options">Optional generation options.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>Generated image result.</returns>
    Task<GeneratedImage> GenerateAsync(
        string prompt,
        ImageGenerationOptions? options = null,
        CancellationToken cancellationToken = default);

    /// <summary>
    /// Generates multiple images from a text prompt.
    /// </summary>
    /// <param name="prompt">Text prompt describing the images to generate.</param>
    /// <param name="count">Number of images to generate.</param>
    /// <param name="options">Optional generation options.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>Array of generated images.</returns>
    Task<GeneratedImage[]> GenerateBatchAsync(
        string prompt,
        int count,
        ImageGenerationOptions? options = null,
        CancellationToken cancellationToken = default);

    /// <summary>
    /// Generates an image with step-by-step progress reporting.
    /// </summary>
    /// <param name="prompt">Text prompt describing the image to generate.</param>
    /// <param name="options">Optional generation options.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>Async enumerable of generation steps with optional previews.</returns>
    IAsyncEnumerable<GenerationStep> GenerateStreamingAsync(
        string prompt,
        ImageGenerationOptions? options = null,
        CancellationToken cancellationToken = default);
}
