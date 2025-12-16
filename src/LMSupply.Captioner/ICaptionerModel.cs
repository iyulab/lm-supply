using LMSupply.Captioner.Models;

namespace LMSupply.Captioner;

/// <summary>
/// Interface for image captioning models.
/// </summary>
public interface ICaptionerModel : IAsyncDisposable
{
    /// <summary>
    /// Gets the model identifier.
    /// </summary>
    string ModelId { get; }

    /// <summary>
    /// Pre-loads the model to avoid cold start latency.
    /// </summary>
    /// <param name="cancellationToken">Cancellation token.</param>
    Task WarmupAsync(CancellationToken cancellationToken = default);

    /// <summary>
    /// Gets information about the loaded model.
    /// </summary>
    /// <returns>Model information or null if not available.</returns>
    ModelInfo? GetModelInfo();

    /// <summary>
    /// Gets whether this model supports visual question answering.
    /// </summary>
    bool SupportsVqa { get; }

    /// <summary>
    /// Generates a caption for an image from a file path.
    /// </summary>
    /// <param name="imagePath">Path to the image file.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>Caption result containing generated text and confidence.</returns>
    Task<CaptionResult> CaptionAsync(string imagePath, CancellationToken cancellationToken = default);

    /// <summary>
    /// Generates a caption for an image from a stream.
    /// </summary>
    /// <param name="imageStream">Stream containing image data.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>Caption result containing generated text and confidence.</returns>
    Task<CaptionResult> CaptionAsync(Stream imageStream, CancellationToken cancellationToken = default);

    /// <summary>
    /// Generates a caption for an image from a byte array.
    /// </summary>
    /// <param name="imageData">Byte array containing image data.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>Caption result containing generated text and confidence.</returns>
    Task<CaptionResult> CaptionAsync(byte[] imageData, CancellationToken cancellationToken = default);

    /// <summary>
    /// Answers a question about an image (VQA).
    /// </summary>
    /// <param name="imagePath">Path to the image file.</param>
    /// <param name="question">Question about the image.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>VQA result containing the answer.</returns>
    /// <exception cref="NotSupportedException">If the model does not support VQA.</exception>
    Task<VqaResult> AnswerAsync(string imagePath, string question, CancellationToken cancellationToken = default);

    /// <summary>
    /// Answers a question about an image (VQA) from a stream.
    /// </summary>
    /// <param name="imageStream">Stream containing image data.</param>
    /// <param name="question">Question about the image.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>VQA result containing the answer.</returns>
    /// <exception cref="NotSupportedException">If the model does not support VQA.</exception>
    Task<VqaResult> AnswerAsync(Stream imageStream, string question, CancellationToken cancellationToken = default);
}
