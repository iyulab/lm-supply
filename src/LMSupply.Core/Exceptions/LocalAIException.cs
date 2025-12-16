namespace LMSupply.Exceptions;

/// <summary>
/// Base exception for all LMSupply errors.
/// </summary>
public class LMSupplyException : Exception
{
    /// <summary>
    /// Initializes a new instance of the <see cref="LMSupplyException"/> class.
    /// </summary>
    public LMSupplyException()
    {
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="LMSupplyException"/> class with a specified error message.
    /// </summary>
    /// <param name="message">The message that describes the error.</param>
    public LMSupplyException(string message) : base(message)
    {
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="LMSupplyException"/> class with a specified error message
    /// and a reference to the inner exception.
    /// </summary>
    /// <param name="message">The message that describes the error.</param>
    /// <param name="innerException">The exception that is the cause of the current exception.</param>
    public LMSupplyException(string message, Exception innerException) : base(message, innerException)
    {
    }
}

/// <summary>
/// Exception thrown when a requested model is not found.
/// </summary>
public class ModelNotFoundException : LMSupplyException
{
    /// <summary>
    /// Gets the model identifier that was not found.
    /// </summary>
    public string? ModelId { get; }

    /// <summary>
    /// Initializes a new instance of the <see cref="ModelNotFoundException"/> class.
    /// </summary>
    /// <param name="message">The message that describes the error.</param>
    /// <param name="modelId">The model identifier that was not found.</param>
    public ModelNotFoundException(string message, string? modelId = null) : base(message)
    {
        ModelId = modelId;
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="ModelNotFoundException"/> class.
    /// </summary>
    /// <param name="message">The message that describes the error.</param>
    /// <param name="modelId">The model identifier that was not found.</param>
    /// <param name="innerException">The exception that is the cause of the current exception.</param>
    public ModelNotFoundException(string message, string? modelId, Exception innerException)
        : base(message, innerException)
    {
        ModelId = modelId;
    }
}

/// <summary>
/// Exception thrown when model loading fails (e.g., ONNX session creation error).
/// </summary>
public class ModelLoadException : LMSupplyException
{
    /// <summary>
    /// Gets the model identifier that failed to load.
    /// </summary>
    public string? ModelId { get; }

    /// <summary>
    /// Initializes a new instance of the <see cref="ModelLoadException"/> class.
    /// </summary>
    /// <param name="message">The message that describes the error.</param>
    /// <param name="modelId">The model identifier that failed to load.</param>
    public ModelLoadException(string message, string? modelId = null) : base(message)
    {
        ModelId = modelId;
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="ModelLoadException"/> class.
    /// </summary>
    /// <param name="message">The message that describes the error.</param>
    /// <param name="modelId">The model identifier that failed to load.</param>
    /// <param name="innerException">The exception that is the cause of the current exception.</param>
    public ModelLoadException(string message, string? modelId, Exception innerException)
        : base(message, innerException)
    {
        ModelId = modelId;
    }
}

/// <summary>
/// Exception thrown when model download fails.
/// </summary>
public class ModelDownloadException : LMSupplyException
{
    /// <summary>
    /// Gets the model identifier for which download failed.
    /// </summary>
    public string? ModelId { get; }

    /// <summary>
    /// Initializes a new instance of the <see cref="ModelDownloadException"/> class.
    /// </summary>
    /// <param name="message">The message that describes the error.</param>
    /// <param name="modelId">The model identifier for which download failed.</param>
    public ModelDownloadException(string message, string? modelId = null) : base(message)
    {
        ModelId = modelId;
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="ModelDownloadException"/> class.
    /// </summary>
    /// <param name="message">The message that describes the error.</param>
    /// <param name="modelId">The model identifier for which download failed.</param>
    /// <param name="innerException">The exception that is the cause of the current exception.</param>
    public ModelDownloadException(string message, string? modelId, Exception innerException)
        : base(message, innerException)
    {
        ModelId = modelId;
    }
}

/// <summary>
/// Exception thrown when model inference fails.
/// </summary>
public class InferenceException : LMSupplyException
{
    /// <summary>
    /// Initializes a new instance of the <see cref="InferenceException"/> class.
    /// </summary>
    /// <param name="message">The message that describes the error.</param>
    public InferenceException(string message) : base(message)
    {
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="InferenceException"/> class.
    /// </summary>
    /// <param name="message">The message that describes the error.</param>
    /// <param name="innerException">The exception that is the cause of the current exception.</param>
    public InferenceException(string message, Exception innerException) : base(message, innerException)
    {
    }
}

/// <summary>
/// Exception thrown when tokenization fails.
/// </summary>
public class TokenizationException : LMSupplyException
{
    /// <summary>
    /// Initializes a new instance of the <see cref="TokenizationException"/> class.
    /// </summary>
    /// <param name="message">The message that describes the error.</param>
    public TokenizationException(string message) : base(message)
    {
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="TokenizationException"/> class.
    /// </summary>
    /// <param name="message">The message that describes the error.</param>
    /// <param name="innerException">The exception that is the cause of the current exception.</param>
    public TokenizationException(string message, Exception innerException) : base(message, innerException)
    {
    }
}
