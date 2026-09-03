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
/// Exception thrown when input exceeds the model's maximum context length.
/// </summary>
public class ContextLengthExceededException : InferenceException
{
    /// <summary>
    /// Gets the number of tokens in the input (if known).
    /// </summary>
    public int? TokenCount { get; }

    /// <summary>
    /// Gets the maximum context length supported by the model.
    /// </summary>
    public int MaxContextLength { get; }

    /// <summary>
    /// Initializes a new instance of the <see cref="ContextLengthExceededException"/> class.
    /// </summary>
    /// <param name="tokenCount">The number of tokens in the input (null if unknown).</param>
    /// <param name="maxContextLength">The maximum context length supported by the model.</param>
    public ContextLengthExceededException(int? tokenCount, int maxContextLength)
        : base(FormatMessage(tokenCount, maxContextLength))
    {
        TokenCount = tokenCount;
        MaxContextLength = maxContextLength;
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="ContextLengthExceededException"/> class.
    /// </summary>
    /// <param name="tokenCount">The number of tokens in the input (null if unknown).</param>
    /// <param name="maxContextLength">The maximum context length supported by the model.</param>
    /// <param name="innerException">The exception that is the cause of the current exception.</param>
    public ContextLengthExceededException(int? tokenCount, int maxContextLength, Exception innerException)
        : base(FormatMessage(tokenCount, maxContextLength), innerException)
    {
        TokenCount = tokenCount;
        MaxContextLength = maxContextLength;
    }

    private static string FormatMessage(int? tokenCount, int maxContextLength) =>
        tokenCount.HasValue
            ? $"Input length ({tokenCount.Value} tokens) exceeds model's maximum context length ({maxContextLength} tokens). Reduce the input or use a model with larger context support."
            : $"Input exceeds model's maximum context length ({maxContextLength} tokens). Reduce the input or use a model with larger context support.";
}

/// <summary>
/// Exception thrown when inference does not complete within the allotted time. A cold DirectML (or
/// other GPU provider) kernel initialization is the most common cause -- ONNX Runtime's
/// <c>RunOptions.Terminate</c> is only checked between operators, so it cannot preempt a hang
/// inside a single kernel, and the underlying native call keeps running on its own thread even
/// after this exception reaches the caller.
/// </summary>
public class InferenceTimeoutException : InferenceException
{
    /// <summary>
    /// Gets the timeout that was exceeded.
    /// </summary>
    public TimeSpan Timeout { get; }

    /// <summary>
    /// Initializes a new instance of the <see cref="InferenceTimeoutException"/> class.
    /// </summary>
    /// <param name="timeout">The timeout that was exceeded.</param>
    public InferenceTimeoutException(TimeSpan timeout)
        : base(
            $"Inference did not complete within {timeout.TotalSeconds:F0}s. This is often caused by " +
            "a cold GPU execution provider (e.g. DirectML) kernel initialization hang, which cannot " +
            "be cancelled cooperatively -- try ExecutionProvider.Cpu, or retry (a warmed-up GPU " +
            "session usually does not reproduce the hang).")
    {
        Timeout = timeout;
    }
}

/// <summary>
/// Exception thrown when an inference backend (e.g. llama-server) returns a non-success HTTP
/// response that no more specific exception type recognizes. Carries the raw status code and
/// response body so callers can see what the backend itself reported instead of a generic
/// HTTP status message.
/// </summary>
public class InferenceBackendException : InferenceException
{
    /// <summary>
    /// Gets the HTTP status code returned by the backend.
    /// </summary>
    public System.Net.HttpStatusCode StatusCode { get; }

    /// <summary>
    /// Gets the raw response body returned by the backend, if any.
    /// </summary>
    public string ResponseBody { get; }

    /// <summary>
    /// Initializes a new instance of the <see cref="InferenceBackendException"/> class.
    /// </summary>
    /// <param name="statusCode">The HTTP status code returned by the backend.</param>
    /// <param name="responseBody">The raw response body returned by the backend, if any.</param>
    public InferenceBackendException(System.Net.HttpStatusCode statusCode, string responseBody)
        : base(FormatMessage(statusCode, responseBody))
    {
        StatusCode = statusCode;
        ResponseBody = responseBody;
    }

    private static string FormatMessage(System.Net.HttpStatusCode statusCode, string responseBody) =>
        string.IsNullOrEmpty(responseBody)
            ? $"Inference backend returned {(int)statusCode} ({statusCode}) with an empty response body."
            : $"Inference backend returned {(int)statusCode} ({statusCode}): {responseBody}";
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

/// <summary>
/// Exception thrown when a user alias conflicts with a system alias name.
/// </summary>
public class AliasConflictException : LMSupplyException
{
    public string AliasName { get; }

    public AliasConflictException(string aliasName)
        : base($"Cannot register alias '{aliasName}': conflicts with a system alias.")
    {
        AliasName = aliasName;
    }
}

/// <summary>
/// Exception thrown when a user alias targets another user alias (chaining not allowed).
/// </summary>
public class AliasChainException : LMSupplyException
{
    public string AliasName { get; }
    public string TargetAlias { get; }

    public AliasChainException(string aliasName, string targetAlias)
        : base($"Cannot register alias '{aliasName}' targeting '{targetAlias}': user alias chaining is not allowed.")
    {
        AliasName = aliasName;
        TargetAlias = targetAlias;
    }
}

/// <summary>
/// Exception thrown when a native library load request would conflict with a different binary
/// already resident under the same library name and the caller opted into strict conflict
/// detection (see <c>RuntimeManagerOptions.FailOnRuntimeConflict</c>). The already-loaded binary
/// is left untouched -- this only fails the requesting call, since forcibly replacing a native
/// binary that other code may already hold handles into is not attempted (see docket
/// iyulab/lm-supply#151).
/// </summary>
public class NativeLibraryConflictException : LMSupplyException
{
    /// <summary>
    /// Gets the normalized library name that conflicted.
    /// </summary>
    public string LibraryName { get; }

    /// <summary>
    /// Gets the path this request tried to load.
    /// </summary>
    public string RequestedPath { get; }

    /// <summary>
    /// Gets the path of the binary that is actually resident for this library name.
    /// </summary>
    public string LoadedPath { get; }

    public NativeLibraryConflictException(string libraryName, string requestedPath, string loadedPath)
        : base(
            $"Cannot load '{libraryName}' from '{requestedPath}': a different binary for the same " +
            $"library name is already resident from '{loadedPath}'. The already-loaded binary was " +
            "left in place -- native libraries are never unloaded mid-process. Set " +
            "RuntimeManagerOptions.FailOnRuntimeConflict = false (the default) to silently keep " +
            "using the resident binary instead of throwing.")
    {
        LibraryName = libraryName;
        RequestedPath = requestedPath;
        LoadedPath = loadedPath;
    }
}
