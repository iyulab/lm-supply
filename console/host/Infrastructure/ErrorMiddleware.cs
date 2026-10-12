using System.Text.Json;

namespace LMSupply.Console.Host.Infrastructure;

public sealed partial class ErrorMiddleware(RequestDelegate next, ILogger<ErrorMiddleware> logger)
{
    public async Task InvokeAsync(HttpContext context)
    {
        try
        {
            await next(context);
        }
        catch (Exception ex) when (!context.Response.HasStarted)
        {
            LogUnhandledException(logger, ex, context.Request.Method, context.Request.Path);
            await WriteErrorAsync(context, ex);
        }
    }

    private static async Task WriteErrorAsync(HttpContext context, Exception ex)
    {
        var (status, type, code, message) = ex switch
        {
            ModelResolutionException mre => (404, "invalid_request_error", "model_not_found", mre.Message),
            ArgumentException ae         => (400, "invalid_request_error", "invalid_value", ae.Message),
            // The framework states the status of a request it could not read: 400 for a body that is not valid JSON,
            // 415 for a media type the endpoint does not take.
            BadHttpRequestException bre  => (bre.StatusCode, "invalid_request_error", "invalid_request", bre.Message),
            OperationCanceledException   => (499, "api_error", "request_cancelled", "Request was cancelled."),
            _                            => (500, "api_error", "internal_error", "An internal error occurred.")
        };

        context.Response.StatusCode = status;
        context.Response.ContentType = "application/json";

        var body = JsonSerializer.Serialize(new
        {
            error = new { message, type, code }
        });
        await context.Response.WriteAsync(body);
    }

    [LoggerMessage(Level = LogLevel.Error, Message = "Unhandled exception for {Method} {Path}")]
    private static partial void LogUnhandledException(ILogger logger, Exception ex, string method, string path);
}
