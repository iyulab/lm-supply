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
            var status = await WriteErrorAsync(context, ex);
            var method = context.Request.Method;
            var path = context.Request.Path.Value ?? string.Empty;
            if (status >= 500)
            {
                LogUnhandledException(logger, ex, method, path);
            }
            else
            {
                LogRefusedRequest(logger, method, path, status);
            }
        }
    }

    private static async Task<int> WriteErrorAsync(HttpContext context, Exception ex)
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
        return status;
    }

    [LoggerMessage(Level = LogLevel.Information, Message = "Refused {Method} {Path} with {Status}")]
    private static partial void LogRefusedRequest(ILogger logger, string method, string path, int status);

    [LoggerMessage(Level = LogLevel.Error, Message = "Unhandled exception for {Method} {Path}")]
    private static partial void LogUnhandledException(ILogger logger, Exception ex, string method, string path);
}
