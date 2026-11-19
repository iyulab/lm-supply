using LMSupply.Console.Host.Models.OpenAI;
using LMSupply.Exceptions;

namespace LMSupply.Console.Host.Infrastructure;

/// <summary>
/// API helper utilities
/// </summary>
public static class ApiHelper
{
    /// <summary>
    /// Generate a unique ID with prefix
    /// </summary>
    public static string GenerateId(string prefix) => $"{prefix}-{Guid.NewGuid():N}";

    /// <summary>
    /// Create an error response
    /// </summary>
    public static IResult Error(string message, string type = "invalid_request_error", int statusCode = 400)
    {
        var error = new ErrorResponse
        {
            Error = new ErrorDetail
            {
                Message = message,
                Type = type
            }
        };

        return Results.Json(error, statusCode: statusCode);
    }

    /// <summary>
    /// Create an internal error response, returning 404 for model-not-found, 400 for known user-input errors, 500 otherwise.
    /// </summary>
    public static IResult InternalError(Exception ex)
    {
        // ModelNotFoundException is a client error (bad model name) → 404
        if (ex is ModelNotFoundException mnf)
        {
            return Results.Json(new ErrorResponse
            {
                Error = new ErrorDetail
                {
                    Message = mnf.Message,
                    Type = "invalid_request_error",
                    Code = "model_not_found"
                }
            }, statusCode: 404);
        }

        // ArgumentException variants are user-input errors → 400
        if (ex is ArgumentException argEx)
        {
            return Results.Json(new ErrorResponse
            {
                Error = new ErrorDetail
                {
                    Message = argEx.Message,
                    Type = "invalid_request_error",
                    Code = "invalid_value"
                }
            }, statusCode: 400);
        }

        // All other exceptions → 500 with opaque message (do not leak ex.Message)
        return Results.Json(new ErrorResponse
        {
            Error = new ErrorDetail
            {
                Message = "An internal server error occurred.",
                Type = "api_error",
                Code = "internal_error"
            }
        }, statusCode: 500);
    }

    /// <summary>
    /// Awaits a model call on an uploaded image. An image that cannot be decoded (a format the loader does not read, a damaged file)
    /// is the caller's mistake, so it surfaces as an <see cref="ArgumentException"/>, which <see cref="InternalError"/> answers
    /// with 400, rather than as the 500 an unexpected failure gets.
    /// </summary>
    public static async Task<T> ForUploadedImage<T>(Task<T> call)
    {
        try
        {
            return await call;
        }
        catch (InvalidDataException ex)
        {
            throw new ArgumentException("The uploaded file is not an image that can be decoded (JPEG, PNG, WebP, GIF, BMP or ICO).", ex);
        }
    }

    /// <summary>
    /// Parse input that can be string or array of strings
    /// </summary>
    public static List<string> ParseInput(System.Text.Json.JsonElement input)
    {
        if (input.ValueKind == System.Text.Json.JsonValueKind.String)
        {
            return [input.GetString()!];
        }

        if (input.ValueKind == System.Text.Json.JsonValueKind.Array)
        {
            return input.EnumerateArray()
                .Select(e => e.GetString()!)
                .ToList();
        }

        throw new ArgumentException("Input must be a string or array of strings");
    }
}
