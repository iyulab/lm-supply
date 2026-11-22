using Microsoft.AspNetCore.Routing;
using Microsoft.AspNetCore.Routing.Template;
using LMSupply.Console.Host.Models.OpenAI;

namespace LMSupply.Console.Host.Infrastructure;

/// <summary>
/// Tells a request that reached the fallback because of its method or content type apart from one for a path that does not exist.
/// Routing prefers the lowest-priority fallback endpoint to a 405/415 when the path matches an endpoint that rejected the request on
/// its method or <c>Accepts</c> constraint, so without this the fallback answers 404 for «PUT /v1/embeddings» and for a form-encoded post.
/// </summary>
internal static class RouteMismatch
{
    /// <summary>
    /// Writes 405 (with <c>Allow</c>) or 415 and returns true when an endpoint matches the request path but not the request;
    /// returns false, writing nothing, when no endpoint matches the path.
    /// </summary>
    public static async Task<bool> TryWriteAsync(HttpContext context)
    {
        var allow = new SortedSet<string>(StringComparer.Ordinal);
        var methodAllowed = false;
        var path = context.Request.Path;

        foreach (var source in context.RequestServices.GetServices<EndpointDataSource>())
        {
            foreach (var endpoint in source.Endpoints)
            {
                if (endpoint is not RouteEndpoint route || route.RoutePattern.Parameters.Any(p => p.IsCatchAll))
                    continue;
                var methods = route.Metadata.GetMetadata<HttpMethodMetadata>()?.HttpMethods;
                if (methods is null)
                    continue;

                var matcher = new TemplateMatcher(new RouteTemplate(route.RoutePattern), new RouteValueDictionary());
                if (!matcher.TryMatch(path, new RouteValueDictionary()))
                    continue;

                foreach (var method in methods)
                    allow.Add(method);
                methodAllowed |= methods.Contains(context.Request.Method, StringComparer.OrdinalIgnoreCase);
            }
        }

        if (allow.Count == 0)
            return false;

        // The method is served here, so what the request failed on is the content type the endpoint accepts.
        var (status, message) = methodAllowed
            ? (StatusCodes.Status415UnsupportedMediaType, $"'{context.Request.ContentType ?? "(none)"}' is not a content type this endpoint accepts")
            : (StatusCodes.Status405MethodNotAllowed, $"'{context.Request.Method}' is not allowed here; use {string.Join(", ", allow)}");
        context.Response.StatusCode = status;
        if (!methodAllowed)
            context.Response.Headers.Allow = string.Join(", ", allow);
        await context.Response.WriteAsJsonAsync(new ErrorResponse
        {
            Error = new ErrorDetail { Message = message, Type = "invalid_request_error" }
        }, context.RequestAborted);
        return true;
    }
}
