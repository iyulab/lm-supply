using System.Text.Json;

namespace LMSupply.Console.Host.Infrastructure;

/// <summary>
/// Server-Sent Events 헬퍼
/// </summary>
public static class SseHelper
{
    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase
    };

    /// <summary>
    /// IAsyncEnumerable을 SSE 스트림으로 전송
    /// </summary>
    public static async Task StreamAsync<T>(
        HttpContext context,
        IAsyncEnumerable<T> source,
        CancellationToken cancellationToken = default)
    {
        // SSE 응답 전에 CORS 헤더를 수동으로 설정 (응답 시작 후 CORS 미들웨어 충돌 방지)
        SetCorsHeaders(context);

        context.Response.Headers.ContentType = "text/event-stream";
        context.Response.Headers.CacheControl = "no-cache";
        context.Response.Headers.Connection = "keep-alive";

        await foreach (var item in source.WithCancellation(cancellationToken))
        {
            var json = JsonSerializer.Serialize(item, JsonOptions);
            await context.Response.WriteAsync($"data: {json}\n\n", cancellationToken);
            await context.Response.Body.FlushAsync(cancellationToken);
        }

        await context.Response.WriteAsync("data: [DONE]\n\n", cancellationToken);
        await context.Response.Body.FlushAsync(cancellationToken);
    }

    /// <summary>
    /// 문자열 스트림을 SSE로 전송 (토큰 스트리밍용)
    /// </summary>
    public static async Task StreamTokensAsync(
        HttpContext context,
        IAsyncEnumerable<string> tokens,
        CancellationToken cancellationToken = default)
    {
        // SSE 응답 전에 CORS 헤더를 수동으로 설정 (응답 시작 후 CORS 미들웨어 충돌 방지)
        SetCorsHeaders(context);

        context.Response.Headers.ContentType = "text/event-stream";
        context.Response.Headers.CacheControl = "no-cache";
        context.Response.Headers.Connection = "keep-alive";

        await foreach (var token in tokens.WithCancellation(cancellationToken))
        {
            var json = JsonSerializer.Serialize(new { token }, JsonOptions);
            await context.Response.WriteAsync($"data: {json}\n\n", cancellationToken);
            await context.Response.Body.FlushAsync(cancellationToken);
        }

        await context.Response.WriteAsync("data: [DONE]\n\n", cancellationToken);
        await context.Response.Body.FlushAsync(cancellationToken);
    }

    /// <summary>
    /// SSE 응답 전에 CORS 헤더를 수동으로 설정
    /// </summary>
    private static void SetCorsHeaders(HttpContext context)
    {
        var origin = context.Request.Headers.Origin.ToString();
        if (!string.IsNullOrEmpty(origin))
        {
            context.Response.Headers.AccessControlAllowOrigin = origin;
            context.Response.Headers.AccessControlAllowCredentials = "true";
        }
    }
}
