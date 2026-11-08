using System.Net;

namespace LMSupply.Llama.Server;

/// <summary>
/// HTTP clients for a llama-server on this machine. A proxy (HTTP_PROXY / HTTPS_PROXY or the system proxy) cannot
/// reach the caller's loopback, so routing loopback requests through it only makes a local server look unreachable.
/// </summary>
internal static class LoopbackHttp
{
    /// <summary>
    /// Creates a client that bypasses any proxy when <paramref name="baseUrl"/> is a loopback address, and uses the
    /// default proxy behaviour otherwise (a remote llama-server may legitimately sit behind one).
    /// </summary>
    public static HttpClient CreateClient(string baseUrl, TimeSpan timeout)
    {
        var client = new HttpClient(CreateHandler(baseUrl), disposeHandler: true) { Timeout = timeout };

        // Every request opens its own connection. llama-server ends the connection after each streaming (SSE) response while
        // that response still says "Keep-Alive: timeout=5, max=100" (checked on builds b11977 and b12182), so a client that
        // reuses it sometimes sends the next request into a connection the server has just closed and fails with "the
        // response ended prematurely" — about once in a hundred or two sequential streaming requests. A new connection to a
        // local server costs microseconds.
        // TODO(upstream: https://github.com/ggml-org/llama.cpp/issues/29680 — the connection of a streaming response is closed although it advertises keep-alive): drop this once a build that keeps its word is the minimum.
        client.DefaultRequestHeaders.ConnectionClose = true;
        return client;
    }

    internal static SocketsHttpHandler CreateHandler(string baseUrl)
        => new() { UseProxy = !IsLoopback(baseUrl) };

    internal static bool IsLoopback(string url)
    {
        if (!Uri.TryCreate(url, UriKind.Absolute, out var uri))
        {
            return false;
        }

        if (uri.IsLoopback)
        {
            return true;
        }

        return IPAddress.TryParse(uri.DnsSafeHost, out var address) && IPAddress.IsLoopback(address);
    }
}
