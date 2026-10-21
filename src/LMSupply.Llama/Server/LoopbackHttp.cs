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
        => new(CreateHandler(baseUrl), disposeHandler: true) { Timeout = timeout };

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
