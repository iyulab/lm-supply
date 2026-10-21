using System.Net;
using System.Net.Sockets;
using System.Text;
using AwesomeAssertions;
using LMSupply.Llama.Server;
using Xunit;

namespace LMSupply.Llama.Tests.Server;

/// <summary>
/// A llama-server started by the library listens on loopback; its clients must not send those requests to a proxy
/// (HTTP_PROXY / HTTPS_PROXY or the system proxy), which cannot reach the caller's loopback.
/// </summary>
public sealed class LoopbackProxyBypassTests
{
    // A proxy address that refuses connections: a request routed through it fails.
    private static readonly WebProxy RefusingProxy = new("http://127.0.0.1:9");

    [Theory]
    [InlineData("http://127.0.0.1:8080", true)]
    [InlineData("http://localhost:8080", true)]
    [InlineData("http://[::1]:8080", true)]
    [InlineData("http://127.5.0.1:8080", true)]
    [InlineData("http://192.168.1.20:8080", false)]
    [InlineData("https://llm.example.com", false)]
    [InlineData("not a url", false)]
    public void IsLoopback_RecognisesLoopbackAddresses(string url, bool expected)
    {
        LoopbackHttp.IsLoopback(url).Should().Be(expected);
    }

    [Fact]
    public void CreateHandler_UsesTheProxyOnlyForRemoteServers()
    {
        using var loopback = LoopbackHttp.CreateHandler("http://127.0.0.1:8080");
        using var remote = LoopbackHttp.CreateHandler("https://llm.example.com");

        loopback.UseProxy.Should().BeFalse();
        remote.UseProxy.Should().BeTrue();
    }

    [Fact]
    public async Task LoopbackRequest_ReachesTheServer_WhenAProxyIsConfigured()
    {
        await using var server = await OneShotHttpServer.StartAsync();

        using var handler = LoopbackHttp.CreateHandler(server.BaseUrl);
        handler.Proxy = RefusingProxy;   // what HTTP_PROXY would supply; ignored because UseProxy is false
        using var client = new HttpClient(handler, disposeHandler: false) { Timeout = TimeSpan.FromSeconds(10) };

        using var response = await client.GetAsync($"{server.BaseUrl}/health", TestContext.Current.CancellationToken);

        response.StatusCode.Should().Be(HttpStatusCode.OK);
    }

    [Fact]
    public async Task ControlCase_TheSameRequestThroughTheProxy_Fails()
    {
        await using var server = await OneShotHttpServer.StartAsync();

        using var handler = new SocketsHttpHandler { UseProxy = true, Proxy = RefusingProxy };
        using var client = new HttpClient(handler, disposeHandler: false) { Timeout = TimeSpan.FromSeconds(10) };

        var act = () => client.GetAsync($"{server.BaseUrl}/health", TestContext.Current.CancellationToken);

        await act.Should().ThrowAsync<HttpRequestException>();
    }

    /// <summary>Answers one HTTP request on a loopback port with 200 OK.</summary>
    private sealed class OneShotHttpServer : IAsyncDisposable
    {
        private readonly TcpListener _listener;
        private readonly Task _serve;

        private OneShotHttpServer(TcpListener listener)
        {
            _listener = listener;
            _serve = ServeAsync();
        }

        public string BaseUrl => $"http://127.0.0.1:{((IPEndPoint)_listener.LocalEndpoint).Port}";

        public static Task<OneShotHttpServer> StartAsync()
        {
            var listener = new TcpListener(IPAddress.Loopback, 0);
            listener.Start();
            return Task.FromResult(new OneShotHttpServer(listener));
        }

        private async Task ServeAsync()
        {
            try
            {
                using var socket = await _listener.AcceptSocketAsync();
                var buffer = new byte[4096];
                await socket.ReceiveAsync(buffer, SocketFlags.None);
                var reply = Encoding.ASCII.GetBytes("HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\nok");
                await socket.SendAsync(reply, SocketFlags.None);
                socket.Shutdown(SocketShutdown.Both);
            }
            catch (SocketException)
            {
                // listener stopped before a request arrived (the control case never connects)
            }
            catch (ObjectDisposedException)
            {
            }
        }

        public async ValueTask DisposeAsync()
        {
            _listener.Stop();
            await _serve;
        }
    }
}
