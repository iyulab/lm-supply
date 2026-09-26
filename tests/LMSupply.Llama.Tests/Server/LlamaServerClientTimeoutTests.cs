using System.Net;
using System.Net.Sockets;
using System.Text;
using AwesomeAssertions;
using LMSupply.Llama.Server;

namespace LMSupply.Llama.Tests.Server;

/// <summary>
/// The request limit of a client that owns its HttpClient. It defaults to 5 minutes, not HttpClient's 100 seconds —
/// multi-step tool-calling loops on local/CPU-bound inference routinely exceed that as each round's prompt grows with
/// prior tool results (ecosystem-e2e ChatToolFlowTests once hit "HttpClient.Timeout of 100 seconds elapsing").
/// A pooled server is shared by every caller that loads the same model, so the limit is applied per request and each
/// lease gets its own view: before, the client the pool created carried the first caller's limit for everyone.
/// These facts run against a real socket that answers late, because the owned HttpClient cannot take a fake handler.
/// </summary>
public class LlamaServerClientTimeoutTests
{
    private static CancellationToken Ct => TestContext.Current.CancellationToken;

    private static readonly ChatCompletionMessage[] Messages = [new() { Role = "user", Content = "hi" }];

    private const string ResponseBody =
        "{\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":\"ok\"},\"finish_reason\":\"stop\"}]}";

    /// <summary>Answers every request after a fixed delay; one connection per request.</summary>
    private sealed class SlowServer : IAsyncDisposable
    {
        private readonly TcpListener _listener = new(IPAddress.Loopback, 0);
        private readonly CancellationTokenSource _stop = new();
        private readonly Task _loop;

        public SlowServer(TimeSpan delay)
        {
            _listener.Start();
            _loop = Task.Run(() => AcceptLoopAsync(delay));
        }

        // 127.0.0.1, not localhost: on Windows localhost tries ::1 first and an IPv4-only listener costs ~2 s.
        public string BaseUrl => $"http://127.0.0.1:{((IPEndPoint)_listener.LocalEndpoint).Port}";

        private async Task AcceptLoopAsync(TimeSpan delay)
        {
            while (!_stop.IsCancellationRequested)
            {
                TcpClient socket;
                try { socket = await _listener.AcceptTcpClientAsync(_stop.Token); }
                catch (OperationCanceledException) { return; }
                catch (ObjectDisposedException) { return; }
                _ = Task.Run(() => AnswerAsync(socket, delay));
            }
        }

        private async Task AnswerAsync(TcpClient socket, TimeSpan delay)
        {
            using (socket)
            {
                try
                {
                    var stream = socket.GetStream();
                    await ReadRequestAsync(stream);
                    await Task.Delay(delay, _stop.Token);
                    var body = Encoding.UTF8.GetBytes(ResponseBody);
                    var head = Encoding.ASCII.GetBytes(
                        $"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {body.Length}\r\nConnection: close\r\n\r\n");
                    await stream.WriteAsync(head, _stop.Token);
                    await stream.WriteAsync(body, _stop.Token);
                }
                catch (Exception) when (_stop.IsCancellationRequested) { }
                catch (IOException) { } // the client gave up first
            }
        }

        private async Task ReadRequestAsync(NetworkStream stream)
        {
            var buffer = new byte[8192];
            var received = new StringBuilder();
            int headerEnd;
            while ((headerEnd = received.ToString().IndexOf("\r\n\r\n", StringComparison.Ordinal)) < 0)
            {
                var n = await stream.ReadAsync(buffer, _stop.Token);
                if (n == 0) return;
                received.Append(Encoding.ASCII.GetString(buffer, 0, n));
            }

            var headers = received.ToString()[..headerEnd];
            var lengthLine = headers.Split("\r\n").FirstOrDefault(l => l.StartsWith("Content-Length:", StringComparison.OrdinalIgnoreCase));
            var length = lengthLine is null ? 0 : int.Parse(lengthLine["Content-Length:".Length..].Trim(), System.Globalization.CultureInfo.InvariantCulture);
            var have = Encoding.ASCII.GetByteCount(received.ToString()) - (headerEnd + 4);
            while (have < length)
            {
                var n = await stream.ReadAsync(buffer, _stop.Token);
                if (n == 0) return;
                have += n;
            }
        }

        public async ValueTask DisposeAsync()
        {
            await _stop.CancelAsync();
            _listener.Stop();
            try { await _loop; } catch (OperationCanceledException) { }
            _stop.Dispose();
        }
    }

    [Fact]
    public void Constructor_NoHttpClientNoRequestTimeout_DefaultsToFiveMinutes()
    {
        using var client = new LlamaServerClient("http://127.0.0.1:9999");

        client.RequestTimeout.Should().Be(TimeSpan.FromMinutes(5),
            "local/CPU-bound inference routinely exceeds HttpClient's 100-second default");
        LlamaServerConfig.DefaultRequestTimeout.Should().Be(TimeSpan.FromMinutes(5));
    }

    [Fact]
    public void Constructor_ExternalHttpClientSupplied_RequestTimeoutIgnored()
    {
        using var externalClient = new HttpClient { Timeout = TimeSpan.FromSeconds(42) };
        using var client = new LlamaServerClient(
            "http://127.0.0.1:9999", externalClient, requestTimeout: TimeSpan.FromMinutes(10));

        client.RequestTimeout.Should().BeNull("a caller-supplied HttpClient owns its own timeout");
        externalClient.Timeout.Should().Be(TimeSpan.FromSeconds(42), "the client never rewrites a supplied HttpClient");
    }

    [Theory]
    [InlineData(0)]
    [InlineData(-5)]
    public void Constructor_NonPositiveRequestTimeout_Throws(int seconds)
    {
        var act = () => new LlamaServerClient("http://127.0.0.1:9999", requestTimeout: TimeSpan.FromSeconds(seconds));

        act.Should().Throw<ArgumentOutOfRangeException>();
    }

    [Fact]
    public void LlamaServerConfig_RequestTimeout_DefaultsToFiveMinutes()
    {
        var config = new LlamaServerConfig { ModelPath = "model.gguf" };

        config.RequestTimeout.Should().Be(TimeSpan.FromMinutes(5));
    }

    [Fact]
    public async Task ACompletionSlowerThanTheLimit_FailsTheWayHttpClientTimeoutDoes()
    {
        await using var server = new SlowServer(TimeSpan.FromSeconds(3));
        using var client = new LlamaServerClient(server.BaseUrl, requestTimeout: TimeSpan.FromMilliseconds(300));

        var act = () => client.GenerateChatWithToolsAsync(Messages, cancellationToken: Ct);

        var thrown = await act.Should().ThrowAsync<TaskCanceledException>();
        thrown.Which.InnerException.Should().BeOfType<TimeoutException>(
            "callers that tell a timeout from a cancellation by the inner exception keep working");
        thrown.Which.Message.Should().Contain("0.3 seconds");
    }

    [Fact]
    public async Task ACompletionWithinTheLimit_Succeeds()
    {
        await using var server = new SlowServer(TimeSpan.FromMilliseconds(200));
        using var client = new LlamaServerClient(server.BaseUrl, requestTimeout: TimeSpan.FromSeconds(30));

        var response = await client.GenerateChatWithToolsAsync(Messages, cancellationToken: Ct);

        response.Choices.Should().ContainSingle().Which.Message!.Content.Should().Be("ok");
    }

    [Fact]
    public async Task InfiniteTimeSpan_MeansNoLimit()
    {
        await using var server = new SlowServer(TimeSpan.FromMilliseconds(200));
        using var client = new LlamaServerClient(server.BaseUrl, requestTimeout: Timeout.InfiniteTimeSpan);

        var response = await client.GenerateChatWithToolsAsync(Messages, cancellationToken: Ct);

        response.Choices.Should().ContainSingle();
    }

    [Fact]
    public async Task TwoLeasesOfOneServer_KeepTheirOwnLimits()
    {
        await using var server = new SlowServer(TimeSpan.FromSeconds(1));
        // The pool creates the server's client with the first caller's limit; a later caller's lease is a view.
        using var pooled = new LlamaServerClient(server.BaseUrl, requestTimeout: TimeSpan.FromMilliseconds(300));
        using var patientLease = pooled.WithRequestTimeout(TimeSpan.FromSeconds(30));
        using var hastyLease = pooled.WithRequestTimeout(TimeSpan.FromMilliseconds(300));

        var patient = await patientLease.GenerateChatWithToolsAsync(Messages, cancellationToken: Ct);
        var hasty = () => hastyLease.GenerateChatWithToolsAsync(Messages, cancellationToken: Ct);

        patient.Choices.Should().ContainSingle("the patient caller's limit is its own, not the first caller's 300 ms");
        await hasty.Should().ThrowAsync<TaskCanceledException>();
    }

    [Fact]
    public async Task DisposingAView_LeavesTheSharedConnectionsUsable()
    {
        await using var server = new SlowServer(TimeSpan.Zero);
        using var pooled = new LlamaServerClient(server.BaseUrl);
        pooled.WithRequestTimeout(TimeSpan.FromSeconds(30)).Dispose();

        var response = await pooled.GenerateChatWithToolsAsync(Messages, cancellationToken: Ct);

        response.Choices.Should().ContainSingle();
    }

    [Fact]
    public async Task TheCallersCancellation_IsNotReportedAsATimeout()
    {
        await using var server = new SlowServer(TimeSpan.FromSeconds(5));
        using var client = new LlamaServerClient(server.BaseUrl, requestTimeout: TimeSpan.FromSeconds(30));
        using var cts = CancellationTokenSource.CreateLinkedTokenSource(Ct);
        cts.CancelAfter(TimeSpan.FromMilliseconds(200));

        var act = () => client.GenerateChatWithToolsAsync(Messages, cancellationToken: cts.Token);

        var thrown = await act.Should().ThrowAsync<OperationCanceledException>();
        thrown.Which.InnerException.Should().NotBeOfType<TimeoutException>();
    }
}
