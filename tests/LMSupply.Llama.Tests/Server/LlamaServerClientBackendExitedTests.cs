using System.Net;
using AwesomeAssertions;
using LMSupply.Exceptions;
using LMSupply.Llama.Server;

namespace LMSupply.Llama.Tests.Server;

/// <summary>
/// A request that fails because the server process exited must say so — with the exit code and the server's last
/// output — instead of surfacing only the transport failure. A failure while the server is alive stays as it was.
/// </summary>
public class LlamaServerClientBackendExitedTests
{
    private static readonly ChatCompletionMessage[] Hello = [new() { Role = "user", Content = "hello" }];

    [Fact]
    public async Task Refused_request_on_an_exited_server_reports_the_exit_code_and_log()
    {
        using var http = new HttpClient(new ThrowingHandler(new HttpRequestException("Connection refused")));
        var client = new LlamaServerClient("http://127.0.0.1:1", http, 4096);
        client.AttachServerState(() => new LlamaServerClient.ServerState(false, 137, "load_model: done\nslot launch\n"));

        var act = () => client.GenerateEmbeddingsBatchAsync(["x"], TestContext.Current.CancellationToken);

        var ex = (await act.Should().ThrowAsync<InferenceBackendExitedException>()).Which;
        ex.ExitCode.Should().Be(137);
        ex.RecentLog.Should().Contain("slot launch");
        ex.Message.Should().Contain("exit code 137").And.Contain("slot launch");
        ex.InnerException.Should().BeOfType<HttpRequestException>();
    }

    [Fact]
    public async Task Stream_cut_by_an_exited_server_reports_the_exit()
    {
        using var http = new HttpClient(new CutStreamHandler());
        var client = new LlamaServerClient("http://127.0.0.1:1", http, 4096);
        client.AttachServerState(() => new LlamaServerClient.ServerState(false, -1, "last line\n"));

        var act = async () =>
        {
            await foreach (var _ in client.GenerateChatStreamAsync(Hello, cancellationToken: TestContext.Current.CancellationToken))
            {
            }
        };

        var ex = (await act.Should().ThrowAsync<InferenceBackendExitedException>()).Which;
        ex.ExitCode.Should().Be(-1);
        ex.InnerException.Should().BeAssignableTo<IOException>();
    }

    [Fact]
    public async Task Failure_while_the_server_is_alive_is_not_translated()
    {
        using var http = new HttpClient(new ThrowingHandler(new HttpRequestException("reset")));
        var client = new LlamaServerClient("http://127.0.0.1:1", http, 4096);
        client.AttachServerState(() => new LlamaServerClient.ServerState(true, null, ""));

        var act = () => client.CountTokensAsync("x", TestContext.Current.CancellationToken);

        await act.Should().ThrowExactlyAsync<HttpRequestException>();
    }

    [Fact]
    public async Task Client_without_a_server_keeps_the_transport_failure()
    {
        using var http = new HttpClient(new ThrowingHandler(new HttpRequestException("refused")));
        var client = new LlamaServerClient("http://127.0.0.1:1", http, 4096);

        var act = () => client.GenerateChatWithToolsAsync(Hello, cancellationToken: TestContext.Current.CancellationToken);

        await act.Should().ThrowExactlyAsync<HttpRequestException>();
    }

    [Fact]
    public async Task Exit_registered_after_the_failure_is_still_seen()
    {
        // The connection is refused a moment before the process's exit is visible.
        var reads = 0;
        using var http = new HttpClient(new ThrowingHandler(new HttpRequestException("refused")));
        var client = new LlamaServerClient("http://127.0.0.1:1", http, 4096);
        client.AttachServerState(() => ++reads < 3
            ? new LlamaServerClient.ServerState(true, null, "")
            : new LlamaServerClient.ServerState(false, 1, "abort\n"));

        var act = () => client.RerankAsync("q", ["d"], cancellationToken: TestContext.Current.CancellationToken);

        (await act.Should().ThrowAsync<InferenceBackendExitedException>()).Which.ExitCode.Should().Be(1);
    }

    private sealed class ThrowingHandler(Exception failure) : HttpMessageHandler
    {
        protected override Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken) =>
            Task.FromException<HttpResponseMessage>(failure);
    }

    /// <summary>Answers 200 with an event stream that breaks after the first event, as a dying server's does.</summary>
    private sealed class CutStreamHandler : HttpMessageHandler
    {
        protected override Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken) =>
            Task.FromResult(new HttpResponseMessage(HttpStatusCode.OK) { Content = new StreamContent(new CutStream()) });
    }

    private sealed class CutStream : Stream
    {
        private readonly byte[] _first = "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\n"u8.ToArray();
        private int _position;

        public override int Read(byte[] buffer, int offset, int count)
        {
            if (_position >= _first.Length)
                throw new IOException("The response ended prematurely.");
            var n = Math.Min(count, _first.Length - _position);
            Array.Copy(_first, _position, buffer, offset, n);
            _position += n;
            return n;
        }

        public override bool CanRead => true;
        public override bool CanSeek => false;
        public override bool CanWrite => false;
        public override long Length => throw new NotSupportedException();
        public override long Position { get => throw new NotSupportedException(); set => throw new NotSupportedException(); }
        public override void Flush() { }
        public override long Seek(long offset, SeekOrigin origin) => throw new NotSupportedException();
        public override void SetLength(long value) => throw new NotSupportedException();
        public override void Write(byte[] buffer, int offset, int count) => throw new NotSupportedException();
    }
}
