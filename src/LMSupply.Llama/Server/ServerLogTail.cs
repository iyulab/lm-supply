namespace LMSupply.Llama.Server;

/// <summary>
/// The last lines a server process wrote, bounded. A long-lived server logs every request, so keeping all of it would
/// grow without limit; keeping none of it leaves a server that dies mid-request with nothing to say why.
/// Safe to append from the process's output callbacks while another thread reads.
/// </summary>
internal sealed class ServerLogTail
{
    /// <summary>The number of lines kept when no capacity is given.</summary>
    public const int DefaultCapacity = 200;

    private readonly string[] _lines;
    private readonly object _gate = new();
    private int _next;
    private int _count;

    public ServerLogTail(int capacity = DefaultCapacity)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(capacity, 1);
        _lines = new string[capacity];
    }

    public int Capacity => _lines.Length;

    public void Append(string line)
    {
        lock (_gate)
        {
            _lines[_next] = line;
            _next = (_next + 1) % _lines.Length;
            if (_count < _lines.Length)
                _count++;
        }
    }

    /// <summary>The kept lines, oldest first, one per line; empty when nothing was written.</summary>
    public override string ToString()
    {
        lock (_gate)
        {
            if (_count == 0)
                return string.Empty;

            var start = (_next - _count + _lines.Length) % _lines.Length;
            var builder = new System.Text.StringBuilder();
            for (var i = 0; i < _count; i++)
                builder.AppendLine(_lines[(start + i) % _lines.Length]);
            return builder.ToString();
        }
    }
}
