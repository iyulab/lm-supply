using System.Buffers.Binary;
using System.Text;
using Concentus;

namespace LMSupply.Transcriber.Audio;

/// <summary>
/// Reads Opus audio from the two containers browsers record into — WebM (Chromium/Edge/WebView2
/// <c>MediaRecorder</c>) and Ogg (Firefox) — and decodes it with a pure-managed Opus decoder.
/// </summary>
/// <remarks>
/// Only what a recording needs is read: the first Opus track (WebM) or logical stream (Ogg), its channel count and
/// pre-skip, and its packets in order. A live WebM recording leaves its Segment and Cluster sizes unknown, so the WebM
/// walk descends into container elements without trusting their sizes.
/// </remarks>
internal static class OpusAudio
{
    /// <summary>Opus always decodes at 48 kHz.</summary>
    internal const int SampleRate = 48000;

    // 120 ms at 48 kHz: the longest Opus packet.
    private const int MaxFrameSamples = 5760;

    /// <summary>The container a stream's first bytes identify, as far as audio recordings go.</summary>
    internal enum Container { Unknown, WebM, Ogg, Mp4 }

    internal static Container Detect(ReadOnlySpan<byte> head)
    {
        if (head.Length >= 4 && head[0] == 0x1A && head[1] == 0x45 && head[2] == 0xDF && head[3] == 0xA3)
            return Container.WebM;
        if (head.Length >= 4 && head[..4].SequenceEqual("OggS"u8))
            return Container.Ogg;
        if (head.Length >= 8 && head[4..8].SequenceEqual("ftyp"u8))
            return Container.Mp4;
        return Container.Unknown;
    }

    /// <summary>
    /// Decodes a WebM or Ogg Opus recording to mono samples at 48 kHz, pre-skip removed.
    /// </summary>
    /// <exception cref="InvalidDataException">The container holds no Opus audio, or is malformed.</exception>
    internal static float[] DecodeToMono(byte[] data)
    {
        var stream = Detect(data) switch
        {
            Container.WebM => ReadWebM(data),
            Container.Ogg => ReadOgg(data),
            _ => throw new InvalidDataException("Not a WebM or Ogg container."),
        };

        if (stream.Packets.Count == 0)
            throw new InvalidDataException("The recording holds no Opus audio packets.");

        var decoder = OpusCodecFactory.CreateDecoder(SampleRate, stream.Channels);
        var frame = new float[MaxFrameSamples * stream.Channels];
        var mono = new List<float>(stream.Packets.Count * 960);
        var skip = stream.PreSkip;

        foreach (var packet in stream.Packets)
        {
            var samples = decoder.Decode(packet, frame, MaxFrameSamples, false);
            for (var i = 0; i < samples; i++)
            {
                if (skip > 0)
                {
                    skip--;
                    continue;
                }

                var sum = 0f;
                for (var ch = 0; ch < stream.Channels; ch++)
                    sum += frame[(i * stream.Channels) + ch];
                mono.Add(sum / stream.Channels);
            }
        }

        return [.. mono];
    }

    private sealed record OpusStream(int Channels, int PreSkip, List<byte[]> Packets);

    // ---- Ogg ---------------------------------------------------------------------------------------------------

    private static OpusStream ReadOgg(byte[] data)
    {
        int? serial = null;
        var channels = 0;
        var preSkip = 0;
        var packets = new List<byte[]>();
        var current = new MemoryStream();
        var headersSeen = 0;
        var pos = 0;

        while (pos + 27 <= data.Length)
        {
            if (!data.AsSpan(pos, 4).SequenceEqual("OggS"u8))
                throw new InvalidDataException($"Ogg page expected at byte {pos}.");

            var pageSerial = BinaryPrimitives.ReadInt32LittleEndian(data.AsSpan(pos + 14, 4));
            var segments = data[pos + 26];
            var tableStart = pos + 27;
            var bodyStart = tableStart + segments;
            if (bodyStart > data.Length)
                throw new InvalidDataException("Truncated Ogg page.");

            var offset = bodyStart;
            for (var s = 0; s < segments; s++)
            {
                var length = data[tableStart + s];
                if (offset + length > data.Length)
                    throw new InvalidDataException("Truncated Ogg segment.");

                var belongs = serial is null || serial == pageSerial;
                if (belongs)
                    current.Write(data, offset, length);
                offset += length;

                if (length < 255 && belongs)
                {
                    var packet = current.ToArray();
                    current.SetLength(0);

                    if (serial is null)
                    {
                        if (packet.Length >= 19 && packet.AsSpan(0, 8).SequenceEqual("OpusHead"u8))
                        {
                            serial = pageSerial;
                            (channels, preSkip) = ReadOpusHead(packet);
                            headersSeen = 1;
                        }
                    }
                    else if (headersSeen == 1)
                    {
                        headersSeen = 2; // OpusTags
                    }
                    else if (packet.Length > 0)
                    {
                        packets.Add(packet);
                    }
                }
            }

            pos = offset;
        }

        if (serial is null)
            throw new InvalidDataException("The Ogg stream carries no Opus track.");

        return new OpusStream(channels, preSkip, packets);
    }

    // ---- WebM (Matroska) ---------------------------------------------------------------------------------------

    private const uint SegmentId = 0x18538067;
    private const uint ClusterId = 0x1F43B675;
    private const uint TracksId = 0x1654AE6B;
    private const uint TrackEntryId = 0xAE;
    private const uint BlockGroupId = 0xA0;
    private const uint AudioId = 0xE1;
    private const uint TrackNumberId = 0xD7;
    private const uint CodecIdId = 0x86;
    private const uint CodecPrivateId = 0x63A2;
    private const uint ChannelsId = 0x9F;
    private const uint SimpleBlockId = 0xA3;
    private const uint BlockId = 0xA1;
    private const uint EbmlHeaderId = 0x1A45DFA3;

    private static readonly HashSet<uint> Containers = [SegmentId, ClusterId, TracksId, TrackEntryId, BlockGroupId, AudioId];

    private static OpusStream ReadWebM(byte[] data)
    {
        // Track entries as they are read; blocks reference a track by number.
        var tracks = new Dictionary<ulong, (string? Codec, byte[]? Private, int Channels)>();
        ulong? entryNumber = null;
        string? entryCodec = null;
        byte[]? entryPrivate = null;
        var entryChannels = 0;

        var blocks = new List<(ulong Track, byte[] Payload)>();
        var pos = 0;
        while (pos < data.Length)
        {
            if (!TryReadElementId(data, ref pos, out var id) || !TryReadSize(data, ref pos, out var size, out var unknown))
                break;

            if (id == TrackEntryId)
            {
                FlushEntry();
            }

            if (Containers.Contains(id))
                continue; // descend: children follow, whatever the declared size

            if (unknown || pos + (long)size > data.Length)
                break;

            var payload = data.AsSpan(pos, (int)size);
            switch (id)
            {
                case TrackNumberId: entryNumber = ReadUnsigned(payload); break;
                case CodecIdId: entryCodec = Encoding.ASCII.GetString(payload).TrimEnd('\0'); break;
                case CodecPrivateId: entryPrivate = payload.ToArray(); break;
                case ChannelsId: entryChannels = (int)ReadUnsigned(payload); break;
                case SimpleBlockId:
                case BlockId:
                    FlushEntry();
                    if (ReadBlock(payload) is { } block)
                        blocks.Add(block);
                    break;
                case EbmlHeaderId:
                    break;
            }

            pos += (int)size;
        }

        FlushEntry();

        var opus = tracks.FirstOrDefault(t => t.Value.Codec == "A_OPUS");
        if (opus.Value.Codec is null)
            throw new InvalidDataException("The WebM file carries no Opus audio track.");

        var (channels, preSkip) = opus.Value.Private is { Length: >= 19 } head && head.AsSpan(0, 8).SequenceEqual("OpusHead"u8)
            ? ReadOpusHead(head)
            : (Math.Max(1, opus.Value.Channels), 0);

        var packets = new List<byte[]>();
        foreach (var (track, payload) in blocks)
        {
            if (track != opus.Key)
                continue;
            packets.AddRange(SplitLaces(payload));
        }

        return new OpusStream(channels, preSkip, packets);

        void FlushEntry()
        {
            if (entryNumber is { } number && !tracks.ContainsKey(number))
                tracks[number] = (entryCodec, entryPrivate, entryChannels);
            entryNumber = null;
            entryCodec = null;
            entryPrivate = null;
            entryChannels = 0;
        }
    }

    /// <summary>A (Simple)Block: track number, 16-bit timecode, flags, then the frame(s) — returned with the flags byte first.</summary>
    private static (ulong Track, byte[] Payload)? ReadBlock(ReadOnlySpan<byte> block)
    {
        var pos = 0;
        if (!TryReadVint(block, ref pos, out var track, out _) || pos + 3 > block.Length)
            return null;

        // Keep the flags byte (lacing mode) in front of the frame data.
        return (track, block[(pos + 2)..].ToArray());
    }

    private static IEnumerable<byte[]> SplitLaces(byte[] flagsAndFrames)
    {
        var lacing = (flagsAndFrames[0] >> 1) & 0x03;
        var data = flagsAndFrames.AsMemory(1);
        if (lacing == 0)
        {
            yield return data.ToArray();
            yield break;
        }

        var span = data.Span;
        var count = span[0] + 1;
        var pos = 1;
        var sizes = new int[count];
        switch (lacing)
        {
            case 1: // Xiph
                for (var i = 0; i < count - 1; i++)
                {
                    int value, size = 0;
                    do { value = span[pos++]; size += value; } while (value == 255);
                    sizes[i] = size;
                }
                break;
            case 3: // EBML
                if (!TryReadVint(span, ref pos, out var first, out _))
                    throw new InvalidDataException("Malformed EBML lacing.");
                sizes[0] = (int)first;
                for (var i = 1; i < count - 1; i++)
                {
                    if (!TryReadVint(span, ref pos, out var raw, out var length))
                        throw new InvalidDataException("Malformed EBML lacing.");
                    var bias = (1L << ((7 * length) - 1)) - 1;
                    sizes[i] = sizes[i - 1] + (int)((long)raw - bias);
                }
                break;
            case 2: // fixed
                var each = (span.Length - pos) / count;
                for (var i = 0; i < count - 1; i++)
                    sizes[i] = each;
                break;
        }

        sizes[count - 1] = span.Length - pos - sizes.Take(count - 1).Sum();
        foreach (var size in sizes)
        {
            yield return data.Slice(pos, size).ToArray();
            pos += size;
        }
    }

    private static (int Channels, int PreSkip) ReadOpusHead(ReadOnlySpan<byte> head) =>
        (Math.Max(1, (int)head[9]), BinaryPrimitives.ReadUInt16LittleEndian(head.Slice(10, 2)));

    private static bool TryReadElementId(byte[] data, ref int pos, out uint id)
    {
        id = 0;
        if (pos >= data.Length)
            return false;
        var first = data[pos];
        var length = first == 0 ? 0 : System.Numerics.BitOperations.LeadingZeroCount((uint)first) - 23;
        if (length is < 1 or > 4 || pos + length > data.Length)
            return false;
        for (var i = 0; i < length; i++)
            id = (id << 8) | data[pos + i];
        pos += length;
        return true;
    }

    private static bool TryReadSize(byte[] data, ref int pos, out ulong size, out bool unknown)
    {
        unknown = false;
        if (!TryReadVint(data, ref pos, out size, out var length))
            return false;
        unknown = size == (1UL << (7 * length)) - 1;
        return true;
    }

    private static bool TryReadVint(ReadOnlySpan<byte> data, ref int pos, out ulong value, out int length)
    {
        value = 0;
        length = 0;
        if (pos >= data.Length || data[pos] == 0)
            return false;
        length = System.Numerics.BitOperations.LeadingZeroCount((uint)data[pos]) - 23;
        if (length > 8 || pos + length > data.Length)
            return false;
        value = (ulong)(data[pos] & (0xFF >> length));
        for (var i = 1; i < length; i++)
            value = (value << 8) | data[pos + i];
        pos += length;
        return true;
    }

    private static ulong ReadUnsigned(ReadOnlySpan<byte> payload)
    {
        ulong value = 0;
        foreach (var b in payload)
            value = (value << 8) | b;
        return value;
    }
}
