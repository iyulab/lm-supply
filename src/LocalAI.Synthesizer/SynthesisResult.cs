namespace LocalAI.Synthesizer;

/// <summary>
/// Represents the result of a speech synthesis operation.
/// </summary>
public sealed class SynthesisResult
{
    /// <summary>
    /// Gets the synthesized audio samples (32-bit float, -1.0 to 1.0).
    /// </summary>
    public required float[] AudioSamples { get; init; }

    /// <summary>
    /// Gets the sample rate of the audio.
    /// </summary>
    public required int SampleRate { get; init; }

    /// <summary>
    /// Gets the number of audio channels (typically 1 for mono).
    /// </summary>
    public int Channels { get; init; } = 1;

    /// <summary>
    /// Gets the duration of the audio in seconds.
    /// </summary>
    public double DurationSeconds => (double)AudioSamples.Length / SampleRate / Channels;

    /// <summary>
    /// Gets the inference time in milliseconds.
    /// </summary>
    public double? InferenceTimeMs { get; init; }

    /// <summary>
    /// Gets the real-time factor (audio duration / processing time).
    /// Values greater than 1 indicate faster-than-real-time synthesis.
    /// </summary>
    public double? RealTimeFactor => InferenceTimeMs.HasValue
        ? DurationSeconds / (InferenceTimeMs.Value / 1000.0)
        : null;

    /// <summary>
    /// Gets the original text that was synthesized.
    /// </summary>
    public string? Text { get; init; }

    /// <summary>
    /// Converts the audio samples to 16-bit PCM bytes.
    /// </summary>
    /// <returns>16-bit PCM audio data.</returns>
    public byte[] ToPcm16Bytes()
    {
        var bytes = new byte[AudioSamples.Length * 2];
        for (int i = 0; i < AudioSamples.Length; i++)
        {
            var sample = Math.Clamp(AudioSamples[i], -1.0f, 1.0f);
            var pcm16 = (short)(sample * 32767);
            bytes[i * 2] = (byte)(pcm16 & 0xFF);
            bytes[i * 2 + 1] = (byte)((pcm16 >> 8) & 0xFF);
        }
        return bytes;
    }

    /// <summary>
    /// Converts the audio samples to a WAV file byte array.
    /// </summary>
    /// <returns>WAV file data.</returns>
    public byte[] ToWavBytes()
    {
        var pcmData = ToPcm16Bytes();
        using var ms = new MemoryStream();
        using var writer = new BinaryWriter(ms);

        // WAV header
        writer.Write("RIFF"u8);
        writer.Write(36 + pcmData.Length); // File size - 8
        writer.Write("WAVE"u8);

        // Format chunk
        writer.Write("fmt "u8);
        writer.Write(16); // Chunk size
        writer.Write((short)1); // PCM format
        writer.Write((short)Channels);
        writer.Write(SampleRate);
        writer.Write(SampleRate * Channels * 2); // Byte rate
        writer.Write((short)(Channels * 2)); // Block align
        writer.Write((short)16); // Bits per sample

        // Data chunk
        writer.Write("data"u8);
        writer.Write(pcmData.Length);
        writer.Write(pcmData);

        return ms.ToArray();
    }
}

/// <summary>
/// Represents a chunk of audio data for streaming synthesis.
/// </summary>
public sealed class AudioChunk
{
    /// <summary>
    /// Gets the audio samples for this chunk (32-bit float).
    /// </summary>
    public required float[] Samples { get; init; }

    /// <summary>
    /// Gets the sample rate of the audio.
    /// </summary>
    public required int SampleRate { get; init; }

    /// <summary>
    /// Gets the chunk index (0-based).
    /// </summary>
    public int Index { get; init; }

    /// <summary>
    /// Gets whether this is the final chunk.
    /// </summary>
    public bool IsFinal { get; init; }

    /// <summary>
    /// Gets the duration of this chunk in seconds.
    /// </summary>
    public double DurationSeconds => (double)Samples.Length / SampleRate;
}
