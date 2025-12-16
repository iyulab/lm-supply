using FluentAssertions;
using LMSupply.Synthesizer;

namespace LMSupply.Synthesizer.Tests;

public class SynthesisResultTests
{
    [Fact]
    public void SynthesisResult_CalculatesDurationCorrectly()
    {
        // Arrange - 1 second of audio at 22050 Hz
        var samples = new float[22050];

        // Act
        var result = new SynthesisResult
        {
            AudioSamples = samples,
            SampleRate = 22050
        };

        // Assert
        result.DurationSeconds.Should().BeApproximately(1.0, 0.001);
    }

    [Fact]
    public void SynthesisResult_CalculatesRealTimeFactor()
    {
        // Arrange - 2 seconds of audio processed in 500ms
        var samples = new float[44100]; // 2 seconds at 22050 Hz

        // Act
        var result = new SynthesisResult
        {
            AudioSamples = samples,
            SampleRate = 22050,
            InferenceTimeMs = 500
        };

        // Assert
        result.RealTimeFactor.Should().BeApproximately(4.0, 0.01);
    }

    [Fact]
    public void SynthesisResult_RealTimeFactor_NullWhenNoInferenceTime()
    {
        // Act
        var result = new SynthesisResult
        {
            AudioSamples = new float[1000],
            SampleRate = 22050
        };

        // Assert
        result.RealTimeFactor.Should().BeNull();
    }

    [Fact]
    public void SynthesisResult_ToPcm16Bytes_ConvertsCorrectly()
    {
        // Arrange
        var samples = new float[] { 0.0f, 1.0f, -1.0f, 0.5f };

        var result = new SynthesisResult
        {
            AudioSamples = samples,
            SampleRate = 22050
        };

        // Act
        var bytes = result.ToPcm16Bytes();

        // Assert
        bytes.Should().HaveCount(samples.Length * 2); // 2 bytes per sample
    }

    [Fact]
    public void SynthesisResult_ToPcm16Bytes_ClipsValues()
    {
        // Arrange - values outside [-1, 1] should be clipped
        var samples = new float[] { 2.0f, -2.0f };

        var result = new SynthesisResult
        {
            AudioSamples = samples,
            SampleRate = 22050
        };

        // Act
        var bytes = result.ToPcm16Bytes();

        // Assert
        // 2.0 clipped to 1.0 -> 32767
        var sample1 = BitConverter.ToInt16(bytes, 0);
        sample1.Should().Be(32767);

        // -2.0 clipped to -1.0 -> -32767 (approximately)
        var sample2 = BitConverter.ToInt16(bytes, 2);
        sample2.Should().Be(-32767);
    }

    [Fact]
    public void SynthesisResult_ToWavBytes_ContainsValidHeader()
    {
        // Arrange
        var samples = new float[100];
        var result = new SynthesisResult
        {
            AudioSamples = samples,
            SampleRate = 22050
        };

        // Act
        var wavBytes = result.ToWavBytes();

        // Assert
        wavBytes.Should().HaveCountGreaterThan(44); // WAV header is 44 bytes

        // Check RIFF header
        wavBytes[0].Should().Be((byte)'R');
        wavBytes[1].Should().Be((byte)'I');
        wavBytes[2].Should().Be((byte)'F');
        wavBytes[3].Should().Be((byte)'F');

        // Check WAVE format
        wavBytes[8].Should().Be((byte)'W');
        wavBytes[9].Should().Be((byte)'A');
        wavBytes[10].Should().Be((byte)'V');
        wavBytes[11].Should().Be((byte)'E');
    }

    [Fact]
    public void SynthesisResult_ToWavBytes_ContainsCorrectSampleRate()
    {
        // Arrange
        var result = new SynthesisResult
        {
            AudioSamples = new float[100],
            SampleRate = 22050
        };

        // Act
        var wavBytes = result.ToWavBytes();

        // Assert - Sample rate is at offset 24-27 (4 bytes, little-endian)
        var sampleRate = BitConverter.ToInt32(wavBytes, 24);
        sampleRate.Should().Be(22050);
    }
}

public class AudioChunkTests
{
    [Fact]
    public void AudioChunk_CalculatesDurationCorrectly()
    {
        // Arrange - 0.5 seconds of audio
        var samples = new float[11025];

        // Act
        var chunk = new AudioChunk
        {
            Samples = samples,
            SampleRate = 22050,
            Index = 0
        };

        // Assert
        chunk.DurationSeconds.Should().BeApproximately(0.5, 0.001);
    }

    [Fact]
    public void AudioChunk_TracksIndex()
    {
        // Act
        var chunk1 = new AudioChunk
        {
            Samples = new float[100],
            SampleRate = 22050,
            Index = 0,
            IsFinal = false
        };

        var chunk2 = new AudioChunk
        {
            Samples = new float[100],
            SampleRate = 22050,
            Index = 1,
            IsFinal = true
        };

        // Assert
        chunk1.Index.Should().Be(0);
        chunk1.IsFinal.Should().BeFalse();
        chunk2.Index.Should().Be(1);
        chunk2.IsFinal.Should().BeTrue();
    }
}
