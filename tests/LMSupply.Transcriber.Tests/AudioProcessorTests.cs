using AwesomeAssertions;
using LMSupply.Transcriber.Audio;

namespace LMSupply.Transcriber.Tests;

/// <summary>
/// Tests for AudioProcessor static helper methods, mostly pure math functions plus one
/// LoadAudioAsync file-decode regression test (see docket iyulab/lm-supply#169).
/// </summary>
public class AudioProcessorTests
{
    private const int WhisperSampleRate = 16000;
    private const int NumSamples = WhisperSampleRate * 30; // 480000

    // --- PadOrTruncate ---

    [Fact]
    public void PadOrTruncate_ExactLength_ShouldReturnSameArray()
    {
        var samples = new float[NumSamples];
        samples[0] = 1.0f;

        var result = AudioProcessor.PadOrTruncate(samples);

        result.Should().BeSameAs(samples, "exact length returns same reference");
    }

    [Fact]
    public void PadOrTruncate_ShorterThanChunk_ShouldPadWithZeros()
    {
        var samples = new float[] { 0.5f, -0.5f, 1.0f };

        var result = AudioProcessor.PadOrTruncate(samples);

        result.Length.Should().Be(NumSamples);
        result[0].Should().Be(0.5f);
        result[1].Should().Be(-0.5f);
        result[2].Should().Be(1.0f);
        result[3].Should().Be(0f, "padded with zeros");
        result[NumSamples - 1].Should().Be(0f, "last element is zero");
    }

    [Fact]
    public void PadOrTruncate_LongerThanChunk_ShouldTruncate()
    {
        var samples = new float[NumSamples + 1000];
        samples[0] = 0.1f;
        samples[NumSamples - 1] = 0.9f;
        samples[NumSamples] = -1.0f; // beyond chunk

        var result = AudioProcessor.PadOrTruncate(samples);

        result.Length.Should().Be(NumSamples);
        result[0].Should().Be(0.1f);
        result[NumSamples - 1].Should().Be(0.9f);
    }

    [Fact]
    public void PadOrTruncate_EmptyArray_ShouldReturnZeroPaddedArray()
    {
        var samples = Array.Empty<float>();

        var result = AudioProcessor.PadOrTruncate(samples);

        result.Length.Should().Be(NumSamples);
        result.Should().OnlyContain(x => x == 0f);
    }

    [Fact]
    public void PadOrTruncate_SingleSample_ShouldPreserveAndPad()
    {
        var samples = new float[] { 0.42f };

        var result = AudioProcessor.PadOrTruncate(samples);

        result.Length.Should().Be(NumSamples);
        result[0].Should().Be(0.42f);
        result[1].Should().Be(0f);
    }

    // --- SplitIntoChunks ---

    [Fact]
    public void SplitIntoChunks_ExactOneChunk_ShouldReturnSingleChunk()
    {
        var samples = new float[NumSamples];
        samples[0] = 1.0f;
        samples[NumSamples - 1] = 0.5f;

        var chunks = AudioProcessor.SplitIntoChunks(samples);

        chunks.Should().HaveCount(1);
        chunks[0].Length.Should().Be(NumSamples);
        chunks[0][0].Should().Be(1.0f);
        chunks[0][NumSamples - 1].Should().Be(0.5f);
    }

    [Fact]
    public void SplitIntoChunks_TwoExactChunks_ShouldReturnTwo()
    {
        var samples = new float[NumSamples * 2];
        samples[0] = 0.1f;
        samples[NumSamples] = 0.2f;

        var chunks = AudioProcessor.SplitIntoChunks(samples);

        chunks.Should().HaveCount(2);
        chunks[0][0].Should().Be(0.1f);
        chunks[1][0].Should().Be(0.2f);
    }

    [Fact]
    public void SplitIntoChunks_PartialLastChunk_ShouldPadWithZeros()
    {
        var extraSamples = 100;
        var samples = new float[NumSamples + extraSamples];
        samples[NumSamples] = 0.7f; // first sample of second chunk

        var chunks = AudioProcessor.SplitIntoChunks(samples);

        chunks.Should().HaveCount(2);
        chunks[0].Length.Should().Be(NumSamples);
        chunks[1].Length.Should().Be(NumSamples);
        chunks[1][0].Should().Be(0.7f);
        chunks[1][extraSamples].Should().Be(0f, "remainder is zero-padded");
    }

    [Fact]
    public void SplitIntoChunks_EmptyArray_ShouldReturnEmptyList()
    {
        var samples = Array.Empty<float>();

        var chunks = AudioProcessor.SplitIntoChunks(samples);

        chunks.Should().BeEmpty();
    }

    [Fact]
    public void SplitIntoChunks_SmallSample_ShouldReturnOnePaddedChunk()
    {
        var samples = new float[] { 1.0f, 2.0f, 3.0f };

        var chunks = AudioProcessor.SplitIntoChunks(samples);

        chunks.Should().HaveCount(1);
        chunks[0].Length.Should().Be(NumSamples);
        chunks[0][0].Should().Be(1.0f);
        chunks[0][1].Should().Be(2.0f);
        chunks[0][2].Should().Be(3.0f);
        chunks[0][3].Should().Be(0f);
    }

    [Fact]
    public void SplitIntoChunks_EachChunkIsIndependent()
    {
        var samples = new float[NumSamples * 2];
        for (int i = 0; i < samples.Length; i++)
            samples[i] = i < NumSamples ? 1.0f : 2.0f;

        var chunks = AudioProcessor.SplitIntoChunks(samples);

        chunks[0].Should().OnlyContain(x => x == 1.0f);
        chunks[1].Should().OnlyContain(x => x == 2.0f);
    }

    // --- GetDurationSeconds ---

    [Theory]
    [InlineData(0, 0.0)]
    [InlineData(16000, 1.0)]
    [InlineData(480000, 30.0)]
    [InlineData(32000, 2.0)]
    [InlineData(8000, 0.5)]
    public void GetDurationSeconds_ShouldComputeCorrectly(int sampleCount, double expectedSeconds)
    {
        var samples = new float[sampleCount];

        AudioProcessor.GetDurationSeconds(samples).Should().BeApproximately(expectedSeconds, 0.0001);
    }

    [Fact]
    public void GetDurationSeconds_LargeAudio_ShouldBeAccurate()
    {
        // 5 minutes of audio
        var samples = new float[16000 * 300];

        AudioProcessor.GetDurationSeconds(samples).Should().BeApproximately(300.0, 0.0001);
    }

    // --- ComputeLogMelSpectrogram ---

    [Fact]
    public void ComputeLogMelSpectrogram_SilentAudio_ShouldReturnCorrectShape()
    {
        var samples = new float[NumSamples]; // all zeros (silence)

        var result = AudioProcessor.ComputeLogMelSpectrogram(samples, numMelBins: 80);

        // Shape: numMelBins * numFrames = 80 * 3000 = 240000
        result.Length.Should().Be(80 * 3000);
    }

    [Fact]
    public void ComputeLogMelSpectrogram_128MelBins_ShouldReturnCorrectShape()
    {
        var samples = new float[NumSamples];

        var result = AudioProcessor.ComputeLogMelSpectrogram(samples, numMelBins: 128);

        result.Length.Should().Be(128 * 3000);
    }

    [Fact]
    public void ComputeLogMelSpectrogram_ShortAudio_ShouldPadAndCompute()
    {
        // Shorter than 30s - should be padded internally
        var samples = new float[16000]; // 1 second

        var result = AudioProcessor.ComputeLogMelSpectrogram(samples);

        result.Length.Should().Be(80 * 3000);
    }

    [Fact]
    public void ComputeLogMelSpectrogram_Values_ShouldBeNormalized()
    {
        // Create a simple sine wave
        var samples = new float[NumSamples];
        for (int i = 0; i < samples.Length; i++)
        {
            samples[i] = MathF.Sin(2 * MathF.PI * 440 * i / WhisperSampleRate); // 440 Hz sine
        }

        var result = AudioProcessor.ComputeLogMelSpectrogram(samples);

        // After normalization: (logMel + 4.0) / 4.0
        // Values should be finite and within a reasonable range
        result.Should().OnlyContain(x => float.IsFinite(x));
    }

    // --- LoadAudioAsync (docket iyulab/lm-supply#169) ---

    [Fact]
    public async Task LoadAudioAsync_RealMp3File_DecodesViaNLayer()
    {
        // Regression test for docket iyulab/lm-supply#169: NAudio 3.0's cross-platform net10.0
        // asset dropped its bundled MP3 decoder entirely, so AudioFileReader threw "MP3 is not
        // supported by the cross-platform build of NAudio" for any .mp3 input. Decodes a real,
        // checked-in MP3 (a 1s 440Hz tone encoded via LAME, not a WAV renamed to .mp3) so this
        // exercises the actual Mp3FileReaderBase + NLayer decode path added to fix it. The
        // fixture is checked in rather than encoded at test time because LAME encoding requires
        // a Windows-only native library — the decode path under test is cross-platform and the
        // test needs to actually run on the cross-platform CI runners.
        const double durationSeconds = 1.0;
        var mp3Path = Path.Combine(AppContext.BaseDirectory, "Fixtures", "tone-440hz-1s.mp3");

        var samples = await AudioProcessor.LoadAudioAsync(mp3Path, TestContext.Current.CancellationToken);

        // Resampled to Whisper's 16kHz; tolerant range accounts for LAME encoder priming
        // and decoder frame alignment rather than asserting an exact sample count.
        var expectedSamples = (int)(durationSeconds * WhisperSampleRate);
        samples.Should().NotBeEmpty();
        samples.Length.Should().BeInRange((int)(expectedSamples * 0.9), (int)(expectedSamples * 1.3));
        samples.Should().Contain(s => s != 0f, "a decoded 440Hz tone should not be silent");
    }
}
