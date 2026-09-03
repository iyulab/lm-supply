using System.Diagnostics;
using System.Numerics;
using MathNet.Numerics.IntegralTransforms;
using NAudio.Wave;
using NAudio.Wave.SampleProviders;
using NLayer.NAudioSupport;

namespace LMSupply.Transcriber.Audio;

/// <summary>
/// Processes audio files for Whisper model input.
/// Handles resampling, mono conversion, and mel spectrogram computation.
/// </summary>
internal static class AudioProcessor
{
    private const int WhisperSampleRate = 16000;
    private const int HopLength = 160;
    private const int ChunkLengthSeconds = 30;
    private const int NumSamples = WhisperSampleRate * ChunkLengthSeconds; // 480000

    /// <summary>
    /// Loads audio from a file and converts to float samples at 16kHz mono.
    /// </summary>
    /// <param name="audioPath">Path to the audio file.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>Float array of audio samples normalized to [-1, 1].</returns>
    public static async Task<float[]> LoadAudioAsync(string audioPath, CancellationToken cancellationToken = default)
    {
        return await Task.Run(() => LoadAudio(audioPath), cancellationToken);
    }

    /// <summary>
    /// Loads audio from a stream and converts to float samples at 16kHz mono.
    /// </summary>
    /// <param name="audioStream">Stream containing audio data.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>Float array of audio samples normalized to [-1, 1].</returns>
    public static async Task<float[]> LoadAudioAsync(Stream audioStream, CancellationToken cancellationToken = default)
    {
        // Copy to memory stream for NAudio compatibility
        using var memoryStream = new MemoryStream();
        await audioStream.CopyToAsync(memoryStream, cancellationToken);
        memoryStream.Position = 0;

        return await Task.Run(() => LoadAudioFromStream(memoryStream), cancellationToken);
    }

    /// <summary>
    /// Loads audio from a byte array and converts to float samples at 16kHz mono.
    /// </summary>
    /// <param name="audioData">Byte array containing audio data.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>Float array of audio samples normalized to [-1, 1].</returns>
    public static async Task<float[]> LoadAudioAsync(byte[] audioData, CancellationToken cancellationToken = default)
    {
        using var memoryStream = new MemoryStream(audioData);
        return await Task.Run(() => LoadAudioFromStream(memoryStream), cancellationToken);
    }

    private static float[] LoadAudio(string audioPath)
    {
        // NAudio 3.0's cross-platform build dropped its bundled MP3 decoder (docket
        // iyulab/lm-supply#169) — AudioFileReader throws for .mp3 input on this project's plain
        // net10.0 TFM. NLayer.NAudioSupport plugs a pure-managed MP3 decoder into NAudio's own
        // Mp3FileReader, keeping this fully cross-platform (no Windows-only reader reintroduced,
        // same reasoning as the WdlResamplingSampleProvider choice below).
        using WaveStream reader = string.Equals(Path.GetExtension(audioPath), ".mp3", StringComparison.OrdinalIgnoreCase)
            ? new Mp3FileReaderBase(audioPath, wf => new Mp3FrameDecompressor(wf))
            : new AudioFileReader(audioPath);
        return ProcessAudioReader(reader);
    }

    private static float[] LoadAudioFromStream(MemoryStream stream)
    {
        using var reader = new WaveFileReader(stream);
        var waveProvider = reader.ToSampleProvider();
        return ProcessSampleProvider(waveProvider, reader.TotalTime);
    }

    private static float[] ProcessAudioReader(WaveStream reader)
    {
        // AudioFileReader already implements ISampleProvider directly (with volume control);
        // Mp3FileReader does not, so it needs the ToSampleProvider() wrap ToMono() requires.
        ISampleProvider sampleProvider = (reader as ISampleProvider ?? reader.ToSampleProvider()).ToMono();

        // Resample if necessary. WdlResamplingSampleProvider (pure managed, cross-platform)
        // replaces MediaFoundationResampler here: the latter wraps a Windows Media Foundation COM
        // API and is unavailable on this project's cross-platform net10.0 TFM (NAudio 3.0 gates it
        // behind a Windows-specific surface) — chasing it into a Windows-conditional build would
        // reintroduce a platform dependency this library never actually needed.
        if (sampleProvider.WaveFormat.SampleRate != WhisperSampleRate)
        {
            sampleProvider = new WdlResamplingSampleProvider(sampleProvider, WhisperSampleRate);
        }

        return ProcessSampleProvider(sampleProvider, reader.TotalTime);
    }

    private static float[] ProcessSampleProvider(ISampleProvider provider, TimeSpan duration)
    {
        var totalSamples = (int)(duration.TotalSeconds * WhisperSampleRate);
        var samples = new float[totalSamples];
        var read = provider.Read(samples.AsSpan(0, totalSamples));

        if (read < totalSamples)
        {
            Array.Resize(ref samples, read);
        }

        return samples;
    }

    /// <summary>
    /// Pads or truncates audio to exactly 30 seconds (480000 samples).
    /// </summary>
    /// <param name="samples">Input audio samples.</param>
    /// <returns>Padded/truncated audio samples.</returns>
    public static float[] PadOrTruncate(float[] samples)
    {
        if (samples.Length == NumSamples)
            return samples;

        var result = new float[NumSamples];

        if (samples.Length > NumSamples)
        {
            // Truncate
            Array.Copy(samples, result, NumSamples);
        }
        else
        {
            // Pad with zeros
            Array.Copy(samples, result, samples.Length);
        }

        return result;
    }

    /// <summary>
    /// Splits audio into 30-second chunks for processing.
    /// </summary>
    /// <param name="samples">Input audio samples.</param>
    /// <returns>List of 30-second audio chunks.</returns>
    public static List<float[]> SplitIntoChunks(float[] samples)
    {
        var chunks = new List<float[]>();
        var position = 0;

        while (position < samples.Length)
        {
            var chunkSize = Math.Min(NumSamples, samples.Length - position);
            var chunk = new float[NumSamples];
            Array.Copy(samples, position, chunk, 0, chunkSize);
            chunks.Add(chunk);
            position += NumSamples;
        }

        return chunks;
    }

    /// <summary>
    /// Computes the log-mel spectrogram for Whisper input.
    /// </summary>
    /// <param name="samples">Audio samples at 16kHz.</param>
    /// <param name="numMelBins">Number of mel frequency bins (80 or 128).</param>
    /// <returns>Log-mel spectrogram tensor [1, numMelBins, 3000].</returns>
    public static float[] ComputeLogMelSpectrogram(float[] samples, int numMelBins = 80)
    {
        // Pad to 30 seconds if needed
        samples = PadOrTruncate(samples);

        // Compute STFT
        const int nFft = 400;
        const int hopLength = HopLength;
        const int numFrames = 3000; // 30 seconds * 16000 / 160

        var stft = ComputeStft(samples, nFft, hopLength, numFrames);

        // Convert to mel spectrogram
        var melFilters = CreateMelFilterBank(nFft, numMelBins, WhisperSampleRate);
        var melSpec = ApplyMelFilters(stft, melFilters, numMelBins, numFrames);

        // Convert to log scale
        var logMelSpec = new float[numMelBins * numFrames];
        for (int i = 0; i < logMelSpec.Length; i++)
        {
            logMelSpec[i] = MathF.Log10(MathF.Max(melSpec[i], 1e-10f));
        }

        // Normalize
        var maxVal = logMelSpec.Max();
        var minValBefore = logMelSpec.Min();
        for (int i = 0; i < logMelSpec.Length; i++)
        {
            logMelSpec[i] = MathF.Max(logMelSpec[i], maxVal - 8.0f);
            logMelSpec[i] = (logMelSpec[i] + 4.0f) / 4.0f;
        }

        // Debug: Log mel spectrogram statistics

        return logMelSpec;
    }

    private static float[] ComputeStft(float[] samples, int nFft, int hopLength, int numFrames)
    {
        var window = CreateHannWindow(nFft);
        var stft = new float[(nFft / 2 + 1) * numFrames];
        var fftBuffer = new Complex[nFft];

        for (int frame = 0; frame < numFrames; frame++)
        {
            var start = frame * hopLength;

            for (int i = 0; i < nFft; i++)
            {
                float sample = (start + i < samples.Length) ? samples[start + i] * window[i] : 0f;
                fftBuffer[i] = new Complex(sample, 0);
            }

            Fourier.Forward(fftBuffer, FourierOptions.NoScaling);

            // Whisper expects power spectrum: |STFT|² = magnitude squared
            for (int k = 0; k <= nFft / 2; k++)
            {
                var re = (float)fftBuffer[k].Real;
                var im = (float)fftBuffer[k].Imaginary;
                stft[k * numFrames + frame] = re * re + im * im;
            }
        }

        return stft;
    }

    private static float[] CreateHannWindow(int length)
    {
        var window = new float[length];
        for (int i = 0; i < length; i++)
        {
            window[i] = 0.5f * (1 - MathF.Cos(2 * MathF.PI * i / length));
        }
        return window;
    }

    private static float[,] CreateMelFilterBank(int nFft, int numMelBins, int sampleRate)
    {
        var melFilters = new float[numMelBins, nFft / 2 + 1];

        var fMin = 0f;
        var fMax = sampleRate / 2f;
        var melMin = HzToMel(fMin);
        var melMax = HzToMel(fMax);

        var melPoints = new float[numMelBins + 2];
        for (int i = 0; i < melPoints.Length; i++)
        {
            melPoints[i] = MelToHz(melMin + (melMax - melMin) * i / (numMelBins + 1));
        }

        var fftFreqs = new float[nFft / 2 + 1];
        for (int i = 0; i < fftFreqs.Length; i++)
        {
            fftFreqs[i] = (float)sampleRate * i / nFft;
        }

        for (int m = 0; m < numMelBins; m++)
        {
            for (int k = 0; k < fftFreqs.Length; k++)
            {
                if (fftFreqs[k] >= melPoints[m] && fftFreqs[k] <= melPoints[m + 1])
                {
                    melFilters[m, k] = (fftFreqs[k] - melPoints[m]) / (melPoints[m + 1] - melPoints[m]);
                }
                else if (fftFreqs[k] >= melPoints[m + 1] && fftFreqs[k] <= melPoints[m + 2])
                {
                    melFilters[m, k] = (melPoints[m + 2] - fftFreqs[k]) / (melPoints[m + 2] - melPoints[m + 1]);
                }
            }
        }

        // Apply Slaney normalization (librosa's default for Whisper compatibility)
        // This normalizes each filter to have approximately constant energy per channel
        // Formula: enorm = 2.0 / (mel_f[m+2] - mel_f[m])
        for (int m = 0; m < numMelBins; m++)
        {
            float enorm = 2.0f / (melPoints[m + 2] - melPoints[m]);
            for (int k = 0; k < fftFreqs.Length; k++)
            {
                melFilters[m, k] *= enorm;
            }
        }

        return melFilters;
    }

    private static float[] ApplyMelFilters(float[] stft, float[,] melFilters, int numMelBins, int numFrames)
    {
        var numFreqBins = stft.Length / numFrames;
        var melSpec = new float[numMelBins * numFrames];

        for (int m = 0; m < numMelBins; m++)
        {
            for (int t = 0; t < numFrames; t++)
            {
                float sum = 0;
                for (int k = 0; k < numFreqBins; k++)
                {
                    sum += melFilters[m, k] * stft[k * numFrames + t];
                }
                melSpec[m * numFrames + t] = sum;
            }
        }

        return melSpec;
    }

    // Slaney mel scale (librosa default, used by OpenAI Whisper)
    // Linear below 1000 Hz, logarithmic above
    private const float SlaneyFMin = 0f;
    private const float SlaneyFSp = 200f / 3f; // 66.667 Hz per mel below 1000 Hz
    private const float SlaneyMinLogHz = 1000f;
    private const float SlaneyMinLogMel = (SlaneyMinLogHz - SlaneyFMin) / SlaneyFSp; // 15.0
    private static readonly float SlaneyLogStep = MathF.Log(6.4f) / 27f;

    private static float HzToMel(float hz)
    {
        if (hz < SlaneyMinLogHz)
            return (hz - SlaneyFMin) / SlaneyFSp;
        return SlaneyMinLogMel + MathF.Log(hz / SlaneyMinLogHz) / SlaneyLogStep;
    }

    private static float MelToHz(float mel)
    {
        if (mel < SlaneyMinLogMel)
            return SlaneyFMin + SlaneyFSp * mel;
        return SlaneyMinLogHz * MathF.Exp(SlaneyLogStep * (mel - SlaneyMinLogMel));
    }

    /// <summary>
    /// Extracts a 30-second chunk starting at the given sample position.
    /// The chunk is zero-padded if there aren't enough remaining samples.
    /// </summary>
    /// <param name="samples">Full audio samples.</param>
    /// <param name="startPosition">Start position in samples.</param>
    /// <returns>A 30-second (480000 samples) chunk.</returns>
    public static float[] ExtractChunk(float[] samples, int startPosition)
    {
        var chunk = new float[NumSamples];
        var remaining = Math.Min(NumSamples, samples.Length - startPosition);
        if (remaining > 0)
            Array.Copy(samples, startPosition, chunk, 0, remaining);
        return chunk;
    }

    /// <summary>
    /// Converts seconds to sample position.
    /// </summary>
    public static int SecondsToSamples(double seconds) => (int)(seconds * WhisperSampleRate);

    /// <summary>
    /// Gets the duration of audio samples in seconds.
    /// </summary>
    /// <param name="samples">Audio samples.</param>
    /// <returns>Duration in seconds.</returns>
    public static double GetDurationSeconds(float[] samples) => (double)samples.Length / WhisperSampleRate;
}
