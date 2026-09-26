using System.Numerics;
using MathNet.Numerics.IntegralTransforms;

namespace LMSupply.Transcriber.Diarization;

/// <summary>
/// Kaldi-compatible log mel filterbank (the front-end speaker-embedding models are trained on): 25 ms frames every
/// 10 ms, DC removal, pre-emphasis 0.97, povey window, 512-point power spectrum, triangular mel bins on Kaldi's
/// mel scale, natural log floored at float epsilon. Mirrors kaldi-native-fbank with the settings sherpa-onnx uses
/// for speaker embeddings: no dither, <c>snip_edges = false</c> (frames centred on the shift, edges reflected),
/// high frequency = Nyquist − 400 Hz.
/// </summary>
internal sealed class KaldiFbank
{
    private const int SampleRate = 16000;
    private const int FrameLength = 400;   // 25 ms
    private const int FrameShift = 160;    // 10 ms
    private const int FftSize = 512;
    private const float PreEmphasis = 0.97f;
    private const float LowFreq = 20f;
    private const float HighFreq = SampleRate / 2f - 400f;

    private readonly int _numBins;
    private readonly float[] _window;
    private readonly (int First, float[] Weights)[] _melBanks;

    public KaldiFbank(int numBins = 80)
    {
        _numBins = numBins;
        _window = new float[FrameLength];
        for (var i = 0; i < FrameLength; i++)
            _window[i] = (float)Math.Pow(0.5 - 0.5 * Math.Cos(2 * Math.PI * i / (FrameLength - 1)), 0.85);
        _melBanks = CreateMelBanks(numBins);
    }

    public int NumBins => _numBins;

    /// <summary>Number of frames for <paramref name="numSamples"/> samples (Kaldi, snip_edges = false).</summary>
    public static int NumFrames(int numSamples) => (numSamples + FrameShift / 2) / FrameShift;

    /// <summary>
    /// Computes the features of 16 kHz samples in the model's input scale (<paramref name="scale"/> = 32768 for
    /// models trained on int16-range audio). Returns frame-major <c>[frames, numBins]</c>.
    /// </summary>
    public float[] Compute(ReadOnlySpan<float> samples, float scale = 32768f)
    {
        var frames = NumFrames(samples.Length);
        var output = new float[frames * _numBins];
        var frame = new float[FrameLength];
        var fft = new Complex[FftSize];
        var power = new float[FftSize / 2];

        for (var f = 0; f < frames; f++)
        {
            // snip_edges = false: the frame is centred on f * shift + shift / 2; samples outside are reflected.
            var start = f * FrameShift + FrameShift / 2 - FrameLength / 2;
            for (var i = 0; i < FrameLength; i++)
                frame[i] = samples[Reflect(start + i, samples.Length)] * scale;

            var mean = 0f;
            for (var i = 0; i < FrameLength; i++)
                mean += frame[i];
            mean /= FrameLength;
            for (var i = 0; i < FrameLength; i++)
                frame[i] -= mean;

            for (var i = FrameLength - 1; i > 0; i--)
                frame[i] -= PreEmphasis * frame[i - 1];
            frame[0] -= PreEmphasis * frame[0];

            for (var i = 0; i < FftSize; i++)
                fft[i] = i < FrameLength ? new Complex(frame[i] * _window[i], 0) : Complex.Zero;
            Fourier.Forward(fft, FourierOptions.NoScaling);
            for (var k = 0; k < FftSize / 2; k++)
                power[k] = (float)(fft[k].Real * fft[k].Real + fft[k].Imaginary * fft[k].Imaginary);

            for (var m = 0; m < _numBins; m++)
            {
                var (first, weights) = _melBanks[m];
                double energy = 0;
                for (var j = 0; j < weights.Length; j++)
                    energy += weights[j] * power[first + j];
                output[f * _numBins + m] = MathF.Log(Math.Max((float)energy, 1.1920929e-07f));
            }
        }

        return output;
    }

    private static int Reflect(int index, int length)
    {
        // Kaldi's ExtractWindow: mirror at both ends without repeating the edge sample.
        while (index < 0 || index >= length)
        {
            if (index < 0)
                index = -index - 1;
            else
                index = 2 * length - 1 - index;
        }
        return index;
    }

    private static float MelScale(float freq) => 1127f * MathF.Log(1f + freq / 700f);

    private static (int First, float[] Weights)[] CreateMelBanks(int numBins)
    {
        const int numFftBins = FftSize / 2;
        const float fftBinWidth = (float)SampleRate / FftSize;
        var melLow = MelScale(LowFreq);
        var melHigh = MelScale(HighFreq);
        var melDelta = (melHigh - melLow) / (numBins + 1);

        var banks = new (int, float[])[numBins];
        for (var b = 0; b < numBins; b++)
        {
            var left = melLow + b * melDelta;
            var center = melLow + (b + 1) * melDelta;
            var right = melLow + (b + 2) * melDelta;

            var weights = new float[numFftBins];
            int first = -1, last = -1;
            for (var i = 0; i < numFftBins; i++)
            {
                var mel = MelScale(fftBinWidth * i);
                if (mel > left && mel < right)
                {
                    weights[i] = mel <= center ? (mel - left) / (center - left) : (right - mel) / (right - center);
                    if (first < 0)
                        first = i;
                    last = i;
                }
            }

            banks[b] = first < 0 ? (0, []) : (first, weights[first..(last + 1)]);
        }

        return banks;
    }
}
