using AwesomeAssertions;
using LMSupply.Transcriber.Diarization;

namespace LMSupply.Transcriber.Tests.Diarization;

/// <summary>
/// The speaker-embedding front-end matches kaldi-native-fbank (the reference sherpa-onnx uses) on a fixed signal:
/// dither 0, snip_edges false, 80 bins, 20..7600 Hz, int16-scaled input. Reference values were produced with
/// kaldi-native-fbank 1.x from the same signal (two tones plus a fixed LCG noise, 1234 samples).
/// </summary>
public class KaldiFbankTests
{
    private static float[] Signal()
    {
        var x = new float[1234];
        long s = 12345;
        for (var i = 0; i < x.Length; i++)
        {
            s = (1103515245L * s + 12345L) % 2147483648L;
            var noise = s / 2147483648.0 - 0.5;
            x[i] = (float)(0.3 * Math.Sin(2 * Math.PI * 440 * i / 16000) + 0.1 * Math.Sin(2 * Math.PI * 1234 * i / 16000) + 0.05 * noise);
        }
        return x;
    }

    [Fact]
    public void MatchesKaldiNativeFbank()
    {
        var fbank = new KaldiFbank();
        var features = fbank.Compute(Signal());

        KaldiFbank.NumFrames(1234).Should().Be(8);
        features.Length.Should().Be(8 * 80);

        float At(int f, int m) => features[f * 80 + m];
        At(0, 0).Should().BeApproximately(15.332102f, 2e-3f);
        At(0, 40).Should().BeApproximately(19.335377f, 2e-3f);
        At(3, 10).Should().BeApproximately(15.163741f, 2e-3f);
        At(7, 79).Should().BeApproximately(19.593512f, 2e-3f);
        At(7, 0).Should().BeApproximately(14.899864f, 2e-3f);
        features.Sum().Should().BeApproximately(11565.963f, 0.5f);
    }
}
