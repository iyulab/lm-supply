using NAudio.Wave;

namespace LMSupply.Integration.Tests.Helpers;

/// <summary>
/// Generates test data (images, audio, text) for functional tests.
/// All data is created programmatically - no external files required.
/// </summary>
internal static class TestDataHelper
{
    // ── Text Test Data ──────────────────────────────────────────────

    public const string EnglishShort = "Hello, this is a test.";
    public const string KoreanShort = "안녕하세요, 테스트입니다.";
    public const string EmojiText = "I love 🐱 cats! 🎉";

    public static string GenerateLongText(int wordCount = 10_000)
    {
        var words = new[] { "the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog",
            "machine", "learning", "artificial", "intelligence", "neural", "network", "deep",
            "model", "training", "data", "algorithm", "prediction", "classification", "regression" };
        var rng = new Random(42);
        return string.Join(' ', Enumerable.Range(0, wordCount).Select(_ => words[rng.Next(words.Length)]));
    }

    // ── Image Test Data ─────────────────────────────────────────────

    /// <summary>
    /// Creates a BMP image with a gradient pattern.
    /// </summary>
    public static byte[] CreateGradientBmp(int width, int height)
    {
        var rowSize = ((width * 3 + 3) / 4) * 4;
        var imageSize = rowSize * height;
        var fileSize = 54 + imageSize;

        using var ms = new MemoryStream();
        using var bw = new BinaryWriter(ms);

        // BMP Header
        bw.Write((byte)'B');
        bw.Write((byte)'M');
        bw.Write(fileSize);
        bw.Write(0);
        bw.Write(54);

        // DIB Header (BITMAPINFOHEADER)
        bw.Write(40);
        bw.Write(width);
        bw.Write(height);
        bw.Write((short)1);
        bw.Write((short)24);
        bw.Write(0);
        bw.Write(imageSize);
        bw.Write(2835);
        bw.Write(2835);
        bw.Write(0);
        bw.Write(0);

        var rowPadding = rowSize - width * 3;
        for (var y = 0; y < height; y++)
        {
            for (var x = 0; x < width; x++)
            {
                byte r = (byte)(x * 255 / Math.Max(width - 1, 1));
                byte g = (byte)(y * 255 / Math.Max(height - 1, 1));
                byte b = (byte)((x + y) * 127 / Math.Max(width + height - 2, 1));
                bw.Write(b);
                bw.Write(g);
                bw.Write(r);
            }
            for (var p = 0; p < rowPadding; p++)
                bw.Write((byte)0);
        }

        return ms.ToArray();
    }

    /// <summary>
    /// Creates a solid-color BMP image.
    /// </summary>
    public static byte[] CreateSolidBmp(int width, int height, byte r = 255, byte g = 255, byte b = 255)
    {
        var rowSize = ((width * 3 + 3) / 4) * 4;
        var imageSize = rowSize * height;
        var fileSize = 54 + imageSize;

        using var ms = new MemoryStream();
        using var bw = new BinaryWriter(ms);

        bw.Write((byte)'B');
        bw.Write((byte)'M');
        bw.Write(fileSize);
        bw.Write(0);
        bw.Write(54);

        bw.Write(40);
        bw.Write(width);
        bw.Write(height);
        bw.Write((short)1);
        bw.Write((short)24);
        bw.Write(0);
        bw.Write(imageSize);
        bw.Write(2835);
        bw.Write(2835);
        bw.Write(0);
        bw.Write(0);

        var rowPadding = rowSize - width * 3;
        for (var y = 0; y < height; y++)
        {
            for (var x = 0; x < width; x++)
            {
                bw.Write(b);
                bw.Write(g);
                bw.Write(r);
            }
            for (var p = 0; p < rowPadding; p++)
                bw.Write((byte)0);
        }

        return ms.ToArray();
    }

    /// <summary>
    /// Creates a 1x1 pixel BMP image.
    /// </summary>
    public static byte[] CreateTinyBmp() => CreateSolidBmp(1, 1, 128, 128, 128);

    /// <summary>
    /// Creates a BMP image with high-contrast geometric shapes.
    /// Better suited for detection/captioning than gradients — has sharp edges,
    /// distinct regions, and multiple color areas.
    /// </summary>
    public static byte[] CreateHighContrastBmp(int width = 640, int height = 480)
    {
        var rowSize = ((width * 3 + 3) / 4) * 4;
        var imageSize = rowSize * height;
        var fileSize = 54 + imageSize;

        using var ms = new MemoryStream();
        using var bw = new BinaryWriter(ms);

        // BMP Header
        bw.Write((byte)'B');
        bw.Write((byte)'M');
        bw.Write(fileSize);
        bw.Write(0);
        bw.Write(54);

        // DIB Header
        bw.Write(40);
        bw.Write(width);
        bw.Write(height);
        bw.Write((short)1);
        bw.Write((short)24);
        bw.Write(0);
        bw.Write(imageSize);
        bw.Write(2835);
        bw.Write(2835);
        bw.Write(0);
        bw.Write(0);

        var rowPadding = rowSize - width * 3;
        for (var y = 0; y < height; y++)
        {
            for (var x = 0; x < width; x++)
            {
                byte rv, gv, bv;
                // Upper half: sky-like blue
                if (y >= height / 2)
                {
                    rv = 135; gv = 206; bv = 235;
                }
                // Lower half: green ground
                else
                {
                    rv = 34; gv = 139; bv = 34;
                }

                // Draw a bright rectangle in the center (potential object)
                if (x > width / 4 && x < width * 3 / 4 &&
                    y > height / 4 && y < height * 3 / 4)
                {
                    rv = 255; gv = 200; bv = 50;
                }

                // Draw a small red square (another potential object)
                if (x > width / 10 && x < width / 5 &&
                    y > height / 3 && y < height * 2 / 3)
                {
                    rv = 220; gv = 20; bv = 60;
                }

                bw.Write(bv);
                bw.Write(gv);
                bw.Write(rv);
            }
            for (var p = 0; p < rowPadding; p++)
                bw.Write((byte)0);
        }

        return ms.ToArray();
    }

    // ── Audio Test Data ─────────────────────────────────────────────

    /// <summary>
    /// Creates a WAV file with a sine wave tone.
    /// </summary>
    public static byte[] CreateToneWav(int sampleRate = 16000, float durationSeconds = 1.0f, float frequency = 440f)
    {
        var numSamples = (int)(sampleRate * durationSeconds);
        var pcmData = new byte[numSamples * 2];

        for (var i = 0; i < numSamples; i++)
        {
            var sample = 0.1f * MathF.Sin(2 * MathF.PI * frequency * i / sampleRate);
            var pcm16 = (short)(Math.Clamp(sample, -1.0f, 1.0f) * 32767);
            pcmData[i * 2] = (byte)(pcm16 & 0xFF);
            pcmData[i * 2 + 1] = (byte)((pcm16 >> 8) & 0xFF);
        }

        return WrapPcmAsWav(pcmData, sampleRate, 1, 16);
    }

    /// <summary>
    /// Creates a WAV file with silence.
    /// </summary>
    public static byte[] CreateSilenceWav(int sampleRate = 16000, float durationSeconds = 3.0f)
    {
        var numSamples = (int)(sampleRate * durationSeconds);
        var pcmData = new byte[numSamples * 2];
        return WrapPcmAsWav(pcmData, sampleRate, 1, 16);
    }

    /// <summary>
    /// Creates an MP3 file from a sine wave tone using the cross-platform LAME encoder
    /// (NAudio 3.0's plain net10.0 TFM no longer carries Windows Media Foundation types —
    /// see LMSupply.Console.Host's AudioConverter for the same encoder pattern).
    /// Returns null if encoding fails for any reason.
    /// </summary>
    public static byte[]? TryCreateToneMp3(int sampleRate = 44100, float durationSeconds = 1.0f, float frequency = 440f)
    {
        try
        {
            var wavBytes = CreateToneWav(sampleRate, durationSeconds, frequency);
            using var wavStream = new MemoryStream(wavBytes);
            using var reader = new WaveFileReader(wavStream);
            using var mp3Stream = new MemoryStream();
            using (var writer = new NAudio.Lame.LameMP3FileWriter(mp3Stream, reader.WaveFormat, NAudio.Lame.LAMEPreset.STANDARD))
            {
                reader.CopyTo(writer);
            }
            return mp3Stream.ToArray();
        }
        catch
        {
            return null;
        }
    }

    /// <summary>
    /// Saves bytes to a temp file with the given extension and returns the path.
    /// Caller is responsible for deleting the file.
    /// </summary>
    public static string SaveToTempFile(byte[] data, string extension)
    {
        var path = Path.Combine(Path.GetTempPath(), $"lmsupply_test_{Guid.NewGuid()}{extension}");
        File.WriteAllBytes(path, data);
        return path;
    }

    private static byte[] WrapPcmAsWav(byte[] pcmData, int sampleRate, short channels, short bitsPerSample)
    {
        using var ms = new MemoryStream();
        using var writer = new BinaryWriter(ms);

        var byteRate = sampleRate * channels * bitsPerSample / 8;
        var blockAlign = (short)(channels * bitsPerSample / 8);

        writer.Write("RIFF"u8);
        writer.Write(36 + pcmData.Length);
        writer.Write("WAVE"u8);

        writer.Write("fmt "u8);
        writer.Write(16);
        writer.Write((short)1); // PCM
        writer.Write(channels);
        writer.Write(sampleRate);
        writer.Write(byteRate);
        writer.Write(blockAlign);
        writer.Write(bitsPerSample);

        writer.Write("data"u8);
        writer.Write(pcmData.Length);
        writer.Write(pcmData);

        return ms.ToArray();
    }
}
