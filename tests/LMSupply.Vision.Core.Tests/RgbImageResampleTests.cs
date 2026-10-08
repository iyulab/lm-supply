using AwesomeAssertions;
using LMSupply.Vision;

namespace LMSupply.Vision.Core.Tests;

/// <summary>
/// Resampling must reproduce the preprocessing the vision models were tuned against. The PNG fixtures under
/// Fixtures/resample are the output of the previous implementation (ImageSharp 3.1 bicubic resize) on the synthetic
/// sources below; the resampler must match them to within one level per channel (float rounding).
/// </summary>
public class RgbImageResampleTests
{
    public static TheoryData<string, int, int, int, int, int, bool> Cases => new()
    {
        { "down-stretch", 640, 480, 7, 224, 224, false },
        { "down-heavy", 1600, 1200, 8, 128, 96, false },
        { "up-stretch", 160, 120, 9, 384, 384, false },
        { "odd", 97, 211, 10, 64, 50, false },
        { "mixed", 300, 80, 11, 150, 160, false },
        { "identity", 300, 200, 12, 300, 200, false },
        { "ceil-crop", 640, 480, 19, 224, 224, true },
    };

    [Theory]
    [MemberData(nameof(Cases))]
    public void Resize_matches_the_reference_output(string name, int sourceWidth, int sourceHeight, int seed, int width, int height, bool centerCrop)
    {
        var source = Synthetic(sourceWidth, sourceHeight, seed);
        var expected = ImageLoader.Decode(File.ReadAllBytes(Path.Combine(AppContext.BaseDirectory, "Fixtures", "resample", name + ".png")));

        RgbImage actual;
        if (centerCrop)
        {
            // The preprocessor's center crop: scale to cover (ceiling), then the centered window.
            float scale = Math.Max((float)width / sourceWidth, (float)height / sourceHeight);
            int coverWidth = (int)Math.Ceiling(sourceWidth * scale), coverHeight = (int)Math.Ceiling(sourceHeight * scale);
            actual = source.Resize(coverWidth, coverHeight).Crop((coverWidth - width) / 2, (coverHeight - height) / 2, width, height);
        }
        else
        {
            actual = source.Resize(width, height);
        }

        actual.Width.Should().Be(expected.Width);
        actual.Height.Should().Be(expected.Height);
        var (max, mean) = Difference(actual, expected);
        max.Should().BeLessThanOrEqualTo(1);
        mean.Should().BeLessThan(0.05);
    }

    [Fact]
    public void Resize_to_the_same_size_returns_an_identical_copy()
    {
        var source = Synthetic(31, 17, 1);

        var copy = source.Resize(31, 17);

        copy.Should().NotBeSameAs(source);
        copy.Pixels.ToArray().Should().Equal(source.Pixels.ToArray());
    }

    [Fact]
    public void A_flat_image_stays_flat_at_any_scale()
    {
        var flat = RgbImage.Filled(53, 29, 12, 200, 99);

        foreach (var (w, h) in new[] { (7, 3), (200, 150), (53, 1), (1, 29) })
        {
            var resized = flat.Resize(w, h);
            for (var y = 0; y < h; y++)
            {
                for (var x = 0; x < w; x++)
                {
                    resized[x, y].Should().Be(((byte)12, (byte)200, (byte)99));
                }
            }
        }
    }

    [Theory]
    [InlineData(640, 480, 300, 300, 300, 225, 0, 37)]
    [InlineData(211, 397, 256, 128, 68, 128, 94, 0)]
    [InlineData(211, 397, 255, 128, 68, 128, 93, 0)]
    public void Letterbox_centers_the_scaled_image_on_black(int sw, int sh, int w, int h, int contentWidth, int contentHeight, int left, int top)
    {
        var source = RgbImage.Filled(sw, sh, 255, 255, 255);

        var boxed = source.Letterbox(w, h);

        boxed.Width.Should().Be(w);
        boxed.Height.Should().Be(h);
        boxed[left, top].Should().Be(((byte)255, (byte)255, (byte)255));
        boxed[left + contentWidth - 1, top + contentHeight - 1].Should().Be(((byte)255, (byte)255, (byte)255));
        if (left > 0)
        {
            boxed[left - 1, top].Should().Be(((byte)0, (byte)0, (byte)0));
        }

        if (top > 0)
        {
            boxed[left, top - 1].Should().Be(((byte)0, (byte)0, (byte)0));
        }
    }

    [Fact]
    public void Crop_copies_the_rectangle()
    {
        var source = Synthetic(40, 30, 3);

        var cropped = source.Crop(5, 7, 10, 4);

        for (var y = 0; y < 4; y++)
        {
            for (var x = 0; x < 10; x++)
            {
                cropped[x, y].Should().Be(source[x + 5, y + 7]);
            }
        }
    }

    [Fact]
    public void Crop_outside_the_image_is_rejected()
    {
        var source = new RgbImage(10, 10);

        var act = () => source.Crop(5, 5, 6, 2);

        act.Should().Throw<ArgumentOutOfRangeException>();
    }

    [Fact]
    public void Png_round_trip_is_lossless()
    {
        var source = Synthetic(57, 33, 4);

        var decoded = ImageLoader.Decode(ImageLoader.EncodePng(source));

        decoded.Pixels.ToArray().Should().Equal(source.Pixels.ToArray());
    }

    [Fact]
    public void Data_that_is_not_an_image_is_rejected_with_InvalidDataException()
    {
        var act = () => ImageLoader.Decode("not an image"u8);

        act.Should().Throw<InvalidDataException>();
    }

    [Theory]
    [InlineData(40000, 40000)]
    [InlineData(65536, 16385)] // 4 bytes x 65536 x 16385 = 2^32 + 262144: wraps to a small positive 32-bit size
    [InlineData(1000000, 1000000)]
    public void A_header_claiming_dimensions_too_large_to_hold_is_rejected_before_decoding(int width, int height)
    {
        // A valid PNG signature and IHDR claiming the dimensions, then a truncated IDAT: the size check must refuse it
        // from the header alone, before any buffer is sized from those dimensions.
        var header = PngHeader(width, height);
        ImageLoader.Identify(header).Should().Be((width, height), "the codec accepts the header, so the size check is what refuses it");

        var act = () => ImageLoader.Decode(header);

        act.Should().Throw<InvalidDataException>().WithMessage("*larger than can be decoded*");
    }

    private static byte[] PngHeader(int width, int height)
    {
        var ihdr = new byte[13];
        System.Buffers.Binary.BinaryPrimitives.WriteInt32BigEndian(ihdr, width);
        System.Buffers.Binary.BinaryPrimitives.WriteInt32BigEndian(ihdr.AsSpan(4), height);
        ihdr[8] = 8; // bit depth
        ihdr[9] = 2; // color type RGB
        using var png = new MemoryStream();
        png.Write([0x89, (byte)'P', (byte)'N', (byte)'G', 0x0D, 0x0A, 0x1A, 0x0A]);
        WriteChunk(png, "IHDR", ihdr);
        WriteChunk(png, "IDAT", [0x78, 0x9C, 0x00]);
        return png.ToArray();
    }

    private static void WriteChunk(Stream stream, string type, byte[] data)
    {
        Span<byte> length = stackalloc byte[4];
        System.Buffers.Binary.BinaryPrimitives.WriteInt32BigEndian(length, data.Length);
        stream.Write(length);
        var typeAndData = new byte[4 + data.Length];
        System.Text.Encoding.ASCII.GetBytes(type).CopyTo(typeAndData, 0);
        data.CopyTo(typeAndData, 4);
        stream.Write(typeAndData);
        Span<byte> crc = stackalloc byte[4];
        System.Buffers.Binary.BinaryPrimitives.WriteUInt32BigEndian(crc, Crc32(typeAndData));
        stream.Write(crc);
    }

    // PNG chunk CRC (ISO 3309 / ITU-T V.42), bitwise.
    private static uint Crc32(byte[] data)
    {
        var crc = 0xFFFFFFFFu;
        foreach (var b in data)
        {
            crc ^= b;
            for (var k = 0; k < 8; k++)
            {
                crc = (crc & 1) != 0 ? (crc >> 1) ^ 0xEDB88320u : crc >> 1;
            }
        }

        return ~crc;
    }

    // Gradients, hard edges, a 1-px checkerboard, a disc and a seeded noise patch — content that exposes kernel and
    // aliasing differences.
    internal static RgbImage Synthetic(int w, int h, int seed)
    {
        var rng = new Random(seed);
        var image = new RgbImage(w, h);
        for (var y = 0; y < h; y++)
        {
            for (var x = 0; x < w; x++)
            {
                byte r = (byte)(255 * x / Math.Max(1, w - 1)), g = (byte)(255 * y / Math.Max(1, h - 1)), b = (byte)(((x * 7) + (y * 3)) & 255);
                if (x > w / 8 && x < w / 3 && y > h / 8 && y < h / 3)
                {
                    (r, g, b) = (250, 20, 30);
                }

                if (x > w / 2 && y > h / 2 && x < 3 * w / 4 && y < 3 * h / 4)
                {
                    var c = ((x + y) & 1) == 0 ? (byte)255 : (byte)0;
                    (r, g, b) = (c, c, c);
                }

                double dx = x - (0.75 * w), dy = y - (0.3 * h);
                if ((dx * dx) + (dy * dy) < (0.12 * w) * (0.12 * w))
                {
                    (r, g, b) = (10, 200, 240);
                }

                if (x < w / 6 && y > 5 * h / 6)
                {
                    (r, g, b) = ((byte)rng.Next(256), (byte)rng.Next(256), (byte)rng.Next(256));
                }

                image[x, y] = (r, g, b);
            }
        }

        return image;
    }

    private static (int Max, double Mean) Difference(RgbImage a, RgbImage b)
    {
        var pa = a.Pixels;
        var pb = b.Pixels;
        int max = 0;
        long sum = 0;
        for (var i = 0; i < pa.Length; i++)
        {
            var d = Math.Abs(pa[i] - pb[i]);
            sum += d;
            max = Math.Max(max, d);
        }

        return (max, (double)sum / pa.Length);
    }
}
