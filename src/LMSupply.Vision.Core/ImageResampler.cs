namespace LMSupply.Vision;

/// <summary>
/// Separable bicubic resampling of RGB24 buffers. Each output sample is a normalized weighted sum of the source samples
/// under a Keys cubic kernel (a = -0.5, radius 2) centered on the output pixel's position in source coordinates; when
/// shrinking, the kernel is stretched by the scale factor so it averages every source pixel it covers. Samples past the
/// image edge are left out and the remaining weights renormalized. The horizontal pass runs first into a float buffer,
/// then the vertical pass rounds to bytes.
/// </summary>
internal static class ImageResampler
{
    private const double Radius = 2;
    private const double Tolerance = 1e-8;

    public static byte[] Resize(byte[] source, int sourceWidth, int sourceHeight, int width, int height)
    {
        var horizontal = Kernels(sourceWidth, width);
        var vertical = Kernels(sourceHeight, height);

        var firstPass = new float[sourceHeight * width * 3];
        for (var y = 0; y < sourceHeight; y++)
        {
            var sourceRow = source.AsSpan(y * sourceWidth * 3, sourceWidth * 3);
            var outRow = firstPass.AsSpan(y * width * 3, width * 3);
            for (var x = 0; x < width; x++)
            {
                var (start, weights) = horizontal[x];
                float r = 0, g = 0, b = 0;
                for (var k = 0; k < weights.Length; k++)
                {
                    var at = (start + k) * 3;
                    var w = weights[k];
                    r += w * sourceRow[at];
                    g += w * sourceRow[at + 1];
                    b += w * sourceRow[at + 2];
                }

                outRow[x * 3] = r;
                outRow[(x * 3) + 1] = g;
                outRow[(x * 3) + 2] = b;
            }
        }

        var result = new byte[height * width * 3];
        for (var y = 0; y < height; y++)
        {
            var (start, weights) = vertical[y];
            var outRow = result.AsSpan(y * width * 3, width * 3);
            for (var x = 0; x < width * 3; x++)
            {
                float sum = 0;
                for (var k = 0; k < weights.Length; k++)
                {
                    sum += weights[k] * firstPass[((start + k) * width * 3) + x];
                }

                outRow[x] = ToByte(sum);
            }
        }

        return result;
    }

    /// <summary>For each output index along one axis: the first source index and the normalized weights from there.</summary>
    internal static (int Start, float[] Weights)[] Kernels(int sourceSize, int size)
    {
        var ratio = (double)sourceSize / size;
        var scale = Math.Max(ratio, 1);
        var radius = (int)TolerantCeiling(scale * Radius);
        var kernels = new (int, float[])[size];
        var values = new double[(2 * radius) + 2];
        for (var i = 0; i < size; i++)
        {
            var center = ((i + 0.5) * ratio) - 0.5;
            var left = Math.Max((int)TolerantCeiling(center - radius), 0);
            var right = Math.Min((int)TolerantFloor(center + radius), sourceSize - 1);
            var count = right - left + 1;
            double sum = 0;
            for (var j = 0; j < count; j++)
            {
                values[j] = Cubic((float)((left + j - center) / scale));
                sum += values[j];
            }

            var weights = new float[count];
            for (var j = 0; j < count; j++)
            {
                weights[j] = sum > 0 ? (float)(values[j] / sum) : 0f;
            }

            kernels[i] = (left, weights);
        }

        return kernels;
    }

    // Keys cubic convolution with a = -0.5 (Catmull-Rom).
    private static double Cubic(double x)
    {
        x = Math.Abs(x);
        if (x <= 1)
        {
            return (((1.5 * x) - 2.5) * x * x) + 1;
        }

        return x < 2 ? (((((-0.5 * x) + 2.5) * x) - 4) * x) + 2 : 0;
    }

    // Window bounds land on integers often; treat values within 1e-8 of an integer as that integer.
    private static double TolerantCeiling(double a) => Math.Abs(Math.IEEERemainder(a, 1)) < Tolerance ? Math.Round(a) : Math.Ceiling(a);

    private static double TolerantFloor(double a) => Math.Abs(Math.IEEERemainder(a, 1)) < Tolerance ? Math.Round(a) : Math.Floor(a);

    private static byte ToByte(float value)
    {
        var scaled = value + 0.5f;
        return scaled <= 0 ? (byte)0 : scaled >= 255 ? (byte)255 : (byte)scaled;
    }
}
