namespace LMSupply.ImageGenerator.Pipeline;

/// <summary>
/// The sinusoidal embedding of the guidance weight a latent consistency model's UNet takes as <c>timestep_cond</c>.
/// Such a model was distilled with classifier-free guidance folded into it, so the guidance weight is an input rather than
/// a second forward pass.
/// </summary>
internal static class GuidanceEmbedding
{
    /// <summary>
    /// Embeds <c>w = guidanceScale - 1</c> the way the reference pipeline does: the weight is scaled by 1000, multiplied
    /// by geometrically spaced frequencies, and the sines of the products are followed by their cosines.
    /// </summary>
    /// <param name="guidanceScale">The guidance scale of the request (1 means no guidance).</param>
    /// <param name="dimension">The model's <c>timestep_cond</c> width.</param>
    public static float[] Create(float guidanceScale, int dimension)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(dimension, 2);

        var w = (guidanceScale - 1.0f) * 1000.0f;
        var half = dimension / 2;
        var step = MathF.Log(10000.0f) / (half - 1);

        var embedding = new float[dimension];
        for (var i = 0; i < half; i++)
        {
            var angle = w * MathF.Exp(i * -step);
            embedding[i] = MathF.Sin(angle);
            embedding[half + i] = MathF.Cos(angle);
        }

        return embedding;
    }
}
