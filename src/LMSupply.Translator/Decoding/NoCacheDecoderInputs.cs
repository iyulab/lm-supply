using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace LMSupply.Translator.Decoding;

/// <summary>
/// The extra inputs a decoder export needs when it is run without KV caching. A merged export (one graph with a
/// <c>use_cache_branch</c> switch) still declares every <c>past_key_values.*</c> input even when the switch is off,
/// and ONNX Runtime refuses a run that leaves any of them out. With the switch off they are never read, so a
/// one-position zero tensor of the declared shape stands in for each. A plain decoder export has none of these
/// inputs and gets nothing added.
/// </summary>
internal sealed class NoCacheDecoderInputs
{
    private static readonly bool[] s_false = [false];
    private static readonly int[] s_scalarShape = [1];

    private readonly NamedOnnxValue[] _values;

    public NoCacheDecoderInputs(IReadOnlyDictionary<string, NodeMetadata> inputMetadata)
    {
        var values = new List<NamedOnnxValue>();

        if (inputMetadata.ContainsKey("use_cache_branch"))
            values.Add(NamedOnnxValue.CreateFromTensor("use_cache_branch", new DenseTensor<bool>(s_false, s_scalarShape)));

        foreach (var (name, metadata) in inputMetadata)
        {
            if (!name.StartsWith("past_key_values.", StringComparison.Ordinal))
                continue;

            // Symbolic dimensions (batch, past length) become 1: one sequence, one ignored past position
            var shape = metadata.Dimensions.Select(d => d > 0 ? d : 1).ToArray();
            var length = shape.Aggregate(1, (a, b) => a * b);

            values.Add(metadata.ElementType switch
            {
                var t when t == typeof(float) => NamedOnnxValue.CreateFromTensor(name, new DenseTensor<float>(new float[length], shape)),
                var t when t == typeof(Float16) => NamedOnnxValue.CreateFromTensor(name, new DenseTensor<Float16>(new Float16[length], shape)),
                var t => throw new NotSupportedException(
                    $"Decoder input '{name}' has element type {t.Name}; only float and float16 past key/value inputs are supported."),
            });
        }

        _values = [.. values];
    }

    /// <summary>The number of inputs added per run (0 for a plain decoder export).</summary>
    public int Count => _values.Length;

    /// <summary>Appends the no-cache inputs to one decoder run's inputs.</summary>
    public void AddTo(List<NamedOnnxValue> inputs) => inputs.AddRange(_values);
}
