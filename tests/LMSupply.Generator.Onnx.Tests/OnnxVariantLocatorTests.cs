using AwesomeAssertions;
using LMSupply.Generator.Internal;

namespace LMSupply.Generator.Onnx.Tests;

/// <summary>
/// The variant search shared by the hub load and <see cref="LocalGenerator.LoadFromPathAsync"/>: a downloaded repository
/// folder resolves to the subfolder that holds the GenAI model, preferring the provider's variant.
/// </summary>
public sealed class OnnxVariantLocatorTests : IDisposable
{
    private readonly string _root = Path.Combine(Path.GetTempPath(), "onnx-variant-" + Guid.NewGuid().ToString("N"));

    public void Dispose()
    {
        if (Directory.Exists(_root))
            Directory.Delete(_root, recursive: true);
    }

    private string Variant(string relative)
    {
        var dir = Path.Combine(_root, relative);
        Directory.CreateDirectory(dir);
        File.WriteAllText(Path.Combine(dir, "genai_config.json"), "{}");
        File.WriteAllText(Path.Combine(dir, "model.onnx"), "dummy");
        return dir;
    }

    [Fact]
    public void A_repository_folder_resolves_to_its_two_level_variant()
    {
        var nested = Variant(Path.Combine("cpu_and_mobile", "cpu-int4-rtn-block-32-acc-level-4"));

        OnnxVariantLocator.Find(_root, ExecutionProvider.Cpu).Should().Be(nested);
    }

    [Fact]
    public void Cuda_prefers_the_gpu_variant_and_cpu_the_cpu_variant()
    {
        var gpu = Variant(Path.Combine("gpu", "gpu-int4-awq-block-128"));
        var cpu = Variant(Path.Combine("cpu_and_mobile", "cpu-int4-rtn-block-32-acc-level-4"));

        OnnxVariantLocator.Find(_root, ExecutionProvider.Cuda).Should().Be(gpu);
        OnnxVariantLocator.Find(_root, ExecutionProvider.Cpu).Should().Be(cpu);
    }

    [Fact]
    public void A_folder_without_any_model_resolves_to_nothing()
    {
        Directory.CreateDirectory(Path.Combine(_root, "cpu_and_mobile", "empty"));

        OnnxVariantLocator.Find(_root, ExecutionProvider.Cpu).Should().BeNull();
    }
}
