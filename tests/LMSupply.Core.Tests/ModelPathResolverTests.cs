using AwesomeAssertions;
using LMSupply.Core.Download;

namespace LMSupply.Core.Tests;

public class StripVariantSuffixTests
{
    [Theory]
    [InlineData("xnorpx/rt-detr2-onnx:s", "xnorpx/rt-detr2-onnx")]
    [InlineData("xnorpx/rt-detr2-onnx:ms", "xnorpx/rt-detr2-onnx")]
    [InlineData("owner/repo:variant", "owner/repo")]
    public void HuggingFaceRepoWithVariant_ShouldStripVariant(string input, string expected)
    {
        ModelPathResolver.StripVariantSuffix(input).Should().Be(expected);
    }

    [Theory]
    [InlineData("xnorpx/rt-detr2-onnx")]
    [InlineData("sentence-transformers/all-MiniLM-L6-v2")]
    [InlineData("owner/repo")]
    public void HuggingFaceRepoWithoutVariant_ShouldReturnUnchanged(string input)
    {
        ModelPathResolver.StripVariantSuffix(input).Should().Be(input);
    }

    [Theory]
    [InlineData("D:\\data\\models\\yolov8n-pose.onnx")]
    [InlineData("D:/data/models/yolov8n-pose.onnx")]
    [InlineData("C:\\Users\\test\\model.onnx")]
    public void WindowsLocalPath_ShouldReturnUnchanged(string input)
    {
        ModelPathResolver.StripVariantSuffix(input).Should().Be(input);
    }

    [Theory]
    [InlineData("default")]
    [InlineData("fast")]
    [InlineData("quality")]
    public void PlainAlias_ShouldReturnUnchanged(string input)
    {
        ModelPathResolver.StripVariantSuffix(input).Should().Be(input);
    }

    [Theory]
    [InlineData("default:fp16")]
    [InlineData("large:q4")]
    public void AliasWithQualifier_NoSlash_ShouldReturnUnchanged(string input)
    {
        // Aliases with qualifiers don't contain '/', so StripVariantSuffix should not touch them.
        // Qualifier splitting is handled by LMSupplyOptionsBase.SplitQualifier.
        ModelPathResolver.StripVariantSuffix(input).Should().Be(input);
    }

    [Fact]
    public void EmptyString_ShouldReturnUnchanged()
    {
        ModelPathResolver.StripVariantSuffix("").Should().Be("");
    }

    [Fact]
    public void ColonAtEnd_ShouldReturnUnchanged()
    {
        ModelPathResolver.StripVariantSuffix("owner/repo:").Should().Be("owner/repo:");
    }

    [Fact]
    public void ColonBeforeSlash_ShouldReturnUnchanged()
    {
        // Edge case: colon appears before the slash (not a variant)
        ModelPathResolver.StripVariantSuffix("proto:owner/repo").Should().Be("proto:owner/repo");
    }
}
