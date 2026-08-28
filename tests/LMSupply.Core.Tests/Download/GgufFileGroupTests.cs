using AwesomeAssertions;
using LMSupply.Core.Download;
using Xunit;

namespace LMSupply.Core.Tests.Download;

public class GgufFileGroupTests
{
    // --- 분할 파일 감지 ---

    [Theory]
    [InlineData("model-Q4_K_M-00001-of-00003.gguf", true)]
    [InlineData("model-Q4_K_M-00003-of-00003.gguf", true)]
    [InlineData("model-Q4_K_M-00001-of-00010.gguf", true)]
    [InlineData("model-Q4_K_M.gguf", false)]
    [InlineData("Llama-3.2-3B-Instruct-Q4_K_M.gguf", false)]
    [InlineData("Qwen2.5-14B-Q8_0.gguf", false)]
    public void IsSplitPart_DetectsCorrectly(string filename, bool expected)
    {
        GgufFileGroup.IsSplitPart(filename).Should().Be(expected);
    }

    // --- 기본 이름 추출 ---

    [Theory]
    [InlineData("model-Q4_K_M-00001-of-00003.gguf", "model-Q4_K_M")]
    [InlineData("Qwen2.5-14B-Q4_K_M-00002-of-00003.gguf", "Qwen2.5-14B-Q4_K_M")]
    [InlineData("model-Q4_K_M.gguf", "model-Q4_K_M")]
    [InlineData("Llama-3.2-3B-Instruct-Q4_K_M.gguf", "Llama-3.2-3B-Instruct-Q4_K_M")]
    public void GetBaseName_ExtractsCorrectly(string filename, string expected)
    {
        GgufFileGroup.GetBaseName(filename).Should().Be(expected);
    }

    // --- 그룹핑: 단일 파일 ---

    [Fact]
    public void GroupFiles_SingleFile_OneGroup()
    {
        var files = new[]
        {
            new GgufRawFile("Llama-3.2-3B-Instruct-Q4_K_M.gguf", 2_100_000_000L),
        };

        var groups = GgufFileGroup.GroupFiles(files).ToList();

        groups.Should().HaveCount(1);
        groups[0].TotalSizeBytes.Should().Be(2_100_000_000L);
        groups[0].PrimaryFileName.Should().Be("Llama-3.2-3B-Instruct-Q4_K_M.gguf");
        groups[0].IsSplit.Should().BeFalse();
        groups[0].Parts.Should().HaveCount(1);
    }

    // --- 그룹핑: 분할 파일 합산 ---

    [Fact]
    public void GroupFiles_SplitFiles_OneGroupWithTotalSize()
    {
        var files = new[]
        {
            new GgufRawFile("model-Q4_K_M-00001-of-00003.gguf", 2_100_000_000L),
            new GgufRawFile("model-Q4_K_M-00002-of-00003.gguf", 2_100_000_000L),
            new GgufRawFile("model-Q4_K_M-00003-of-00003.gguf", 2_100_000_000L),
        };

        var groups = GgufFileGroup.GroupFiles(files).ToList();

        groups.Should().HaveCount(1);
        groups[0].TotalSizeBytes.Should().Be(6_300_000_000L);
        groups[0].IsSplit.Should().BeTrue();
        groups[0].Parts.Should().HaveCount(3);
        groups[0].PrimaryFileName.Should().Be("model-Q4_K_M-00001-of-00003.gguf");
    }

    // --- 그룹핑: 혼합 (분할 + 단일) ---

    [Fact]
    public void GroupFiles_MixedFiles_GroupsCorrectly()
    {
        var files = new[]
        {
            new GgufRawFile("model-Q4_K_M-00001-of-00002.gguf", 3_000_000_000L),
            new GgufRawFile("model-Q4_K_M-00002-of-00002.gguf", 3_000_000_000L),
            new GgufRawFile("model-Q8_0.gguf", 8_000_000_000L),
        };

        var groups = GgufFileGroup.GroupFiles(files).ToList();

        groups.Should().HaveCount(2);

        var splitGroup = groups.First(g => g.IsSplit);
        splitGroup.TotalSizeBytes.Should().Be(6_000_000_000L);
        splitGroup.Parts.Should().HaveCount(2);

        var singleGroup = groups.First(g => !g.IsSplit);
        singleGroup.TotalSizeBytes.Should().Be(8_000_000_000L);
        singleGroup.PrimaryFileName.Should().Be("model-Q8_0.gguf");
    }

    // --- 그룹핑: 여러 단일 파일 ---

    [Fact]
    public void GroupFiles_MultipleQuantizations_EachIsOwnGroup()
    {
        var files = new[]
        {
            new GgufRawFile("model-Q4_K_M.gguf", 3_000_000_000L),
            new GgufRawFile("model-Q6_K.gguf", 5_000_000_000L),
            new GgufRawFile("model-Q8_0.gguf", 8_000_000_000L),
        };

        var groups = GgufFileGroup.GroupFiles(files).ToList();

        groups.Should().HaveCount(3);
        groups.Should().AllSatisfy(g => g.IsSplit.Should().BeFalse());
    }

    // --- TotalSizeGB ---

    [Fact]
    public void TotalSizeGB_CalculatesCorrectly()
    {
        var group = new GgufFileGroup
        {
            PrimaryFileName = "model.gguf",
            Parts = ["model.gguf"],
            TotalSizeBytes = 4L * 1024 * 1024 * 1024
        };

        group.TotalSizeGB.Should().BeApproximately(4.0, 0.01);
    }
}
