using FluentAssertions;

namespace LMSupply.Segmenter.Tests;

public class Ade20kLabelsTests
{
    [Fact]
    public void Labels_ShouldHave150Classes()
    {
        Ade20kLabels.Labels.Should().HaveCount(150);
    }

    [Theory]
    [InlineData(0, "wall")]
    [InlineData(2, "sky")]
    [InlineData(12, "person")]
    [InlineData(20, "car")]
    [InlineData(149, "flag")]
    public void GetLabel_ValidClassId_ShouldReturnCorrectLabel(int classId, string expectedLabel)
    {
        Ade20kLabels.GetLabel(classId).Should().Be(expectedLabel);
    }

    [Theory]
    [InlineData(-1)]
    [InlineData(150)]
    [InlineData(200)]
    public void GetLabel_InvalidClassId_ShouldReturnUnknown(int classId)
    {
        Ade20kLabels.GetLabel(classId).Should().Be("unknown");
    }

    [Fact]
    public void Labels_ShouldContainCommonCategories()
    {
        Ade20kLabels.Labels.Should().Contain("wall");
        Ade20kLabels.Labels.Should().Contain("sky");
        Ade20kLabels.Labels.Should().Contain("floor");
        Ade20kLabels.Labels.Should().Contain("person");
        Ade20kLabels.Labels.Should().Contain("car");
        Ade20kLabels.Labels.Should().Contain("tree");
    }

    [Fact]
    public void Labels_FirstElement_ShouldBeWall()
    {
        Ade20kLabels.Labels[0].Should().Be("wall");
    }
}
