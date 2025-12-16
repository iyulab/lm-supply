using FluentAssertions;

namespace LMSupply.Detector.Tests;

public class CocoLabelsTests
{
    [Fact]
    public void Labels_ShouldHave80Classes()
    {
        CocoLabels.Labels.Should().HaveCount(80);
    }

    [Theory]
    [InlineData(0, "person")]
    [InlineData(2, "car")]
    [InlineData(15, "cat")]
    [InlineData(16, "dog")]
    [InlineData(79, "toothbrush")]
    public void GetLabel_ValidClassId_ShouldReturnCorrectLabel(int classId, string expectedLabel)
    {
        CocoLabels.GetLabel(classId).Should().Be(expectedLabel);
    }

    [Theory]
    [InlineData(-1)]
    [InlineData(80)]
    [InlineData(100)]
    public void GetLabel_InvalidClassId_ShouldReturnUnknown(int classId)
    {
        CocoLabels.GetLabel(classId).Should().Be("unknown");
    }

    [Fact]
    public void Labels_ShouldContainCommonObjects()
    {
        CocoLabels.Labels.Should().Contain("person");
        CocoLabels.Labels.Should().Contain("car");
        CocoLabels.Labels.Should().Contain("dog");
        CocoLabels.Labels.Should().Contain("cat");
        CocoLabels.Labels.Should().Contain("chair");
        CocoLabels.Labels.Should().Contain("laptop");
    }

    [Fact]
    public void Labels_FirstElement_ShouldBePerson()
    {
        CocoLabels.Labels[0].Should().Be("person");
    }
}
