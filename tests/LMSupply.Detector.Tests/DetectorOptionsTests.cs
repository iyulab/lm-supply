using FluentAssertions;

namespace LMSupply.Detector.Tests;

public class DetectorOptionsTests
{
    [Fact]
    public void DefaultOptions_ShouldHaveCorrectDefaults()
    {
        var options = new DetectorOptions();

        options.ModelId.Should().Be("default");
        options.ConfidenceThreshold.Should().Be(0.25f);
        options.IouThreshold.Should().Be(0.45f);
        options.MaxDetections.Should().Be(100);
        options.Provider.Should().Be(ExecutionProvider.Auto);
    }

    [Fact]
    public void Clone_ShouldCreateIndependentCopy()
    {
        var original = new DetectorOptions
        {
            ModelId = "custom",
            ConfidenceThreshold = 0.5f,
            IouThreshold = 0.6f,
            MaxDetections = 50,
            ClassFilter = new HashSet<int> { 0, 1, 2 }
        };

        var cloned = original.Clone();

        cloned.Should().NotBeSameAs(original);
        cloned.ModelId.Should().Be("custom");
        cloned.ConfidenceThreshold.Should().Be(0.5f);
        cloned.IouThreshold.Should().Be(0.6f);
        cloned.MaxDetections.Should().Be(50);
        cloned.ClassFilter.Should().BeEquivalentTo(new[] { 0, 1, 2 });
    }

    [Fact]
    public void Clone_WithClassFilter_ShouldNotShareReference()
    {
        var original = new DetectorOptions { ClassFilter = new HashSet<int> { 0, 1 } };
        var cloned = original.Clone();

        // ClassFilter is IReadOnlySet, so we cast to HashSet to test modification
        ((HashSet<int>)cloned.ClassFilter!).Add(2);

        original.ClassFilter.Should().HaveCount(2);
        cloned.ClassFilter.Should().HaveCount(3);
    }

    [Fact]
    public void ClassFilter_WhenNull_ShouldAllowAllClasses()
    {
        var options = new DetectorOptions();

        options.ClassFilter.Should().BeNull();
    }
}
