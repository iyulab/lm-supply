using AwesomeAssertions;
using LMSupply.Detector;
using LMSupply.Detector.Models;
using Xunit;

namespace LMSupply.Detector.Tests;

/// <summary>
/// Which vocabulary a detection is labelled from.
///
/// <para>
/// Post-processing called <c>CocoLabels.GetLabel(classId)</c> in seven places, so every model — whatever it
/// was actually trained on — reported COCO-80 names. A single-class detector's class 0 came back as
/// <c>"person"</c>, which is not merely wrong but wrong in a way nothing downstream can see: the string is a
/// plausible COCO label, so a consumer filtering on it gets confident nonsense rather than an error. The
/// redaction use case a consumer asked about (faces, plates) is unreachable while this holds, because those
/// models have their own vocabularies.
/// </para>
/// <para>
/// The label list is now the single source of truth and <c>NumClasses</c> derives from it, so the two cannot
/// disagree — the previous shape let a model declare <c>NumClasses = 1</c> and still be labelled from eighty.
/// </para>
/// </summary>
public class DetectorLabelSetTests
{
    private static DetectorModelInfo Info(IReadOnlyList<string>? labels = null) => new()
    {
        Id = "test/model",
        AliasName = "test",
        DisplayName = "Test",
        ClassLabels = labels ?? CocoLabels.Labels,
    };

    [Fact]
    public void DefaultLabelSet_IsStillCoco80()
    {
        var info = Info();

        info.NumClasses.Should().Be(80, "every model shipped so far is COCO-trained; the default must not move");
        info.LabelFor(0).Should().Be("person");
        info.LabelFor(79).Should().Be("toothbrush");
    }

    [Fact]
    public void AModelWithItsOwnVocabulary_IsLabelledFromIt()
    {
        var info = Info(["face"]);

        info.LabelFor(0).Should().Be("face",
            "a face detector's class 0 is a face; returning \"person\" is the failure this change exists to stop");
        info.NumClasses.Should().Be(1);
    }

    [Fact]
    public void NumClasses_AlwaysAgreesWithTheLabelList()
    {
        Info(["face"]).NumClasses.Should().Be(1);
        Info(["plate", "face"]).NumClasses.Should().Be(2);
        Info().NumClasses.Should().Be(CocoLabels.Labels.Count,
            "the count is derived, not declared — the old shape allowed NumClasses = 1 beside eighty labels");
    }

    [Fact]
    public void APoseModelResolvedFromHuggingFace_IsLabelledWithItsOneClass()
    {
        // The registry already knew a pose model has one class (it set NumClasses = 1) while still handing
        // post-processing eighty names -- the exact disagreement the derived count now makes impossible.
        var pose = DetectorModelRegistry.Default.Resolve("someone/yolov8n-pose");

        pose.NumClasses.Should().Be(1);
        pose.LabelFor(0).Should().Be("person");
        pose.LabelFor(1).Should().Be("unknown", "there is no second class to name");
    }

    [Fact]
    public void ANonPoseModelResolvedFromHuggingFace_KeepsCoco()
    {
        var detector = DetectorModelRegistry.Default.Resolve("someone/rtdetr-l");

        detector.NumClasses.Should().Be(CocoLabels.Labels.Count);
        detector.LabelFor(0).Should().Be("person");
    }

    [Fact]
    public void AClassIdOutsideTheVocabulary_IsNotGivenAPlausibleName()
    {
        var info = Info(["face"]);

        info.LabelFor(5).Should().Be("unknown",
            "an out-of-range id must read as unknown; borrowing a COCO name for it is how a mislabelled " +
            "detection passes downstream filters unnoticed");
        info.LabelFor(-1).Should().Be("unknown");
    }
}
