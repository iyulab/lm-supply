using FluentAssertions;
using LMSupply.Vision;

namespace LMSupply.Vision.Core.Tests;

public class PreprocessProfileTests
{
    [Fact]
    public void ImageNet_ShouldHaveCorrectDimensions()
    {
        // Act
        var profile = PreprocessProfile.ImageNet;

        // Assert
        profile.Width.Should().Be(224);
        profile.Height.Should().Be(224);
        profile.ResizeMode.Should().Be(ResizeMode.CenterCrop);
        profile.ChannelFirst.Should().BeTrue();
    }

    [Fact]
    public void ImageNet_ShouldHaveCorrectNormalizationValues()
    {
        // Act
        var profile = PreprocessProfile.ImageNet;

        // Assert
        profile.Mean.Should().HaveCount(3);
        profile.Std.Should().HaveCount(3);

        // ImageNet mean: [0.485, 0.456, 0.406]
        profile.Mean[0].Should().BeApproximately(0.485f, 0.001f);
        profile.Mean[1].Should().BeApproximately(0.456f, 0.001f);
        profile.Mean[2].Should().BeApproximately(0.406f, 0.001f);

        // ImageNet std: [0.229, 0.224, 0.225]
        profile.Std[0].Should().BeApproximately(0.229f, 0.001f);
        profile.Std[1].Should().BeApproximately(0.224f, 0.001f);
        profile.Std[2].Should().BeApproximately(0.225f, 0.001f);
    }

    [Fact]
    public void Clip_ShouldHaveCorrectDimensions()
    {
        // Act
        var profile = PreprocessProfile.Clip;

        // Assert
        profile.Width.Should().Be(224);
        profile.Height.Should().Be(224);
        profile.ResizeMode.Should().Be(ResizeMode.CenterCrop);
    }

    [Fact]
    public void Clip_ShouldHaveDistinctNormalizationFromImageNet()
    {
        // Act
        var clip = PreprocessProfile.Clip;
        var imageNet = PreprocessProfile.ImageNet;

        // Assert
        clip.Mean.Should().NotBeEquivalentTo(imageNet.Mean);
        clip.Std.Should().NotBeEquivalentTo(imageNet.Std);
    }

    [Fact]
    public void ViTGpt2_ShouldUseImageNetNormalization()
    {
        // Act
        var vitGpt2 = PreprocessProfile.ViTGpt2;
        var imageNet = PreprocessProfile.ImageNet;

        // Assert
        vitGpt2.Width.Should().Be(imageNet.Width);
        vitGpt2.Height.Should().Be(imageNet.Height);
        vitGpt2.Mean.Should().BeEquivalentTo(imageNet.Mean);
        vitGpt2.Std.Should().BeEquivalentTo(imageNet.Std);
    }

    [Fact]
    public void Florence2_ShouldHaveLargerDimensions()
    {
        // Act
        var profile = PreprocessProfile.Florence2;

        // Assert
        profile.Width.Should().Be(768);
        profile.Height.Should().Be(768);
        profile.ResizeMode.Should().Be(ResizeMode.Stretch);
    }

    [Fact]
    public void SmolVLM_ShouldUseSimpleNormalization()
    {
        // Act
        var profile = PreprocessProfile.SmolVLM;

        // Assert
        profile.Width.Should().Be(384);
        profile.Height.Should().Be(384);

        // SmolVLM uses 0.5 for all channels
        profile.Mean.Should().AllSatisfy(v => v.Should().Be(0.5f));
        profile.Std.Should().AllSatisfy(v => v.Should().Be(0.5f));
    }

    [Fact]
    public void CustomProfile_ShouldPreserveAllProperties()
    {
        // Arrange
        var mean = new float[] { 0.1f, 0.2f, 0.3f };
        var std = new float[] { 0.4f, 0.5f, 0.6f };

        // Act
        var profile = new PreprocessProfile(
            Width: 512,
            Height: 256,
            Mean: mean,
            Std: std,
            ResizeMode: ResizeMode.Fit,
            ChannelFirst: false);

        // Assert
        profile.Width.Should().Be(512);
        profile.Height.Should().Be(256);
        profile.Mean.Should().BeEquivalentTo(mean);
        profile.Std.Should().BeEquivalentTo(std);
        profile.ResizeMode.Should().Be(ResizeMode.Fit);
        profile.ChannelFirst.Should().BeFalse();
    }

    [Theory]
    [InlineData(ResizeMode.Stretch)]
    [InlineData(ResizeMode.Fit)]
    [InlineData(ResizeMode.CenterCrop)]
    [InlineData(ResizeMode.ShortEdgeCrop)]
    public void ResizeMode_ShouldSupportAllValues(ResizeMode mode)
    {
        // Act
        var profile = new PreprocessProfile(
            Width: 224,
            Height: 224,
            Mean: [0.5f, 0.5f, 0.5f],
            Std: [0.5f, 0.5f, 0.5f],
            ResizeMode: mode);

        // Assert
        profile.ResizeMode.Should().Be(mode);
    }
}
