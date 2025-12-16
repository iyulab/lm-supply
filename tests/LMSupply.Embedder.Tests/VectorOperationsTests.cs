using FluentAssertions;
using LMSupply.Embedder.Utils;

namespace LMSupply.Embedder.Tests;

public class VectorOperationsTests
{
    [Fact]
    public void CosineSimilarity_IdenticalVectors_ReturnsOne()
    {
        var vec = new float[] { 1, 2, 3, 4, 5 };
        var result = VectorOperations.CosineSimilarity(vec, vec);
        result.Should().BeApproximately(1.0f, 0.00001f);
    }

    [Fact]
    public void CosineSimilarity_OrthogonalVectors_ReturnsZero()
    {
        var vec1 = new float[] { 1, 0, 0 };
        var vec2 = new float[] { 0, 1, 0 };
        var result = VectorOperations.CosineSimilarity(vec1, vec2);
        result.Should().BeApproximately(0.0f, 0.00001f);
    }

    [Fact]
    public void CosineSimilarity_OppositeVectors_ReturnsNegativeOne()
    {
        var vec1 = new float[] { 1, 2, 3 };
        var vec2 = new float[] { -1, -2, -3 };
        var result = VectorOperations.CosineSimilarity(vec1, vec2);
        result.Should().BeApproximately(-1.0f, 0.00001f);
    }

    [Fact]
    public void EuclideanDistance_IdenticalVectors_ReturnsZero()
    {
        var vec = new float[] { 1, 2, 3 };
        var result = VectorOperations.EuclideanDistance(vec, vec);
        result.Should().BeApproximately(0.0f, 0.00001f);
    }

    [Fact]
    public void EuclideanDistance_KnownVectors_ReturnsCorrectValue()
    {
        var vec1 = new float[] { 0, 0, 0 };
        var vec2 = new float[] { 3, 4, 0 };
        var result = VectorOperations.EuclideanDistance(vec1, vec2);
        result.Should().BeApproximately(5.0f, 0.00001f);
    }

    [Fact]
    public void DotProduct_KnownVectors_ReturnsCorrectValue()
    {
        var vec1 = new float[] { 1, 2, 3 };
        var vec2 = new float[] { 4, 5, 6 };
        var result = VectorOperations.DotProduct(vec1, vec2);
        result.Should().BeApproximately(32.0f, 0.00001f); // 1*4 + 2*5 + 3*6
    }

    [Fact]
    public void NormalizeL2_NormalizesVectorToUnitLength()
    {
        var vec = new float[] { 3, 4 };
        VectorOperations.NormalizeL2(vec);

        var norm = VectorOperations.Norm(vec);
        norm.Should().BeApproximately(1.0f, 0.00001f);
        vec[0].Should().BeApproximately(0.6f, 0.00001f);
        vec[1].Should().BeApproximately(0.8f, 0.00001f);
    }

    [Fact]
    public void NormalizeL2_ZeroVector_RemainsZero()
    {
        var vec = new float[] { 0, 0, 0 };
        VectorOperations.NormalizeL2(vec);

        vec.Should().AllBeEquivalentTo(0.0f);
    }

    [Fact]
    public void Norm_KnownVector_ReturnsCorrectValue()
    {
        var vec = new float[] { 3, 4 };
        var result = VectorOperations.Norm(vec);
        result.Should().BeApproximately(5.0f, 0.00001f);
    }

    [Fact]
    public void Add_AddsVectorsInPlace()
    {
        var dest = new float[] { 1, 2, 3 };
        var source = new float[] { 4, 5, 6 };
        VectorOperations.Add(dest, source);

        dest[0].Should().Be(5.0f);
        dest[1].Should().Be(7.0f);
        dest[2].Should().Be(9.0f);
    }

    [Fact]
    public void Divide_DividesVectorByScalar()
    {
        var vec = new float[] { 10, 20, 30 };
        VectorOperations.Divide(vec, 10);

        vec[0].Should().Be(1.0f);
        vec[1].Should().Be(2.0f);
        vec[2].Should().Be(3.0f);
    }

    [Fact]
    public void CosineSimilarity_LargeVectors_WorksCorrectly()
    {
        var vec1 = Enumerable.Range(0, 384).Select(i => (float)i).ToArray();
        var vec2 = Enumerable.Range(0, 384).Select(i => (float)i * 2).ToArray();

        var result = VectorOperations.CosineSimilarity(vec1, vec2);
        result.Should().BeInRange(0.99f, 1.0f);
    }

    [Fact]
    public void EuclideanDistance_SingleDimension()
    {
        var vec1 = new float[] { 0 };
        var vec2 = new float[] { 5 };
        var result = VectorOperations.EuclideanDistance(vec1, vec2);
        result.Should().BeApproximately(5.0f, 0.00001f);
    }

    [Fact]
    public void DotProduct_ZeroVectors_ReturnsZero()
    {
        var vec1 = new float[] { 0, 0, 0 };
        var vec2 = new float[] { 1, 2, 3 };
        var result = VectorOperations.DotProduct(vec1, vec2);
        result.Should().BeApproximately(0.0f, 0.00001f);
    }

    [Fact]
    public void Norm_ZeroVector_ReturnsZero()
    {
        var vec = new float[] { 0, 0, 0 };
        var result = VectorOperations.Norm(vec);
        result.Should().BeApproximately(0.0f, 0.00001f);
    }

    [Fact]
    public void Norm_UnitVector_ReturnsOne()
    {
        var vec = new float[] { 1, 0, 0 };
        var result = VectorOperations.Norm(vec);
        result.Should().BeApproximately(1.0f, 0.00001f);
    }

    [Fact]
    public void Add_WithNegativeValues()
    {
        var dest = new float[] { -1, -2, -3 };
        var source = new float[] { 1, 2, 3 };
        VectorOperations.Add(dest, source);

        dest[0].Should().Be(0.0f);
        dest[1].Should().Be(0.0f);
        dest[2].Should().Be(0.0f);
    }

    [Fact]
    public void NormalizeL2_AlreadyNormalized()
    {
        var vec = new float[] { 0.6f, 0.8f };
        VectorOperations.NormalizeL2(vec);

        var norm = VectorOperations.Norm(vec);
        norm.Should().BeApproximately(1.0f, 0.0001f);
    }

    [Fact]
    public void Divide_ByFractionalScalar()
    {
        var vec = new float[] { 1, 2, 3 };
        VectorOperations.Divide(vec, 0.5f);

        vec[0].Should().BeApproximately(2.0f, 0.00001f);
        vec[1].Should().BeApproximately(4.0f, 0.00001f);
        vec[2].Should().BeApproximately(6.0f, 0.00001f);
    }

    [Fact]
    public void CosineSimilarity_ThrowsOnEmptyVector()
    {
        var empty = Array.Empty<float>();
        var vec = new float[] { 1, 2, 3 };

        var act1 = () => VectorOperations.CosineSimilarity(empty, vec);
        var act2 = () => VectorOperations.CosineSimilarity(vec, empty);

        act1.Should().Throw<ArgumentException>();
        act2.Should().Throw<ArgumentException>();
    }

    [Fact]
    public void CosineSimilarity_ThrowsOnLengthMismatch()
    {
        var vec1 = new float[] { 1, 2, 3 };
        var vec2 = new float[] { 1, 2 };

        var act = () => VectorOperations.CosineSimilarity(vec1, vec2);
        act.Should().Throw<ArgumentException>();
    }

    [Fact]
    public void EuclideanDistance_ThrowsOnEmptyVector()
    {
        var empty = Array.Empty<float>();
        var vec = new float[] { 1, 2, 3 };

        var act = () => VectorOperations.EuclideanDistance(empty, vec);
        act.Should().Throw<ArgumentException>();
    }

    [Fact]
    public void DotProduct_ThrowsOnLengthMismatch()
    {
        var vec1 = new float[] { 1, 2, 3, 4 };
        var vec2 = new float[] { 1, 2 };

        var act = () => VectorOperations.DotProduct(vec1, vec2);
        act.Should().Throw<ArgumentException>()
            .WithMessage("*lengths must match*");
    }
}
