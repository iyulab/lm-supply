namespace LMSupply.Text.Tests;

public class EncodedTypesTests
{
    #region EncodedSequence Tests

    [Fact]
    public void EncodedSequence_ShouldStoreValues()
    {
        // Arrange
        var inputIds = new long[] { 101, 7592, 2088, 102, 0 };
        var attentionMask = new long[] { 1, 1, 1, 1, 0 };

        // Act
        var encoded = new EncodedSequence(inputIds, attentionMask, 4);

        // Assert
        encoded.InputIds.Should().BeEquivalentTo(inputIds);
        encoded.AttentionMask.Should().BeEquivalentTo(attentionMask);
        encoded.Length.Should().Be(4);
    }

    [Fact]
    public void EncodedSequence_FromInts_ShouldConvertCorrectly()
    {
        // Arrange
        var inputIds = new int[] { 101, 7592, 2088, 102 };
        var attentionMask = new int[] { 1, 1, 1, 1 };

        // Act
        var encoded = EncodedSequence.FromInts(inputIds, attentionMask, 4);

        // Assert
        encoded.InputIds.Should().BeEquivalentTo(new long[] { 101, 7592, 2088, 102 });
        encoded.AttentionMask.Should().BeEquivalentTo(new long[] { 1, 1, 1, 1 });
        encoded.Length.Should().Be(4);
    }

    #endregion

    #region EncodedPair Tests

    [Fact]
    public void EncodedPair_ShouldStoreValues()
    {
        // Arrange
        var inputIds = new long[] { 101, 7592, 102, 2088, 102, 0 };
        var attentionMask = new long[] { 1, 1, 1, 1, 1, 0 };
        var tokenTypeIds = new long[] { 0, 0, 0, 1, 1, 0 };

        // Act
        var encoded = new EncodedPair(inputIds, attentionMask, tokenTypeIds, 5);

        // Assert
        encoded.InputIds.Should().BeEquivalentTo(inputIds);
        encoded.AttentionMask.Should().BeEquivalentTo(attentionMask);
        encoded.TokenTypeIds.Should().BeEquivalentTo(tokenTypeIds);
        encoded.Length.Should().Be(5);
    }

    #endregion

    #region EncodedBatch Tests

    [Fact]
    public void EncodedBatch_ShouldInitializeCorrectly()
    {
        // Act
        var batch = new EncodedBatch(batchSize: 2, sequenceLength: 8);

        // Assert
        batch.BatchSize.Should().Be(2);
        batch.SequenceLength.Should().Be(8);
        batch.InputIds.GetLength(0).Should().Be(2);
        batch.InputIds.GetLength(1).Should().Be(8);
        batch.AttentionMask.GetLength(0).Should().Be(2);
        batch.AttentionMask.GetLength(1).Should().Be(8);
    }

    [Fact]
    public void EncodedBatch_SetSequence_ShouldPopulateCorrectly()
    {
        // Arrange
        var batch = new EncodedBatch(batchSize: 2, sequenceLength: 4);
        var seq1 = new EncodedSequence(new long[] { 101, 1, 102, 0 }, new long[] { 1, 1, 1, 0 }, 3);
        var seq2 = new EncodedSequence(new long[] { 101, 2, 3, 102 }, new long[] { 1, 1, 1, 1 }, 4);

        // Act
        batch.SetSequence(0, seq1);
        batch.SetSequence(1, seq2);

        // Assert
        batch.InputIds[0, 0].Should().Be(101);
        batch.InputIds[0, 1].Should().Be(1);
        batch.InputIds[1, 0].Should().Be(101);
        batch.InputIds[1, 3].Should().Be(102);
        batch.AttentionMask[0, 3].Should().Be(0);
        batch.AttentionMask[1, 3].Should().Be(1);
    }

    [Fact]
    public void EncodedBatch_GetFlatInputIds_ShouldFlattenCorrectly()
    {
        // Arrange
        var batch = new EncodedBatch(batchSize: 2, sequenceLength: 3);
        batch.SetSequence(0, new EncodedSequence(new long[] { 1, 2, 3 }, new long[] { 1, 1, 1 }, 3));
        batch.SetSequence(1, new EncodedSequence(new long[] { 4, 5, 6 }, new long[] { 1, 1, 1 }, 3));

        // Act
        var flat = batch.GetFlatInputIds();

        // Assert
        flat.Should().BeEquivalentTo(new long[] { 1, 2, 3, 4, 5, 6 });
    }

    [Fact]
    public void EncodedBatch_GetFlatAttentionMask_ShouldFlattenCorrectly()
    {
        // Arrange
        var batch = new EncodedBatch(batchSize: 2, sequenceLength: 3);
        batch.SetSequence(0, new EncodedSequence(new long[] { 1, 2, 0 }, new long[] { 1, 1, 0 }, 2));
        batch.SetSequence(1, new EncodedSequence(new long[] { 4, 5, 6 }, new long[] { 1, 1, 1 }, 3));

        // Act
        var flat = batch.GetFlatAttentionMask();

        // Assert
        flat.Should().BeEquivalentTo(new long[] { 1, 1, 0, 1, 1, 1 });
    }

    #endregion

    #region EncodedPairBatch Tests

    [Fact]
    public void EncodedPairBatch_ShouldInitializeCorrectly()
    {
        // Act
        var batch = new EncodedPairBatch(batchSize: 3, sequenceLength: 16);

        // Assert
        batch.BatchSize.Should().Be(3);
        batch.SequenceLength.Should().Be(16);
        batch.InputIds.GetLength(0).Should().Be(3);
        batch.TokenTypeIds.GetLength(0).Should().Be(3);
    }

    [Fact]
    public void EncodedPairBatch_SetPair_ShouldPopulateCorrectly()
    {
        // Arrange
        var batch = new EncodedPairBatch(batchSize: 1, sequenceLength: 6);
        var pair = new EncodedPair(
            new long[] { 101, 1, 102, 2, 102, 0 },
            new long[] { 1, 1, 1, 1, 1, 0 },
            new long[] { 0, 0, 0, 1, 1, 0 },
            5);

        // Act
        batch.SetPair(0, pair);

        // Assert
        batch.InputIds[0, 0].Should().Be(101);
        batch.InputIds[0, 2].Should().Be(102);
        batch.InputIds[0, 3].Should().Be(2);
        batch.TokenTypeIds[0, 0].Should().Be(0);
        batch.TokenTypeIds[0, 3].Should().Be(1);
        batch.AttentionMask[0, 5].Should().Be(0);
    }

    [Fact]
    public void EncodedPairBatch_GetFlatTokenTypeIds_ShouldFlattenCorrectly()
    {
        // Arrange
        var batch = new EncodedPairBatch(batchSize: 2, sequenceLength: 3);
        batch.SetPair(0, new EncodedPair(
            new long[] { 1, 2, 3 },
            new long[] { 1, 1, 1 },
            new long[] { 0, 0, 1 },
            3));
        batch.SetPair(1, new EncodedPair(
            new long[] { 4, 5, 6 },
            new long[] { 1, 1, 0 },
            new long[] { 0, 1, 1 },
            2));

        // Act
        var flat = batch.GetFlatTokenTypeIds();

        // Assert
        flat.Should().BeEquivalentTo(new long[] { 0, 0, 1, 0, 1, 1 });
    }

    #endregion
}
