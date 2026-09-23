using AwesomeAssertions;
using LMSupply;
using LMSupply.Reranker.Models;

namespace LMSupply.Reranker.Tests;

/// <summary>
/// <see cref="IModelInfoBase.ContextLength"/> is what model listings report generically; a reranker knows its
/// input limit, so the listing must not say "unknown".
/// </summary>
public class ModelInfoContextLengthTests
{
    public static TheoryData<string> CatalogModelIds()
    {
        var data = new TheoryData<string>();
        foreach (var model in RerankerModelRegistry.Default.GetAvailableModels())
        {
            data.Add(model.Id);
        }
        return data;
    }

    [Theory]
    [MemberData(nameof(CatalogModelIds))]
    public void ContextLength_IsTheMaxSequenceLength(string modelId)
    {
        var model = RerankerModelRegistry.Default.GetAvailableModels().Single(m => m.Id == modelId);

        ((IModelInfoBase)model).ContextLength.Should().Be(model.MaxSequenceLength);
    }
}
