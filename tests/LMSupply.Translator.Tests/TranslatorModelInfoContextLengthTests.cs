using AwesomeAssertions;
using LMSupply;
using LMSupply.Translator.Models;

namespace LMSupply.Translator.Tests;

/// <summary>
/// <see cref="IModelInfoBase.ContextLength"/> is what model listings report generically; a translator knows its
/// maximum sequence length, so the listing must not say "unknown".
/// </summary>
public class TranslatorModelInfoContextLengthTests
{
    [Fact]
    public void ContextLength_IsTheMaxLength_ForEveryCatalogModel()
    {
        var models = TranslatorModelRegistry.Default.GetAvailableModels();

        models.Should().NotBeEmpty();
        foreach (var model in models)
        {
            ((IModelInfoBase)model).ContextLength.Should().Be(model.MaxLength, model.Id);
        }
    }
}
