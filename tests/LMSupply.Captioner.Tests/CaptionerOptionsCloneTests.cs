using System.Reflection;
using AwesomeAssertions;

namespace LMSupply.Captioner.Tests;

/// <summary>
/// <see cref="CaptionerOptions.Clone"/> must carry every option — a hand-written copy is how an option gets dropped
/// when it is added later. The reflection fact below sets every public settable property to a non-default value and
/// requires the copy to carry it, whichever way <c>Clone</c> is written.
/// </summary>
public sealed class CaptionerOptionsCloneTests
{
    private static CancellationToken Ct => TestContext.Current.CancellationToken;

    [Fact]
    public void Clone_CarriesEverySettableProperty()
    {
        var original = new CaptionerOptions();
        var properties = typeof(CaptionerOptions)
            .GetProperties(BindingFlags.Public | BindingFlags.Instance)
            .Where(p => p.CanRead && p.CanWrite && p.SetMethod!.IsPublic)
            .ToList();

        foreach (var property in properties)
            property.SetValue(original, NonDefault(property.PropertyType, property.GetValue(original)));

        var copy = original.Clone();

        copy.Should().NotBeSameAs(original);
        foreach (var property in properties)
            property.GetValue(copy).Should().Be(property.GetValue(original), $"Clone must copy {property.Name}");
    }

    [Fact]
    public async Task Load_DoesNotWriteTheQualifierIntoTheCallersOptions()
    {
        // Refused before any download (Detail on ViT-GPT2), after the qualifier has been parsed into the options.
        var options = new CaptionerOptions { Detail = CaptionDetail.Detailed, DisableAutoDownload = true };

        var load = () => LocalCaptioner.LoadAsync("default:fp16", options, cancellationToken: Ct);

        await load.Should().ThrowAsync<NotSupportedException>();
        options.QuantizationHint.Should().BeNull("the load works on a copy of the caller's options");
    }

    private static object? NonDefault(Type type, object? current)
    {
        var underlying = Nullable.GetUnderlyingType(type) ?? type;
        if (underlying == typeof(string)) return "x-" + (current ?? "set");
        if (underlying == typeof(bool)) return !(current as bool? ?? false);
        if (underlying == typeof(int)) return (current as int? ?? 0) + 7;
        if (underlying == typeof(float)) return (current as float? ?? 0f) + 0.25f;
        if (underlying.IsEnum)
        {
            var values = Enum.GetValues(underlying);
            foreach (var value in values)
            {
                if (!Equals(value, current))
                    return value;
            }
        }

        throw new NotSupportedException(
            $"Add a non-default value for {type} to this test so the clone check covers the new property.");
    }
}
