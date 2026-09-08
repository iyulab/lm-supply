using System.Reflection;
using AwesomeAssertions;
using LMSupply.Generator.Internal.Llama;
using LMSupply.Llama.Server;

namespace LMSupply.Generator.Tests;

/// <summary>
/// <see cref="LlamaOptions"/> and <see cref="LlamaServerConfig"/> are copied by hand at three
/// points in the load path (VRAM auto-tune, CPU fallback, OOM retry). A property added to either type
/// and forgotten at a copy site is dropped silently for exactly the loads that go through that path —
/// the same failure class as an override that copies every field but one. These tests set every
/// settable property to a non-default value, push it through each copy, and require it to survive,
/// so the compiler-free "did you add it to the clone?" question has an answer that fails.
/// </summary>
public class LlamaOptionsCloneCompletenessTests
{
    [Fact]
    public void CloneLlamaOptionsWithGpuLayers_CarriesEveryPropertyExceptTheOffloadPair()
    {
        var source = Populate(new LlamaOptions());

        var clone = LlamaServerGeneratorModel.CloneLlamaOptionsWithGpuLayers(source, gpuLayers: 3);

        clone.GpuLayerCount.Should().Be(3);
        clone.GpuOffloadRatio.Should().BeNull("a ratio would override the layer count the clone was made to set");
        AssertSameExcept(source, clone, nameof(LlamaOptions.GpuLayerCount), nameof(LlamaOptions.GpuOffloadRatio));
    }

    [Fact]
    public void CloneLlamaOptionsForCpuFallback_CarriesEveryPropertyExceptTheOffloadPair()
    {
        var source = Populate(new LlamaOptions());

        var clone = LlamaServerGeneratorModel.CloneLlamaOptionsForCpuFallback(source);

        clone.GpuLayerCount.Should().Be(0);
        clone.GpuOffloadRatio.Should().BeNull();
        AssertSameExcept(source, clone, nameof(LlamaOptions.GpuLayerCount), nameof(LlamaOptions.GpuOffloadRatio));
    }

    [Fact]
    public void CloneConfigWithGpuLayers_CarriesEveryPropertyExceptGpuLayers()
    {
        // `init`-only properties cannot be set by reflection after construction, so the config is
        // built through a constructor-time initializer and then overwritten field by field via the
        // backing setters reflection can still reach.
        var source = Populate(new LlamaServerConfig { ModelPath = "seed" });

        var clone = LlamaServerGeneratorModel.CloneConfigWithGpuLayers(source, gpuLayers: 5);

        clone.GpuLayers.Should().Be(5);
        AssertSameExcept(source, clone, nameof(LlamaServerConfig.GpuLayers));
    }

    // ── helpers ─────────────────────────────────────────────────────────────────────────────

    private static readonly BindingFlags Settable = BindingFlags.Public | BindingFlags.Instance;

    private static T Populate<T>(T target) where T : class
    {
        foreach (var prop in typeof(T).GetProperties(Settable).Where(p => p.SetMethod != null))
        {
            prop.SetValue(target, NonDefaultValue(prop.PropertyType, prop.GetValue(target)));
        }

        return target;
    }

    private static void AssertSameExcept<T>(T source, T clone, params string[] except)
    {
        foreach (var prop in typeof(T).GetProperties(Settable).Where(p => p.SetMethod != null && !except.Contains(p.Name)))
        {
            var expected = prop.GetValue(source);
            var actual = prop.GetValue(clone);
            actual.Should().BeEquivalentTo(expected,
                $"{typeof(T).Name}.{prop.Name} must survive the clone — add it to the copy site");
            actual.Should().NotBe(Default(prop.PropertyType),
                $"the test must give {typeof(T).Name}.{prop.Name} a non-default value or the assertion above is vacuous");
        }
    }

    private static object? Default(Type t) => t.IsValueType ? Activator.CreateInstance(t) : null;

    /// <summary>A value that differs from the type's default and from the current one.</summary>
    private static object NonDefaultValue(Type type, object? current)
    {
        var underlying = Nullable.GetUnderlyingType(type) ?? type;

        if (underlying.IsEnum)
        {
            var values = Enum.GetValues(underlying).Cast<object>().ToList();
            return values.First(v => !Equals(v, current) && !Equals(v, Activator.CreateInstance(underlying)));
        }

        if (underlying == typeof(int)) return Equals(current, 7) ? 8 : 7;
        if (underlying == typeof(uint)) return Equals(current, 7u) ? 8u : 7u;
        if (underlying == typeof(float)) return Equals(current, 1.5f) ? 2.5f : 1.5f;
        if (underlying == typeof(bool)) return current is true ? false : true;
        if (underlying == typeof(string)) return Equals(current, "probe") ? "probe-2" : "probe";
        if (underlying == typeof(TimeSpan)) return Equals(current, TimeSpan.FromSeconds(33)) ? TimeSpan.FromSeconds(34) : TimeSpan.FromSeconds(33);
        if (underlying == typeof(IReadOnlyList<string>)) return new[] { "--probe" };

        throw new NotSupportedException($"No probe value for {type} — extend NonDefaultValue when a property of that type is added.");
    }
}
