using System.Collections;
using System.Reflection;
using AwesomeAssertions;

namespace LMSupply.Integration.Tests;

/// <summary>
/// Every options type with a <c>Clone()</c> must carry every settable option across the copy. Loaders keep
/// the clone, not the caller's instance, so an option missing from the copy is set by the caller and never
/// seen by the model — no error, no log. In 0.64.0 five clones dropped <see cref="LMSupplyOptionsBase.LogLevel"/>
/// (added to the base type after the clones were written) and the ONNX generator factory dropped seven
/// options in a hand-written copy. This test finds every <c>Clone()</c> by reflection, so the next options
/// type that grows one is covered without a per-module copy of this test.
/// </summary>
public class OptionsCloneCompletenessTests
{
    public static TheoryData<string> CloneableOptionTypes =>
        [.. Cloneable().Select(t => t.FullName!).Order(StringComparer.Ordinal)];

    [Fact]
    public void TheScanFindsTheKnownCloneableTypes()
    {
        // Positive control: an empty theory above would pass vacuously.
        var names = Cloneable().Select(t => t.Name).ToList();
        names.Should().Contain(["GeneratorOptions", "SynthesizerOptions", "RerankerOptions", "TranscriberOptions"]);
    }

    [Theory]
    [MemberData(nameof(CloneableOptionTypes))]
    public void Clone_CarriesEverySettableOption(string typeName)
    {
        var type = Cloneable().Single(t => t.FullName == typeName);
        var props = type.GetProperties(BindingFlags.Public | BindingFlags.Instance).Where(p => p.SetMethod != null).ToList();
        var source = Activator.CreateInstance(type)!;
        var originals = props.ToDictionary(p => p.Name, p => p.GetValue(source));
        foreach (var prop in props)
            prop.SetValue(source, NonDefault(prop.PropertyType, prop.GetValue(source)));

        var clone = type.GetMethod("Clone", Type.EmptyTypes)!.Invoke(source, null);

        foreach (var prop in props)
        {
            var expected = prop.GetValue(source);
            var actual = prop.GetValue(clone);
            actual.Should().BeEquivalentTo(expected, $"{type.Name}.{prop.Name} must survive Clone — add it to the copy");
            actual.Should().NotBeEquivalentTo(originals[prop.Name],
                $"the probe must move {type.Name}.{prop.Name} off its initial value or the check above is vacuous");
        }
    }

    private static IEnumerable<Type> Cloneable()
        => Directory.EnumerateFiles(AppContext.BaseDirectory, "LMSupply*.dll")
            .Select(path => Assembly.Load(AssemblyName.GetAssemblyName(path)))
            .SelectMany(a =>
            {
                try { return (IEnumerable<Type>)a.GetExportedTypes(); }
                catch (ReflectionTypeLoadException ex) { return ex.Types.Where(t => t is { IsPublic: true }).Select(t => t!); }
            })
            .Where(t => t is { IsClass: true, IsAbstract: false } && t.Name.EndsWith("Options", StringComparison.Ordinal))
            .Where(t => t.GetConstructor(Type.EmptyTypes) is not null)
            .Where(t => t.GetMethod("Clone", Type.EmptyTypes) is { } m && m.ReturnType == t);

    /// <summary>A value that differs from the type's default and from the current one.</summary>
    private static object NonDefault(Type type, object? current)
    {
        var t = Nullable.GetUnderlyingType(type) ?? type;
        if (t.IsEnum)
            return Enum.GetValues(t).Cast<object>().First(v => !Equals(v, current) && !Equals(v, Activator.CreateInstance(t)));
        if (t == typeof(int)) return Equals(current, 7) ? 8 : 7;
        if (t == typeof(uint)) return Equals(current, 7u) ? 8u : 7u;
        if (t == typeof(long)) return Equals(current, 7L) ? 8L : 7L;
        if (t == typeof(float)) return Equals(current, 1.5f) ? 2.5f : 1.5f;
        if (t == typeof(double)) return Equals(current, 1.5d) ? 2.5d : 1.5d;
        if (t == typeof(bool)) return current is true ? false : true;
        if (t == typeof(string)) return Equals(current, "probe") ? "probe-2" : "probe";
        if (t == typeof(TimeSpan)) return Equals(current, TimeSpan.FromSeconds(33)) ? TimeSpan.FromSeconds(34) : TimeSpan.FromSeconds(33);
        if (t.IsAssignableFrom(typeof(HashSet<int>))) return new HashSet<int> { 3, 5 };
        if (t.IsAssignableFrom(typeof(List<string>))) return new List<string> { "probe" };
        if (t.IsClass && t.GetConstructor(Type.EmptyTypes) is not null && !typeof(IEnumerable).IsAssignableFrom(t))
            return Activator.CreateInstance(t)!;
        throw new NotSupportedException($"No probe value for {type} — extend NonDefault when an option of that type is added.");
    }
}
