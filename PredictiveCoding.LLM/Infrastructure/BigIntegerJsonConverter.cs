using System.Buffers;
using System.Globalization;
using System.Numerics;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace PredictiveCoding.LLM.Infrastructure;

/// <summary>
/// BigInteger ↔ JSON. System.Text.Json does not support BigInteger natively;
/// this converter writes a plain JSON number (the JSON grammar puts no limit on
/// digits) and reads back numbers of ANY length — plus strings, so counters
/// serialized by other conventions still open. Odometer files written when the
/// counters were <c>long</c> parse unchanged: they are just small numbers.
/// </summary>
public sealed class BigIntegerJsonConverter : JsonConverter<BigInteger>
{
    public override BigInteger Read(ref Utf8JsonReader reader, Type typeToConvert, JsonSerializerOptions options)
    {
        if (reader.TokenType == JsonTokenType.String)
            return BigInteger.Parse(reader.GetString()!, NumberStyles.Integer, CultureInfo.InvariantCulture);

        if (reader.TokenType != JsonTokenType.Number)
            throw new JsonException($"Expected a JSON number for BigInteger, found {reader.TokenType}.");

        // The raw digits, straight from the buffer: going through long/double
        // would defeat the purpose (truncation is exactly what BigInteger avoids).
        ReadOnlySpan<byte> raw = reader.HasValueSequence ? reader.ValueSequence.ToArray() : reader.ValueSpan;
        return BigInteger.Parse(Encoding.UTF8.GetString(raw), NumberStyles.Integer, CultureInfo.InvariantCulture);
    }

    public override void Write(Utf8JsonWriter writer, BigInteger value, JsonSerializerOptions options)
        => writer.WriteRawValue(value.ToString(CultureInfo.InvariantCulture), skipInputValidation: true);
}
