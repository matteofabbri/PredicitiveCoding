using MQTTnet;
using PredictiveCoding.Core;

namespace PredictiveCoding.Data.Mqtt;

/// <summary>A message received from an MQTT broker.</summary>
public sealed record MqttMessage(string Topic, string Payload);

/// <summary>
/// Streaming sensor source over MQTT (MQTTnet). Collecting messages needs a
/// broker; the payload-to-tensor helper is static and runs offline.
/// </summary>
public static class MqttSensor
{
    /// <summary>
    /// Connects to a broker, subscribes to <paramref name="topic"/> and collects up to
    /// <paramref name="maxMessages"/> messages or until <paramref name="timeout"/> elapses.
    /// </summary>
    public static async Task<IReadOnlyList<MqttMessage>> CollectAsync(
        string host, string topic, int maxMessages, TimeSpan timeout, int port = 1883, CancellationToken cancellationToken = default)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(host);
        ArgumentException.ThrowIfNullOrWhiteSpace(topic);
        if (maxMessages <= 0) throw new ArgumentOutOfRangeException(nameof(maxMessages));

        var factory = new MqttClientFactory();
        using IMqttClient client = factory.CreateMqttClient();

        var messages = new List<MqttMessage>();
        var completed = new TaskCompletionSource();

        client.ApplicationMessageReceivedAsync += args =>
        {
            messages.Add(new MqttMessage(args.ApplicationMessage.Topic, args.ApplicationMessage.ConvertPayloadToString() ?? string.Empty));
            if (messages.Count >= maxMessages)
                completed.TrySetResult();
            return Task.CompletedTask;
        };

        MqttClientOptions options = new MqttClientOptionsBuilder().WithTcpServer(host, port).Build();
        await client.ConnectAsync(options, cancellationToken).ConfigureAwait(false);
        await client.SubscribeAsync(topic, cancellationToken: cancellationToken).ConfigureAwait(false);

        await Task.WhenAny(completed.Task, Task.Delay(timeout, cancellationToken)).ConfigureAwait(false);
        await client.DisconnectAsync(cancellationToken: cancellationToken).ConfigureAwait(false);
        return messages;
    }

    /// <summary>Wraps a sequence of numeric sensor readings into a 1-D tensor.</summary>
    public static DenseTensor<float> ToTensor(IReadOnlyList<float> values)
    {
        ArgumentNullException.ThrowIfNull(values);
        if (values.Count == 0)
            throw new ArgumentException("At least one value is required.", nameof(values));
        return new DenseTensor<float>([.. values], values.Count);
    }
}
