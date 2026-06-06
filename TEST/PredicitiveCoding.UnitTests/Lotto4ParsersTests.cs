using PredictiveCoding.Core;
using PredictiveCoding.Data.Audio;
using PredictiveCoding.Data.Mastodon;
using PredictiveCoding.Data.Mqtt;
using PredictiveCoding.Data.PubMed;
using PredictiveCoding.Data.Reddit;
using PredictiveCoding.Data.Sparql;
using PredictiveCoding.Data.Video;

namespace PredicitiveCoding.UnitTests;

/// <summary>
/// Offline tests for the batch-4 sources whose parsers/helpers run without
/// network or hardware (PubMed, Reddit, Mastodon, SPARQL validation, microphone
/// PCM conversion, MQTT tensor helper, webcam unavailable-device error).
/// </summary>
public sealed class Lotto4ParsersTests
{
    [Fact]
    public void PubMed_ParseIds_And_Summaries()
    {
        Assert.Equal(["123", "456"], PubMedSource.ParseIds("""{ "esearchresult": { "idlist": ["123", "456"] } }"""));

        const string summary = """
            { "result": { "uids": ["123"], "123": {
                "uid": "123", "title": "A study", "pubdate": "2020 Jan", "fulljournalname": "Nature",
                "authors": [ { "name": "Doe J" }, { "name": "Roe K" } ] } } }
            """;
        PubMedArticle article = Assert.Single(PubMedSource.ParseSummaries(summary));
        Assert.Equal("A study", article.Title);
        Assert.Equal("Nature", article.Journal);
        Assert.Contains("Doe J", article.Authors);
    }

    [Fact]
    public void Reddit_ParsePosts()
    {
        const string json = """
            { "data": { "children": [ { "kind": "t3", "data": {
                "id": "abc", "title": "Title", "author": "user", "url": "https://x", "score": 42,
                "subreddit": "test", "selftext": "body" } } ] } }
            """;
        RedditPost post = Assert.Single(RedditSource.ParsePosts(json));
        Assert.Equal("abc", post.Id);
        Assert.Equal("Title", post.Title);
        Assert.Equal(42, post.Score);
    }

    [Fact]
    public void Mastodon_ParseStatuses_StripsHtml()
    {
        const string json = """
            [ { "id": "1", "content": "<p>Hello <b>world</b></p>", "url": "https://m/1",
                "created_at": "2026-06-05T10:00:00Z", "account": { "username": "alice" } } ]
            """;
        MastodonStatus status = Assert.Single(MastodonSource.ParseStatuses(json));
        Assert.Equal("alice", status.Account);
        Assert.Equal("Hello world", status.Text);
        Assert.NotNull(status.CreatedAt);
    }

    [Fact]
    public void Sparql_ConstructionAndValidation()
    {
        Assert.NotNull(SparqlSource.DBpedia());
        Assert.NotNull(SparqlSource.Wikidata());
        Assert.Throws<ArgumentException>(() => new SparqlSource(""));
    }

    [Fact]
    public async Task Sparql_QueryRejectsEmpty()
    {
        using var source = SparqlSource.DBpedia();
        await Assert.ThrowsAsync<ArgumentException>(() => source.QueryAsync(""));
    }

    [Fact]
    public void Microphone_PcmToSamples_Converts16Bit()
    {
        byte[] pcm =
        [
            .. BitConverter.GetBytes((short)0),
            .. BitConverter.GetBytes((short)16384),
            .. BitConverter.GetBytes((short)-16384),
        ];
        DenseTensor<float> samples = MicrophoneCapture.PcmToSamples(pcm);
        Assert.Equal([0f, 0.5f, -0.5f], samples.Buffer);
    }

    [Fact]
    public void Mqtt_ToTensor()
    {
        Assert.Equal(3, MqttSensor.ToTensor([1f, 2f, 3f]).Length);
        Assert.Throws<ArgumentException>(() => MqttSensor.ToTensor([]));
    }

    [Fact]
    public async Task Mqtt_CollectValidatesArguments()
    {
        await Assert.ThrowsAsync<ArgumentException>(() => MqttSensor.CollectAsync("", "topic", 1, TimeSpan.FromSeconds(1)));
        await Assert.ThrowsAsync<ArgumentException>(() => MqttSensor.CollectAsync("host", "", 1, TimeSpan.FromSeconds(1)));
    }

    [Fact]
    public void Webcam_UnavailableDevice_Throws()
        => Assert.Throws<InvalidOperationException>(() => WebcamCapture.CaptureFrames(1, deviceIndex: 99));
}
