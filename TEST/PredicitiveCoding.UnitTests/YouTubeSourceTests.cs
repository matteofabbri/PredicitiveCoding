using PredictiveCoding.Data.YouTube;
using YoutubeExplode;

namespace PredicitiveCoding.UnitTests;

/// <summary>
/// Offline tests for <see cref="YouTubeSource"/>: construction, ID/URL parsing
/// and argument validation. The network methods are not exercised here — hitting
/// YouTube would be non-deterministic and against the spirit of a unit test.
/// </summary>
public sealed class YouTubeSourceTests
{
    // ── Construction ─────────────────────────────────────────────────────────

    [Fact]
    public void Constructors_ExposeAClient()
    {
        Assert.NotNull(new YouTubeSource().Client);
        Assert.NotNull(new YouTubeSource(new HttpClient()).Client);
        Assert.NotNull(new YouTubeSource(new YoutubeClient()).Client);
    }

    [Fact]
    public void Constructor_NullClient_Throws()
        => Assert.Throws<ArgumentNullException>(() => new YouTubeSource((YoutubeClient)null!));

    // ── ID / URL parsing (happy path) ────────────────────────────────────────

    [Theory]
    [InlineData("dQw4w9WgXcQ")]
    [InlineData("https://www.youtube.com/watch?v=dQw4w9WgXcQ")]
    [InlineData("https://youtu.be/dQw4w9WgXcQ")]
    public void ParseVideoId_AcceptsIdsAndUrls(string input)
        => Assert.Equal("dQw4w9WgXcQ", YouTubeSource.ParseVideoId(input).ToString());

    [Theory]
    [InlineData("UC_x5XG1OV2P6uZZ5FSM9Ttw")]
    [InlineData("https://www.youtube.com/channel/UC_x5XG1OV2P6uZZ5FSM9Ttw")]
    public void ParseChannelId_AcceptsIdsAndUrls(string input)
        => Assert.Equal("UC_x5XG1OV2P6uZZ5FSM9Ttw", YouTubeSource.ParseChannelId(input).ToString());

    [Theory]
    [InlineData("PLI5YfMzCfRtZ8eV576YoY3vIYrHjyVm_e")]
    [InlineData("https://www.youtube.com/playlist?list=PLI5YfMzCfRtZ8eV576YoY3vIYrHjyVm_e")]
    public void ParsePlaylistId_AcceptsIdsAndUrls(string input)
        => Assert.Equal("PLI5YfMzCfRtZ8eV576YoY3vIYrHjyVm_e", YouTubeSource.ParsePlaylistId(input).ToString());

    // ── ID / URL parsing (invalid) ───────────────────────────────────────────

    [Fact]
    public void ParseVideoId_RejectsNullEmptyAndGarbage()
    {
        Assert.Throws<ArgumentNullException>(() => YouTubeSource.ParseVideoId(null!));
        Assert.Throws<ArgumentException>(() => YouTubeSource.ParseVideoId(""));
        Assert.Throws<ArgumentException>(() => YouTubeSource.ParseVideoId("   "));
        Assert.Throws<ArgumentException>(() => YouTubeSource.ParseVideoId("not a valid video id"));
    }

    [Fact]
    public void ParsePlaylistId_RejectsGarbage()
        => Assert.Throws<ArgumentException>(() => YouTubeSource.ParsePlaylistId("not a valid playlist"));

    [Fact]
    public void ParseChannelId_RejectsGarbage()
        => Assert.Throws<ArgumentException>(() => YouTubeSource.ParseChannelId("not a valid channel"));

    // ── Method-level validation (no network reached) ─────────────────────────

    [Fact]
    public async Task AsyncMethods_RejectInvalidVideoId()
    {
        var source = new YouTubeSource();
        await Assert.ThrowsAsync<ArgumentException>(() => source.GetVideoAsync("bad id with spaces"));
        await Assert.ThrowsAsync<ArgumentException>(() => source.GetStreamManifestAsync("bad id with spaces"));
        await Assert.ThrowsAsync<ArgumentException>(() => source.GetCaptionTextAsync("bad id with spaces"));
        await Assert.ThrowsAsync<ArgumentException>(() => source.GetAudioClipAsync("bad id with spaces"));
        await Assert.ThrowsAsync<ArgumentException>(() => source.GetVideoClipAsync("bad id with spaces"));
    }

    [Fact]
    public void EnumerationMethods_RejectInvalidInputSynchronously()
    {
        var source = new YouTubeSource();
        Assert.Throws<ArgumentException>(() => source.GetPlaylistVideosAsync("bad playlist with spaces"));
        Assert.Throws<ArgumentException>(() => source.GetChannelUploadsAsync("bad channel with spaces"));
        Assert.Throws<ArgumentException>(() => source.SearchVideosAsync(""));
        Assert.Throws<ArgumentException>(() => source.SearchPlaylistsAsync("   "));
        Assert.Throws<ArgumentNullException>(() => source.SearchChannelsAsync(null!));
    }

    [Fact]
    public async Task DownloadMethods_RejectEmptyFilePath()
    {
        var source = new YouTubeSource();
        // A valid id passes parsing, then the empty path is rejected before any download.
        await Assert.ThrowsAsync<ArgumentException>(() => source.DownloadAudioAsync("dQw4w9WgXcQ", ""));
        await Assert.ThrowsAsync<ArgumentException>(() => source.DownloadVideoAsync("dQw4w9WgXcQ", ""));
    }
}
