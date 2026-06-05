using PredictiveCoding.Core;
using PredictiveCoding.Data.Audio;
using PredictiveCoding.Data.Video;
using YoutubeExplode;
using YoutubeExplode.Channels;
using YoutubeExplode.Playlists;
using YoutubeExplode.Search;
using YoutubeExplode.Videos;
using YoutubeExplode.Videos.ClosedCaptions;
using YoutubeExplode.Videos.Streams;
// "Video" is also the name of this assembly's video namespace, which would
// shadow the YoutubeExplode metadata type — alias it to keep both usable.
using YtVideo = YoutubeExplode.Videos.Video;

namespace PredictiveCoding.Data.YouTube;

/// <summary>
/// Data source backed by the <c>YoutubeExplode</c> package. Exposes everything
/// the library can pull from YouTube — video/playlist/channel metadata, stream
/// manifests, closed captions and search — and bridges the audio/video streams
/// into the project's <see cref="DenseTensor{T}"/> via the NAudio and OpenCV
/// loaders.
/// </summary>
/// <remarks>
/// All network methods are asynchronous and require internet access; only the
/// static ID/URL parsing helpers run offline.
/// </remarks>
public sealed class YouTubeSource
{
    private readonly YoutubeClient _client;
    private readonly IAudioLoader _audioLoader;
    private readonly IVideoLoader _videoLoader;

    /// <summary>Creates a source with a default <see cref="YoutubeClient"/>.</summary>
    public YouTubeSource() : this(new YoutubeClient()) { }

    /// <summary>Creates a source over a caller-supplied <see cref="HttpClient"/>.</summary>
    public YouTubeSource(HttpClient httpClient) : this(new YoutubeClient(httpClient)) { }

    /// <summary>Creates a source over an existing client, optionally overriding the media loaders.</summary>
    public YouTubeSource(YoutubeClient client, IAudioLoader? audioLoader = null, IVideoLoader? videoLoader = null)
    {
        _client = client ?? throw new ArgumentNullException(nameof(client));
        _audioLoader = audioLoader ?? NAudioLoader.Instance;
        _videoLoader = videoLoader ?? OpenCvVideoLoader.Instance;
    }

    /// <summary>The underlying YoutubeExplode client, for advanced scenarios.</summary>
    public YoutubeClient Client => _client;

    // ── ID / URL parsing (offline) ─────────────────────────────────────────────

    /// <summary>Parses a video ID or any YouTube video URL.</summary>
    public static VideoId ParseVideoId(string idOrUrl)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(idOrUrl);
        return VideoId.TryParse(idOrUrl)
            ?? throw new ArgumentException($"Invalid YouTube video ID or URL: '{idOrUrl}'.", nameof(idOrUrl));
    }

    /// <summary>Parses a playlist ID or playlist URL.</summary>
    public static PlaylistId ParsePlaylistId(string idOrUrl)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(idOrUrl);
        return PlaylistId.TryParse(idOrUrl)
            ?? throw new ArgumentException($"Invalid YouTube playlist ID or URL: '{idOrUrl}'.", nameof(idOrUrl));
    }

    /// <summary>Parses a channel ID, handle, user or channel URL.</summary>
    public static ChannelId ParseChannelId(string idOrUrl)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(idOrUrl);
        return ChannelId.TryParse(idOrUrl)
            ?? throw new ArgumentException($"Invalid YouTube channel ID or URL: '{idOrUrl}'.", nameof(idOrUrl));
    }

    // ── Metadata ───────────────────────────────────────────────────────────────

    /// <summary>Fetches metadata for a single video.</summary>
    public async Task<YtVideo> GetVideoAsync(string videoIdOrUrl, CancellationToken cancellationToken = default)
        => await _client.Videos.GetAsync(ParseVideoId(videoIdOrUrl), cancellationToken).ConfigureAwait(false);

    /// <summary>Fetches metadata for a playlist.</summary>
    public async Task<Playlist> GetPlaylistAsync(string playlistIdOrUrl, CancellationToken cancellationToken = default)
        => await _client.Playlists.GetAsync(ParsePlaylistId(playlistIdOrUrl), cancellationToken).ConfigureAwait(false);

    /// <summary>Fetches metadata for a channel.</summary>
    public async Task<Channel> GetChannelAsync(string channelIdOrUrl, CancellationToken cancellationToken = default)
        => await _client.Channels.GetAsync(ParseChannelId(channelIdOrUrl), cancellationToken).ConfigureAwait(false);

    /// <summary>Fetches the manifest of all available media streams for a video.</summary>
    public async Task<StreamManifest> GetStreamManifestAsync(string videoIdOrUrl, CancellationToken cancellationToken = default)
        => await _client.Videos.Streams.GetManifestAsync(ParseVideoId(videoIdOrUrl), cancellationToken).ConfigureAwait(false);

    // ── Enumerations ─────────────────────────────────────────────────────────────

    /// <summary>Streams the videos of a playlist.</summary>
    public IAsyncEnumerable<PlaylistVideo> GetPlaylistVideosAsync(string playlistIdOrUrl, CancellationToken cancellationToken = default)
        => _client.Playlists.GetVideosAsync(ParsePlaylistId(playlistIdOrUrl), cancellationToken);

    /// <summary>Streams the uploads of a channel.</summary>
    public IAsyncEnumerable<PlaylistVideo> GetChannelUploadsAsync(string channelIdOrUrl, CancellationToken cancellationToken = default)
        => _client.Channels.GetUploadsAsync(ParseChannelId(channelIdOrUrl), cancellationToken);

    /// <summary>Streams video search results for a query.</summary>
    public IAsyncEnumerable<VideoSearchResult> SearchVideosAsync(string query, CancellationToken cancellationToken = default)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(query);
        return _client.Search.GetVideosAsync(query, cancellationToken);
    }

    /// <summary>Streams playlist search results for a query.</summary>
    public IAsyncEnumerable<PlaylistSearchResult> SearchPlaylistsAsync(string query, CancellationToken cancellationToken = default)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(query);
        return _client.Search.GetPlaylistsAsync(query, cancellationToken);
    }

    /// <summary>Streams channel search results for a query.</summary>
    public IAsyncEnumerable<ChannelSearchResult> SearchChannelsAsync(string query, CancellationToken cancellationToken = default)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(query);
        return _client.Search.GetChannelsAsync(query, cancellationToken);
    }

    /// <summary>Streams mixed (video/playlist/channel) search results for a query.</summary>
    public IAsyncEnumerable<ISearchResult> SearchAsync(string query, CancellationToken cancellationToken = default)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(query);
        return _client.Search.GetResultsAsync(query, cancellationToken);
    }

    // ── Closed captions ──────────────────────────────────────────────────────────

    /// <summary>Fetches the closed-caption track manifest for a video.</summary>
    public async Task<ClosedCaptionManifest> GetCaptionManifestAsync(string videoIdOrUrl, CancellationToken cancellationToken = default)
        => await _client.Videos.ClosedCaptions.GetManifestAsync(ParseVideoId(videoIdOrUrl), cancellationToken).ConfigureAwait(false);

    /// <summary>
    /// Downloads a closed-caption track and returns its text. When
    /// <paramref name="language"/> is null the first available track is used.
    /// Returns an empty string when the video has no captions.
    /// </summary>
    public async Task<string> GetCaptionTextAsync(string videoIdOrUrl, string? language = null, CancellationToken cancellationToken = default)
    {
        VideoId id = ParseVideoId(videoIdOrUrl);
        ClosedCaptionManifest manifest =
            await _client.Videos.ClosedCaptions.GetManifestAsync(id, cancellationToken).ConfigureAwait(false);

        ClosedCaptionTrackInfo? trackInfo = language is null
            ? manifest.Tracks.FirstOrDefault()
            : manifest.TryGetByLanguage(language) ?? manifest.Tracks.FirstOrDefault();
        if (trackInfo is null)
            return string.Empty;

        ClosedCaptionTrack track =
            await _client.Videos.ClosedCaptions.GetAsync(trackInfo, cancellationToken).ConfigureAwait(false);
        return string.Join(" ", track.Captions.Select(c => c.Text));
    }

    // ── Raw stream download ──────────────────────────────────────────────────────

    /// <summary>Downloads the best MP4 audio stream of a video to <paramref name="filePath"/>.</summary>
    public async Task<string> DownloadAudioAsync(string videoIdOrUrl, string filePath,
        IProgress<double>? progress = null, CancellationToken cancellationToken = default)
    {
        VideoId id = ParseVideoId(videoIdOrUrl);
        ArgumentException.ThrowIfNullOrEmpty(filePath);

        StreamManifest manifest = await _client.Videos.Streams.GetManifestAsync(id, cancellationToken).ConfigureAwait(false);
        IStreamInfo stream = PickAudioStream(manifest)
            ?? throw new InvalidOperationException($"No MP4 audio stream available for '{id}'.");

        await _client.Videos.Streams.DownloadAsync(stream, filePath, progress, cancellationToken).ConfigureAwait(false);
        return filePath;
    }

    /// <summary>Downloads an MP4 video stream (≤ <paramref name="maxVideoHeight"/>) to <paramref name="filePath"/>.</summary>
    public async Task<string> DownloadVideoAsync(string videoIdOrUrl, string filePath,
        int maxVideoHeight = 720, IProgress<double>? progress = null, CancellationToken cancellationToken = default)
    {
        VideoId id = ParseVideoId(videoIdOrUrl);
        ArgumentException.ThrowIfNullOrEmpty(filePath);

        StreamManifest manifest = await _client.Videos.Streams.GetManifestAsync(id, cancellationToken).ConfigureAwait(false);
        IStreamInfo stream = PickVideoStream(manifest, maxVideoHeight)
            ?? throw new InvalidOperationException($"No MP4 video stream available for '{id}'.");

        await _client.Videos.Streams.DownloadAsync(stream, filePath, progress, cancellationToken).ConfigureAwait(false);
        return filePath;
    }

    // ── Media → tensor ───────────────────────────────────────────────────────────

    /// <summary>Downloads the audio of a video and decodes it into an <see cref="AudioClip"/>.</summary>
    public async Task<AudioClip> GetAudioClipAsync(string videoIdOrUrl,
        AudioLoadOptions? options = null, CancellationToken cancellationToken = default)
    {
        string path = Path.Combine(Path.GetTempPath(), $"yt_audio_{Guid.NewGuid():N}.m4a");
        try
        {
            await DownloadAudioAsync(videoIdOrUrl, path, progress: null, cancellationToken).ConfigureAwait(false);
            return _audioLoader.Load(path, options);
        }
        finally
        {
            TryDelete(path);
        }
    }

    /// <summary>Downloads the audio of a video and returns it as a 1-D sample tensor.</summary>
    public async Task<DenseTensor<float>> GetAudioSamplesAsync(string videoIdOrUrl,
        AudioLoadOptions? options = null, CancellationToken cancellationToken = default)
        => (await GetAudioClipAsync(videoIdOrUrl, options, cancellationToken).ConfigureAwait(false)).Samples;

    /// <summary>Downloads a video and decodes its frames into a <see cref="VideoClip"/>.</summary>
    public async Task<VideoClip> GetVideoClipAsync(string videoIdOrUrl,
        VideoLoadOptions? options = null, int maxVideoHeight = 720, CancellationToken cancellationToken = default)
    {
        string path = Path.Combine(Path.GetTempPath(), $"yt_video_{Guid.NewGuid():N}.mp4");
        try
        {
            await DownloadVideoAsync(videoIdOrUrl, path, maxVideoHeight, progress: null, cancellationToken).ConfigureAwait(false);
            return _videoLoader.Load(path, options);
        }
        finally
        {
            TryDelete(path);
        }
    }

    /// <summary>Downloads a video and returns its frames as a tensor.</summary>
    public async Task<DenseTensor<float>> GetVideoFramesAsync(string videoIdOrUrl,
        VideoLoadOptions? options = null, int maxVideoHeight = 720, CancellationToken cancellationToken = default)
        => (await GetVideoClipAsync(videoIdOrUrl, options, maxVideoHeight, cancellationToken).ConfigureAwait(false)).Frames;

    // ── Helpers ──────────────────────────────────────────────────────────────────

    // Best MP4 audio stream: prefer audio-only (highest bitrate), else muxed.
    private static IStreamInfo? PickAudioStream(StreamManifest manifest)
    {
        var audioOnly = manifest.GetAudioOnlyStreams().Where(s => s.Container == Container.Mp4).ToArray();
        if (audioOnly.Length > 0)
            return audioOnly.GetWithHighestBitrate();

        var muxed = manifest.GetMuxedStreams().Where(s => s.Container == Container.Mp4).ToArray();
        return muxed.Length > 0 ? muxed.GetWithHighestBitrate() : null;
    }

    // Best MP4 video stream within the height cap (muxed + video-only candidates).
    private static IVideoStreamInfo? PickVideoStream(StreamManifest manifest, int maxVideoHeight)
    {
        List<IVideoStreamInfo> candidates = manifest.GetMuxedStreams().Cast<IVideoStreamInfo>()
            .Concat(manifest.GetVideoOnlyStreams())
            .Where(s => s.Container == Container.Mp4)
            .ToList();
        if (candidates.Count == 0)
            return null;

        List<IVideoStreamInfo> capped = candidates.Where(s => s.VideoQuality.MaxHeight <= maxVideoHeight).ToList();
        List<IVideoStreamInfo> pool = capped.Count > 0 ? capped : candidates;
        return pool.OrderByDescending(s => s.VideoQuality.MaxHeight).First();
    }

    private static void TryDelete(string path)
    {
        try
        {
            if (File.Exists(path)) File.Delete(path);
        }
        catch
        {
            // Best-effort cleanup of the temp download.
        }
    }
}
