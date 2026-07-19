using System.Net.Http.Headers;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace PredictiveCoding.Data.LmStudio;

/// <summary>A model advertised by the LM Studio server (<c>GET /v1/models</c>).</summary>
public sealed record LmStudioModel(string Id, string? OwnedBy);

/// <summary>A single chat message (<c>system</c> / <c>user</c> / <c>assistant</c>).</summary>
public sealed record LmStudioMessage(string Role, string Content)
{
    /// <summary>A system message that steers the assistant.</summary>
    public static LmStudioMessage System(string content) => new("system", content);

    /// <summary>A user turn.</summary>
    public static LmStudioMessage User(string content) => new("user", content);

    /// <summary>An assistant turn (for multi-turn context).</summary>
    public static LmStudioMessage Assistant(string content) => new("assistant", content);
}

/// <summary>The result of a chat-completion request.</summary>
/// <param name="Content">The assistant's answer.</param>
/// <param name="Reasoning">Chain-of-thought, when a reasoning model returns it separately (<c>reasoning_content</c>).</param>
public sealed record LmStudioCompletion(
    string Content,
    string? Reasoning,
    string? FinishReason,
    string? Model,
    int? PromptTokens,
    int? CompletionTokens);

/// <summary>The result of an embedding request.</summary>
public sealed record LmStudioEmbedding(
    IReadOnlyList<float> Vector,
    string? Model,
    int? PromptTokens);

/// <summary>
/// Data source backed by a local <b>LM Studio</b> server, which exposes an
/// OpenAI-compatible REST API (default <c>http://localhost:1234/v1</c>). It can
/// generate text (chat completions → text records for the LLM tokenizers) and
/// produce embeddings (→ vectors that feed a <c>DenseTensor</c>), or list the
/// currently loaded models. Response parsing is static and runs offline.
/// </summary>
public sealed class LmStudioSource : IDisposable
{
    /// <summary>Default base URL of the LM Studio local server.</summary>
    public const string DefaultBaseUrl = "http://localhost:1234/v1";

    private static readonly JsonSerializerOptions RequestJson = new()
    {
        DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
    };

    private readonly string _baseUrl;
    private readonly string? _defaultModel;
    private readonly string? _apiKey;
    private readonly HttpClient _http;
    private readonly bool _ownsHttp;

    /// <param name="baseUrl">Server base URL; defaults to <see cref="DefaultBaseUrl"/>.</param>
    /// <param name="defaultModel">Model id to use when a call does not specify one. When null, LM Studio uses the currently loaded model.</param>
    /// <param name="httpClient">Optional shared client; when omitted one is created and owned by this instance.</param>
    /// <param name="apiKey">Optional bearer token. LM Studio ignores it by default, but it is sent when provided (useful behind a proxy).</param>
    public LmStudioSource(
        string? baseUrl = null,
        string? defaultModel = null,
        HttpClient? httpClient = null,
        string? apiKey = null)
    {
        _baseUrl = (string.IsNullOrWhiteSpace(baseUrl) ? DefaultBaseUrl : baseUrl).TrimEnd('/');
        _defaultModel = defaultModel;
        _apiKey = apiKey;
        _http = httpClient ?? new HttpClient();
        _ownsHttp = httpClient is null;
    }

    /// <summary>Lists the models the server currently exposes.</summary>
    public async Task<IReadOnlyList<LmStudioModel>> ListModelsAsync(CancellationToken cancellationToken = default)
        => ParseModels(await GetAsync("/models", cancellationToken).ConfigureAwait(false));

    /// <summary>Sends a chat conversation and returns the assistant's reply.</summary>
    /// <param name="messages">The conversation so far (at least one message).</param>
    /// <param name="model">Overrides the default model for this call.</param>
    /// <param name="temperature">Sampling temperature.</param>
    /// <param name="maxTokens">Optional cap on generated tokens.</param>
    public async Task<LmStudioCompletion> ChatAsync(
        IReadOnlyList<LmStudioMessage> messages,
        string? model = null,
        double temperature = 0.7,
        int? maxTokens = null,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(messages);
        if (messages.Count == 0)
            throw new ArgumentException("At least one message is required.", nameof(messages));

        var body = new
        {
            model = model ?? _defaultModel,
            messages = messages.Select(m => new { role = m.Role, content = m.Content }),
            temperature,
            max_tokens = maxTokens,
            stream = false,
        };

        return ParseChatCompletion(await PostAsync("/chat/completions", body, cancellationToken).ConfigureAwait(false));
    }

    /// <summary>Convenience single-prompt generation with an optional system prompt.</summary>
    public Task<LmStudioCompletion> GenerateAsync(
        string prompt,
        string? systemPrompt = null,
        string? model = null,
        double temperature = 0.7,
        int? maxTokens = null,
        CancellationToken cancellationToken = default)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(prompt);

        var messages = new List<LmStudioMessage>(2);
        if (!string.IsNullOrWhiteSpace(systemPrompt))
            messages.Add(LmStudioMessage.System(systemPrompt));
        messages.Add(LmStudioMessage.User(prompt));

        return ChatAsync(messages, model, temperature, maxTokens, cancellationToken);
    }

    /// <summary>Produces an embedding vector for the given text (requires a loaded embedding model).</summary>
    public async Task<LmStudioEmbedding> EmbedAsync(
        string input,
        string? model = null,
        CancellationToken cancellationToken = default)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(input);
        var body = new { model = model ?? _defaultModel, input };
        return ParseEmbedding(await PostAsync("/embeddings", body, cancellationToken).ConfigureAwait(false));
    }

    public void Dispose()
    {
        if (_ownsHttp) _http.Dispose();
    }

    // ── HTTP plumbing ──────────────────────────────────────────────────────────

    private async Task<string> GetAsync(string path, CancellationToken cancellationToken)
    {
        using var request = new HttpRequestMessage(HttpMethod.Get, _baseUrl + path);
        Authorize(request);
        using HttpResponseMessage response = await _http.SendAsync(request, cancellationToken).ConfigureAwait(false);
        response.EnsureSuccessStatusCode();
        return await response.Content.ReadAsStringAsync(cancellationToken).ConfigureAwait(false);
    }

    private async Task<string> PostAsync(string path, object body, CancellationToken cancellationToken)
    {
        string json = JsonSerializer.Serialize(body, RequestJson);
        using var request = new HttpRequestMessage(HttpMethod.Post, _baseUrl + path)
        {
            Content = new StringContent(json, Encoding.UTF8, "application/json"),
        };
        Authorize(request);
        using HttpResponseMessage response = await _http.SendAsync(request, cancellationToken).ConfigureAwait(false);
        string content = await response.Content.ReadAsStringAsync(cancellationToken).ConfigureAwait(false);
        if (!response.IsSuccessStatusCode)
            throw new HttpRequestException(
                $"{path} returned {(int)response.StatusCode} ({response.StatusCode}): {ExtractErrorMessage(content)}",
                inner: null, response.StatusCode);
        return content;
    }

    /// <summary>The server's <c>error.message</c> when the body carries one, else the raw body.</summary>
    private static string ExtractErrorMessage(string body)
    {
        try
        {
            using var document = JsonDocument.Parse(body);
            if (document.RootElement.TryGetProperty("error", out JsonElement error))
                return (error.ValueKind == JsonValueKind.Object && error.TryGetProperty("message", out JsonElement message)
                    ? message.GetString() : error.GetString()) ?? body;
        }
        catch (JsonException) { }
        return body;
    }

    private void Authorize(HttpRequestMessage request)
    {
        if (!string.IsNullOrEmpty(_apiKey))
            request.Headers.Authorization = new AuthenticationHeaderValue("Bearer", _apiKey);
    }

    // ── Offline parsers ────────────────────────────────────────────────────────

    /// <summary>Parses a <c>/v1/models</c> response (offline).</summary>
    public static IReadOnlyList<LmStudioModel> ParseModels(string json)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(json);
        using var document = JsonDocument.Parse(json);

        var models = new List<LmStudioModel>();
        if (document.RootElement.TryGetProperty("data", out JsonElement data) && data.ValueKind == JsonValueKind.Array)
            foreach (JsonElement model in data.EnumerateArray())
                models.Add(new LmStudioModel(GetString(model, "id") ?? string.Empty, GetString(model, "owned_by")));
        return models;
    }

    /// <summary>Parses a chat-completion response (offline).</summary>
    public static LmStudioCompletion ParseChatCompletion(string json)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(json);
        using var document = JsonDocument.Parse(json);
        JsonElement root = document.RootElement;

        string content = string.Empty;
        string? reasoning = null;
        string? finishReason = null;
        if (root.TryGetProperty("choices", out JsonElement choices) &&
            choices.ValueKind == JsonValueKind.Array && choices.GetArrayLength() > 0)
        {
            JsonElement first = choices[0];
            if (first.TryGetProperty("message", out JsonElement message))
            {
                content = GetString(message, "content") ?? string.Empty;
                reasoning = GetString(message, "reasoning_content");
            }
            finishReason = GetString(first, "finish_reason");
        }

        int? promptTokens = null, completionTokens = null;
        if (root.TryGetProperty("usage", out JsonElement usage) && usage.ValueKind == JsonValueKind.Object)
        {
            promptTokens = GetInt(usage, "prompt_tokens");
            completionTokens = GetInt(usage, "completion_tokens");
        }

        return new LmStudioCompletion(content, reasoning, finishReason, GetString(root, "model"), promptTokens, completionTokens);
    }

    /// <summary>Parses an embedding response, returning the first vector (offline).</summary>
    public static LmStudioEmbedding ParseEmbedding(string json)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(json);
        using var document = JsonDocument.Parse(json);
        JsonElement root = document.RootElement;

        var vector = new List<float>();
        if (root.TryGetProperty("data", out JsonElement data) &&
            data.ValueKind == JsonValueKind.Array && data.GetArrayLength() > 0 &&
            data[0].TryGetProperty("embedding", out JsonElement embedding) && embedding.ValueKind == JsonValueKind.Array)
        {
            foreach (JsonElement value in embedding.EnumerateArray())
                if (value.ValueKind == JsonValueKind.Number)
                    vector.Add(value.GetSingle());
        }

        int? promptTokens = root.TryGetProperty("usage", out JsonElement usage) && usage.ValueKind == JsonValueKind.Object
            ? GetInt(usage, "prompt_tokens")
            : null;

        return new LmStudioEmbedding(vector, GetString(root, "model"), promptTokens);
    }

    private static string? GetString(JsonElement parent, string name)
        => parent.TryGetProperty(name, out JsonElement element) && element.ValueKind == JsonValueKind.String
            ? element.GetString()
            : null;

    private static int? GetInt(JsonElement parent, string name)
        => parent.TryGetProperty(name, out JsonElement element) && element.ValueKind == JsonValueKind.Number
            ? element.GetInt32()
            : null;
}
