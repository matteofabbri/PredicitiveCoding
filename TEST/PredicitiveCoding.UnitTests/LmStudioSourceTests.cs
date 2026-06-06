using PredictiveCoding.Data.LmStudio;

namespace PredicitiveCoding.UnitTests;

/// <summary>Offline tests for the LM Studio source: parsing of the
/// OpenAI-compatible responses and argument validation. No network is used.</summary>
public sealed class LmStudioSourceTests
{
    [Fact]
    public void ParseModels_ReadsEntries()
    {
        const string json = """
            { "object": "list", "data": [
                { "id": "llama-3.2-3b-instruct", "object": "model", "owned_by": "organization_owner" },
                { "id": "nomic-embed-text-v1.5", "object": "model", "owned_by": "organization_owner" } ] }
            """;

        IReadOnlyList<LmStudioModel> models = LmStudioSource.ParseModels(json);

        Assert.Equal(2, models.Count);
        Assert.Equal("llama-3.2-3b-instruct", models[0].Id);
        Assert.Equal("organization_owner", models[0].OwnedBy);
        Assert.Equal("nomic-embed-text-v1.5", models[1].Id);
    }

    [Fact]
    public void ParseChatCompletion_ReadsContentAndUsage()
    {
        const string json = """
            { "id": "chatcmpl-1", "object": "chat.completion", "model": "llama-3.2-3b-instruct",
              "choices": [ { "index": 0,
                  "message": { "role": "assistant", "content": "Hello there!", "reasoning_content": "The user greeted me." },
                  "finish_reason": "stop" } ],
              "usage": { "prompt_tokens": 12, "completion_tokens": 3, "total_tokens": 15 } }
            """;

        LmStudioCompletion completion = LmStudioSource.ParseChatCompletion(json);

        Assert.Equal("Hello there!", completion.Content);
        Assert.Equal("The user greeted me.", completion.Reasoning);
        Assert.Equal("stop", completion.FinishReason);
        Assert.Equal("llama-3.2-3b-instruct", completion.Model);
        Assert.Equal(12, completion.PromptTokens);
        Assert.Equal(3, completion.CompletionTokens);
    }

    [Fact]
    public void ParseEmbedding_ReadsVector()
    {
        const string json = """
            { "object": "list", "model": "nomic-embed-text-v1.5",
              "data": [ { "object": "embedding", "index": 0, "embedding": [0.1, -0.2, 0.3] } ],
              "usage": { "prompt_tokens": 4, "total_tokens": 4 } }
            """;

        LmStudioEmbedding embedding = LmStudioSource.ParseEmbedding(json);

        Assert.Equal(3, embedding.Vector.Count);
        Assert.True(MathF.Abs(embedding.Vector[0] - 0.1f) < 1e-4f);
        Assert.True(MathF.Abs(embedding.Vector[1] + 0.2f) < 1e-4f);
        Assert.True(MathF.Abs(embedding.Vector[2] - 0.3f) < 1e-4f);
        Assert.Equal("nomic-embed-text-v1.5", embedding.Model);
        Assert.Equal(4, embedding.PromptTokens);
    }

    [Fact]
    public void Parsers_RejectEmpty()
    {
        Assert.Throws<ArgumentException>(() => LmStudioSource.ParseModels(""));
        Assert.Throws<ArgumentException>(() => LmStudioSource.ParseChatCompletion(""));
        Assert.Throws<ArgumentException>(() => LmStudioSource.ParseEmbedding(""));
    }

    [Fact]
    public async Task Methods_ValidateArguments()
    {
        using var source = new LmStudioSource();
        await Assert.ThrowsAsync<ArgumentException>(() => source.GenerateAsync(""));
        await Assert.ThrowsAsync<ArgumentException>(() => source.EmbedAsync(""));
        await Assert.ThrowsAsync<ArgumentException>(() => source.ChatAsync([]));
    }
}
