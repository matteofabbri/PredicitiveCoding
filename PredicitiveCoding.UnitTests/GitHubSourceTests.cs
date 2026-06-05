using PredictiveCoding.Data.GitHub;

namespace PredicitiveCoding.UnitTests;

/// <summary>
/// Offline tests for the GitHub source: construction and argument validation.
/// Network calls are not exercised.
/// </summary>
public sealed class GitHubSourceTests
{
    [Fact]
    public void Constructor_ExposesClient()
        => Assert.NotNull(new GitHubSource().Client);

    [Fact]
    public void Constructor_EmptyProduct_Throws()
        => Assert.Throws<ArgumentException>(() => new GitHubSource(""));

    [Fact]
    public async Task Methods_ValidateArgumentsBeforeNetwork()
    {
        var source = new GitHubSource();
        await Assert.ThrowsAsync<ArgumentException>(() => source.GetRepositoryAsync("", "repo"));
        await Assert.ThrowsAsync<ArgumentException>(() => source.GetRepositoryAsync("owner", " "));
        await Assert.ThrowsAsync<ArgumentException>(() => source.GetReadmeTextAsync("owner", ""));
        await Assert.ThrowsAsync<ArgumentException>(() => source.SearchRepositoriesAsync(""));
    }
}
