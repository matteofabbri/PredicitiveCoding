using Octokit;

namespace PredictiveCoding.Data.GitHub;

/// <summary>A GitHub repository, normalised.</summary>
public sealed record GitHubRepo(string FullName, string? Description, int Stars, string? Language, string Url);

/// <summary>
/// Data source for GitHub via Octokit: repository metadata, README text and
/// repository search. Network methods need internet access; construction and
/// argument validation run offline.
/// </summary>
public sealed class GitHubSource
{
    private readonly GitHubClient _client;

    public GitHubSource(string productName = "PredictiveCoding.Data", string? token = null)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(productName);
        _client = new GitHubClient(new ProductHeaderValue(productName));
        if (!string.IsNullOrWhiteSpace(token))
            _client.Credentials = new Credentials(token);
    }

    /// <summary>The underlying Octokit client, for advanced scenarios.</summary>
    public GitHubClient Client => _client;

    /// <summary>Fetches a repository's metadata.</summary>
    public async Task<GitHubRepo> GetRepositoryAsync(string owner, string name, CancellationToken cancellationToken = default)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(owner);
        ArgumentException.ThrowIfNullOrWhiteSpace(name);
        Repository repository = await _client.Repository.Get(owner, name).ConfigureAwait(false);
        return Map(repository);
    }

    /// <summary>Fetches the raw text of a repository's README.</summary>
    public async Task<string> GetReadmeTextAsync(string owner, string name, CancellationToken cancellationToken = default)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(owner);
        ArgumentException.ThrowIfNullOrWhiteSpace(name);
        Readme readme = await _client.Repository.Content.GetReadme(owner, name).ConfigureAwait(false);
        return readme.Content;
    }

    /// <summary>Searches repositories.</summary>
    public async Task<IReadOnlyList<GitHubRepo>> SearchRepositoriesAsync(string query, CancellationToken cancellationToken = default)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(query);
        SearchRepositoryResult result = await _client.Search.SearchRepo(new SearchRepositoriesRequest(query)).ConfigureAwait(false);
        return [.. result.Items.Select(Map)];
    }

    private static GitHubRepo Map(Repository repository)
        => new(repository.FullName, repository.Description, repository.StargazersCount, repository.Language, repository.HtmlUrl);
}
