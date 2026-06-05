using MimeKit;

namespace PredictiveCoding.Data.Email;

/// <summary>A parsed email message, normalised.</summary>
public sealed record EmailMessage(
    string? From,
    IReadOnlyList<string> To,
    string? Subject,
    DateTimeOffset Date,
    string? TextBody,
    string? HtmlBody);

/// <summary>
/// Reads email messages (.eml / MIME) with MimeKit.
/// </summary>
public static class EmailReader
{
    /// <summary>Reads an .eml file.</summary>
    public static EmailMessage Read(string path)
    {
        ArgumentException.ThrowIfNullOrEmpty(path);
        if (!File.Exists(path))
            throw new FileNotFoundException("Email file not found.", path);
        return Map(MimeMessage.Load(path));
    }

    /// <summary>Reads a MIME message from a stream.</summary>
    public static EmailMessage Parse(Stream stream)
    {
        ArgumentNullException.ThrowIfNull(stream);
        return Map(MimeMessage.Load(stream));
    }

    private static EmailMessage Map(MimeMessage message)
        => new(
            From: message.From.Count > 0 ? message.From.ToString() : null,
            To: [.. message.To.Select(address => address.ToString())],
            Subject: message.Subject,
            Date: message.Date,
            TextBody: message.TextBody,
            HtmlBody: message.HtmlBody);
}
