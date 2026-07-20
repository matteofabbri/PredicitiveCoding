namespace PredictiveCoding.Data.Text;

/// <summary>A candidate language and its estimated probability for a piece of text.</summary>
/// <param name="Language">Language label (e.g. "Italian", "English").</param>
/// <param name="Probability">Posterior probability in [0,1]; candidates are returned highest-first.</param>
public readonly record struct DetectedLanguage(string Language, double Probability);
