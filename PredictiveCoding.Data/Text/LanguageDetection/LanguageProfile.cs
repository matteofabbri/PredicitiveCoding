// Copyright 2014 Pēteris Ņikiforovs
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

using System.Text.Json.Serialization;

namespace PredictiveCoding.Data.Text;

// ─────────────────────────────────────────────────────────────────────────────
// LanguageProfile  —  one embedded langdetect profile (Resources/LangProfiles/<code>).
//
// The native JSON format of the original language-detection project (Nakatani
// Shuyo): per-language counts of 1–3 char n-grams over Wikipedia text, plus the
// total n-gram count per order. The 53 profile files are imported verbatim from
// Maddalena.Cognitive (production-proven); only the parser changed (Newtonsoft →
// System.Text.Json), the data is byte-identical.
// ─────────────────────────────────────────────────────────────────────────────

internal sealed class LanguageProfile
{
    /// <summary>ISO 639-1 code of the profile (e.g. "it"; Chinese uses "zh-cn"/"zh-tw").</summary>
    [JsonPropertyName("name")]
    public string Code { get; set; } = "";

    /// <summary>n-gram → occurrence count. Keys keep case and word-boundary spaces (" ca", "ne ").</summary>
    [JsonPropertyName("freq")]
    public Dictionary<string, int> Frequencies { get; set; } = new();

    /// <summary>Total n-gram occurrences per order; index 0 = 1-grams, 1 = 2-grams, 2 = 3-grams.</summary>
    [JsonPropertyName("n_words")]
    public long[] WordCount { get; set; } = [];
}
