using PredictiveCoding.Data.Crypto;
using PredictiveCoding.Data.Finance;
using PredictiveCoding.Data.OpenStreetMap;
using PredictiveCoding.Data.Scholarly;
using PredictiveCoding.Data.StackExchange;
using PredictiveCoding.Data.Weather;
using PredictiveCoding.Data.WorldBank;

namespace PredicitiveCoding.UnitTests;

/// <summary>
/// Offline parser tests for the HTTP/CSV-backed batch-3 sources (Weather,
/// Coinbase, Stack Exchange, OpenAlex, World Bank, Overpass, Stooq).
/// </summary>
public sealed class Lotto3ParsersTests
{
    [Fact]
    public void Weather_ParseHourly_BuildsSeries()
    {
        const string json = """
            { "hourly": {
                "time": ["2026-06-05T00:00", "2026-06-05T01:00"],
                "temperature_2m": [12.5, 13.0],
                "wind_speed_10m": [3.0, 3.5] } }
            """;
        WeatherSeries series = WeatherSource.ParseHourly(json);

        Assert.Equal(2, series.Times.Count);
        Assert.Equal(["temperature_2m", "wind_speed_10m"], series.Variables);
        Assert.Equal(new[] { 2, 2 }, series.Values.Dimensions);
        Assert.Equal(12.5f, series.Values[0, 0]);
        Assert.Equal(3.5f, series.Values[1, 1]);
    }

    [Fact]
    public void Coinbase_ParseSpotPrice_And_Candles()
    {
        Assert.Equal(45000.50m, CoinbaseSource.ParseSpotPrice("""{ "data": { "amount": "45000.50" } }"""));

        DenseTensorAssert(CoinbaseSource.ParseCandles("[[1700000000,100,110,105,108,5.0],[1700003600,108,112,107,111,6.0]]"));

        static void DenseTensorAssert(PredictiveCoding.Core.DenseTensor<float> candles)
        {
            Assert.Equal(new[] { 2, 6 }, candles.Dimensions);
            Assert.Equal(105f, candles[0, 3]); // open
            Assert.Equal(111f, candles[1, 4]); // close
        }
    }

    [Fact]
    public void StackExchange_ParseQuestions()
    {
        const string json = """
            { "items": [ { "question_id": 42, "title": "How to X?", "link": "https://so/42", "score": 10, "tags": ["c#", "data"] } ] }
            """;
        StackExchangeQuestion q = Assert.Single(StackExchangeSource.ParseQuestions(json));
        Assert.Equal(42, q.Id);
        Assert.Equal("How to X?", q.Title);
        Assert.Equal(10, q.Score);
        Assert.Contains("c#", q.Tags);
    }

    [Fact]
    public void OpenAlex_ParseWorks()
    {
        const string json = """
            { "results": [ { "id": "https://openalex.org/W1", "title": "A Work", "publication_year": 2020,
              "doi": "https://doi.org/10.1/x", "authorships": [ { "author": { "display_name": "Jane Doe" } } ] } ] }
            """;
        ScholarlyWork work = Assert.Single(OpenAlexSource.ParseWorks(json));
        Assert.Equal("A Work", work.Title);
        Assert.Equal(2020, work.Year);
        Assert.Contains("Jane Doe", work.Authors);
    }

    [Fact]
    public void WorldBank_ParseIndicator()
    {
        const string json = """
            [ { "page": 1 },
              [ { "indicator": { "id": "NY.GDP.MKTP.CD", "value": "GDP" },
                  "country": { "id": "US", "value": "United States" },
                  "date": "2020", "value": 21000000000000.0 } ] ]
            """;
        WorldBankPoint point = Assert.Single(WorldBankSource.ParseIndicator(json));
        Assert.Equal("US", point.CountryIso);
        Assert.Equal("NY.GDP.MKTP.CD", point.Indicator);
        Assert.Equal(2020, point.Year);
        Assert.True(point.Value > 1e13);
    }

    [Fact]
    public void Overpass_ParseElements()
    {
        const string json = """
            { "elements": [ { "type": "node", "id": 1, "lat": 45.0, "lon": 9.0,
              "tags": { "amenity": "cafe", "name": "Bar Roma" } } ] }
            """;
        OsmElement element = Assert.Single(OverpassSource.ParseElements(json));
        Assert.Equal("node", element.Type);
        Assert.Equal(1, element.Id);
        Assert.Equal(45.0, element.Lat);
        Assert.Equal("cafe", element.Tags["amenity"]);
    }

    [Fact]
    public void Stooq_ParseCsv()
    {
        const string csv = "Date,Open,High,Low,Close,Volume\n2026-06-03,1.0,2.0,0.5,1.5,1000\n2026-06-04,1.5,2.5,1.0,2.0,2000";
        OhlcvSeries series = StooqSource.ParseCsv(csv);

        Assert.Equal(2, series.Dates.Count);
        Assert.Equal(new DateOnly(2026, 6, 3), series.Dates[0]);
        Assert.Equal(new[] { 2, 5 }, series.Values.Dimensions);
        Assert.Equal(1.0f, series.Values[0, 0]); // open
        Assert.Equal(2.0f, series.Values[1, 3]); // close
    }

    [Fact]
    public void Parsers_RejectEmptyInput()
    {
        Assert.Throws<ArgumentException>(() => WeatherSource.ParseHourly(""));
        Assert.Throws<ArgumentException>(() => CoinbaseSource.ParseCandles(""));
        Assert.Throws<ArgumentException>(() => StackExchangeSource.ParseQuestions(""));
        Assert.Throws<ArgumentException>(() => StooqSource.ParseCsv(""));
    }
}
