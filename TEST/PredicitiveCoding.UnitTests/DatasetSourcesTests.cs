using PredictiveCoding.Data.Datasets;

namespace PredicitiveCoding.UnitTests;

/// <summary>Offline tests for the R (Rdatasets) and Python (seaborn-data) dataset
/// sources: catalogue/index parsing, mixed-type CSV parsing and argument
/// validation. No network is used.</summary>
public sealed class DatasetSourcesTests
{
    // ── Shared string-typed CSV table ────────────────────────────────────────

    [Fact]
    public void DatasetCsv_Parse_KeepsMixedTypeColumns()
    {
        // The classic iris layout: a row-name column, numeric columns and a string factor.
        const string csv = """
            rownames,Sepal.Length,Sepal.Width,Petal.Length,Petal.Width,Species
            1,5.1,3.5,1.4,0.2,setosa
            2,4.9,3,1.4,0.2,setosa
            """;

        DatasetTable table = DatasetCsv.Parse(csv);

        Assert.Equal(6, table.ColumnCount);
        Assert.Equal(2, table.RowCount);
        Assert.Equal(["rownames", "Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width", "Species"], table.Columns);
        Assert.Equal(["setosa", "setosa"], table.Column("Species"));
        Assert.Equal("5.1", table.Rows[0][1]);
    }

    [Fact]
    public void DatasetCsv_Parse_HandlesQuotedDelimiters()
    {
        DatasetTable table = DatasetCsv.Parse("name,note\n\"Doe, Jane\",hello");

        Assert.Equal("Doe, Jane", table.Rows[0][0]);
        Assert.Equal("hello", table.Rows[0][1]);
    }

    [Fact]
    public void DatasetCsv_Column_MissingThrows()
    {
        DatasetTable table = DatasetCsv.Parse("a,b\n1,2");
        Assert.Equal(-1, table.IndexOf("zzz"));
        Assert.Throws<ArgumentException>(() => table.Column("zzz"));
    }

    // ── R: Rdatasets ─────────────────────────────────────────────────────────

    [Fact]
    public void RDatasets_ParseIndex_ReadsEntriesByHeaderName()
    {
        const string csv = """
            "Package","Item","Title","Rows","Cols","n_binary","n_character","n_factor","n_logical","n_numeric","CSV","Doc"
            "AER","Affairs","Fair's Extramarital Affairs Data",601,9,2,0,2,0,7,"https://vincentarelbundock.github.io/Rdatasets/csv/AER/Affairs.csv","https://vincentarelbundock.github.io/Rdatasets/doc/AER/Affairs.html"
            "datasets","iris","Edgar Anderson's Iris Data",150,5,0,0,1,0,4,"https://vincentarelbundock.github.io/Rdatasets/csv/datasets/iris.csv","https://vincentarelbundock.github.io/Rdatasets/doc/datasets/iris.html"
            """;

        IReadOnlyList<RDatasetInfo> datasets = RDatasetsSource.ParseIndex(csv);

        Assert.Equal(2, datasets.Count);
        RDatasetInfo affairs = datasets[0];
        Assert.Equal("AER", affairs.Package);
        Assert.Equal("Affairs", affairs.Item);
        Assert.Equal("Fair's Extramarital Affairs Data", affairs.Title);
        Assert.Equal(601, affairs.Rows);
        Assert.Equal(9, affairs.Cols);
        Assert.EndsWith("/csv/AER/Affairs.csv", affairs.CsvUrl);
        Assert.EndsWith("/doc/AER/Affairs.html", affairs.DocUrl);
        Assert.Equal("iris", datasets[1].Item);
    }

    [Fact]
    public void RDatasets_DocUrlFor_BuildsMirrorPath()
        => Assert.Equal(
            "https://vincentarelbundock.github.io/Rdatasets/doc/datasets/iris.html",
            RDatasetsSource.DocUrlFor("datasets", "iris"));

    [Fact]
    public async Task RDatasets_ConstructionAndValidation()
    {
        using var source = new RDatasetsSource();
        await Assert.ThrowsAsync<ArgumentException>(() => source.GetCsvAsync("", "iris"));
        await Assert.ThrowsAsync<ArgumentException>(() => source.GetCsvAsync("datasets", ""));
        Assert.Throws<ArgumentException>(() => RDatasetsSource.ParseIndex(""));
        Assert.Throws<ArgumentException>(() => RDatasetsSource.DocUrlFor("datasets", ""));
    }

    // ── Python: seaborn-data ─────────────────────────────────────────────────

    [Fact]
    public void Seaborn_ParseIndex_KeepsOnlyCsvBlobs()
    {
        const string json = """
            { "sha": "abc", "tree": [
                { "path": "README.md", "type": "blob" },
                { "path": "tips.csv", "type": "blob" },
                { "path": "iris.csv", "type": "blob" },
                { "path": "raw", "type": "tree" } ] }
            """;

        IReadOnlyList<string> names = SeabornSource.ParseIndex(json);

        Assert.Equal(["iris", "tips"], names); // sorted, .csv stripped, README & subtree excluded
    }

    [Fact]
    public void Seaborn_WellKnownDatasets_ContainTheClassics()
    {
        Assert.Contains("iris", SeabornSource.WellKnownDatasets);
        Assert.Contains("penguins", SeabornSource.WellKnownDatasets);
        Assert.Contains("tips", SeabornSource.WellKnownDatasets);
        Assert.Contains("titanic", SeabornSource.WellKnownDatasets);
    }

    [Fact]
    public async Task Seaborn_ConstructionAndValidation()
    {
        using var source = new SeabornSource();
        await Assert.ThrowsAsync<ArgumentException>(() => source.GetCsvAsync(""));
        Assert.Throws<ArgumentException>(() => SeabornSource.ParseIndex(""));
    }

    // ── Vega: vega-datasets ──────────────────────────────────────────────────

    [Fact]
    public void Vega_ParseIndex_KeepsOnlyCsvUnderDataFolder()
    {
        const string json = """
            { "tree": [
                { "path": "data/cars.csv", "type": "blob" },
                { "path": "data/movies.csv", "type": "blob" },
                { "path": "data/airports.json", "type": "blob" },
                { "path": "README.md", "type": "blob" },
                { "path": "data", "type": "tree" } ] }
            """;

        IReadOnlyList<string> names = VegaDatasetsSource.ParseIndex(json);

        Assert.Equal(["cars", "movies"], names); // sorted, data/ stripped, .json & non-data excluded
    }

    [Fact]
    public void Vega_WellKnownDatasets_ContainTheClassics()
    {
        Assert.Contains("cars", VegaDatasetsSource.WellKnownDatasets);
        Assert.Contains("seattle-weather", VegaDatasetsSource.WellKnownDatasets);
    }

    [Fact]
    public async Task Vega_ConstructionAndValidation()
    {
        using var source = new VegaDatasetsSource();
        await Assert.ThrowsAsync<ArgumentException>(() => source.GetCsvAsync(""));
        Assert.Throws<ArgumentException>(() => VegaDatasetsSource.ParseIndex(""));
    }

    // ── Plotly: plotly/datasets ──────────────────────────────────────────────

    [Fact]
    public void Plotly_ParseIndex_KeepsRootCsvBlobs()
    {
        const string json = """
            { "tree": [
                { "path": "iris.csv", "type": "blob" },
                { "path": "tips.csv", "type": "blob" },
                { "path": "README.md", "type": "blob" },
                { "path": "api", "type": "tree" } ] }
            """;

        IReadOnlyList<string> names = PlotlyDatasetsSource.ParseIndex(json);

        Assert.Equal(["iris", "tips"], names);
    }

    [Fact]
    public void Plotly_WellKnownDatasets_ContainTheClassics()
    {
        Assert.Contains("iris", PlotlyDatasetsSource.WellKnownDatasets);
        Assert.Contains("tips", PlotlyDatasetsSource.WellKnownDatasets);
    }

    [Fact]
    public async Task Plotly_ConstructionAndValidation()
    {
        using var source = new PlotlyDatasetsSource();
        await Assert.ThrowsAsync<ArgumentException>(() => source.GetCsvAsync(""));
        Assert.Throws<ArgumentException>(() => PlotlyDatasetsSource.ParseIndex(""));
    }

    // ── OpenML: REST API ─────────────────────────────────────────────────────

    [Fact]
    public void OpenML_ParseIndex_ReadsDidAndName()
    {
        const string json = """
            { "data": { "dataset": [
                { "did": 61, "name": "iris", "status": "active" },
                { "did": 31, "name": "credit-g", "status": "active" } ] } }
            """;

        IReadOnlyList<OpenMLDatasetInfo> datasets = OpenMLSource.ParseIndex(json);

        Assert.Equal(2, datasets.Count);
        Assert.Equal(61, datasets[0].Id);
        Assert.Equal("iris", datasets[0].Name);
        Assert.Equal("active", datasets[0].Status);
        Assert.Equal("credit-g", datasets[1].Name);
    }

    [Fact]
    public void OpenML_WellKnownDatasets_IncludeIrisById()
        => Assert.Contains(OpenMLSource.WellKnownDatasets, d => d.Id == 61 && d.Name == "iris");

    [Fact]
    public async Task OpenML_Validation()
    {
        using var source = new OpenMLSource();
        await Assert.ThrowsAsync<ArgumentOutOfRangeException>(() => source.GetCsvAsync(0));
        await Assert.ThrowsAsync<ArgumentOutOfRangeException>(() => source.ListDatasetsAsync(0));
        Assert.Throws<ArgumentException>(() => OpenMLSource.ParseIndex(""));
    }
}
