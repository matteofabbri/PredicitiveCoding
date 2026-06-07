using PredictiveCoding.OmniMind.Web.Components;
using PredictiveCoding.OmniMind.Web.Models;
using PredictiveCoding.OmniMind.Web.Services;

var builder = WebApplication.CreateBuilder(args);

// ── Configuration ────────────────────────────────────────────────────────────
builder.Services.Configure<OmniMindOptions>(
    builder.Configuration.GetSection(OmniMindOptions.SectionName));

// One pooled HttpClient for every live provider in a run.
builder.Services.AddHttpClient("omnimind", c =>
{
    c.Timeout = TimeSpan.FromSeconds(30);
    c.DefaultRequestHeaders.UserAgent.ParseAdd("PredictiveCoding.OmniMind/1.0 (+web)");
});

// Run-history persistence (Sperone) + the orchestration service. Both singletons.
builder.Services.AddSingleton<HistoryStore>();
builder.Services.AddSingleton<MindService>();

// Blazor Server (interactive, real-time over SignalR).
builder.Services.AddRazorComponents()
    .AddInteractiveServerComponents();

var app = builder.Build();

// Open (and recover) the history database eagerly so the first request is fast.
app.Services.GetRequiredService<HistoryStore>();

if (!app.Environment.IsDevelopment())
{
    app.UseExceptionHandler("/", createScopeForErrors: true);
    app.UseHsts();
}

app.UseStaticFiles();
app.UseAntiforgery();

app.MapRazorComponents<App>()
    .AddInteractiveServerRenderMode();

// Flush a durable Sperone checkpoint on shutdown (saves already checkpoint per run).
app.Lifetime.ApplicationStopping.Register(() =>
{
    try { app.Services.GetRequiredService<HistoryStore>().Checkpoint(); }
    catch { /* best effort on shutdown */ }
});

app.Run();
