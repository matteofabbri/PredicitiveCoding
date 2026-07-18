using System.Text.Json;

namespace PredictiveCoding.Brain.Viz;

// ─────────────────────────────────────────────────────────────────────────────
// HtmlVisualizer  —  renders a VizData as one fully self-contained HTML page.
//
// The page is a single local file: all CSS and JavaScript are inline, the data
// is embedded as JSON, and there are NO external resources of any kind (no CDN,
// no fonts, no <script src>, no <link>). The title and every other data value
// reach the page only through the JSON embedding, whose serializer escapes '<'
// — so hostile strings cannot break out of the <script> block.
//
// Four sections, each rendered only when its VizData slot is populated:
//   1. Live episode  — animated player: the target's landmark cloud on a canvas
//                      and, per step, the single column's hypothesis set
//                      narrowing versus the voting population converging.
//   2. Benchmark     — accuracy / mean-steps table, single vs voting.
//   3. Pose invariance — per-object ✓/✗ of the absolute column vs the
//                      displacement recogniser after a rigid move.
//   4. Real-sensor scan — the actual image on a canvas with the patch sensor
//                      animating over its fixation grid, plus the descriptor
//                      (dominant orientation / share / luminance) read-outs.
// ─────────────────────────────────────────────────────────────────────────────

/// <summary>Renders <see cref="VizData"/> into a standalone HTML page.</summary>
public static class HtmlVisualizer
{
    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
    };

    /// <summary>The complete HTML document for <paramref name="data"/>.</summary>
    public static string Render(VizData data)
    {
        ArgumentNullException.ThrowIfNull(data);
        return Template
            .Replace("{{DATA_JSON}}", JsonSerializer.Serialize(data, JsonOptions))
            .Replace("{{GENERATED_UTC}}", DateTime.UtcNow.ToString("yyyy-MM-dd HH:mm 'UTC'"));
    }

    /// <summary>Renders and writes the page to <paramref name="path"/> (UTF-8).</summary>
    public static void Save(string path, VizData data)
        => File.WriteAllText(path, Render(data));

    // The JS below uses no C# interpolation — placeholders are plain-text markers.
    private const string Template = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Thousand Brains</title>
<style>
  :root {
    --bg: #ffffff; --card: #f6f8fa; --border: #d0d7de; --text: #1f2328;
    --dim: #656d76; --accent: #0969da; --ok: #1a7f37; --bad: #cf222e;
  }
  * { box-sizing: border-box; margin: 0; }
  body {
    background: var(--bg); color: var(--text);
    font: 15px/1.5 -apple-system, 'Segoe UI', Roboto, sans-serif;
    padding: 32px 16px 48px;
  }
  main { max-width: 1100px; margin: 0 auto; }
  header { max-width: 1100px; margin: 0 auto 28px; }
  h1 { font-size: 26px; font-weight: 650; }
  .subtitle { color: var(--dim); margin-top: 4px; }
  section {
    background: var(--card); border: 1px solid var(--border); border-radius: 10px;
    padding: 22px 24px; margin-bottom: 22px;
  }
  h2 { font-size: 17px; font-weight: 600; margin-bottom: 14px; }
  .row { display: flex; gap: 26px; flex-wrap: wrap; align-items: flex-start; }
  canvas { background: #ffffff; border: 1px solid var(--border); border-radius: 8px; }
  .track { margin-bottom: 18px; }
  .track h3 { font-size: 13px; color: var(--dim); font-weight: 600; margin-bottom: 6px; }
  .barwrap { background: #eaeef2; border-radius: 5px; height: 22px; width: 320px; overflow: hidden; }
  .bar { background: var(--accent); height: 100%; transition: width .25s; }
  .bar.locked { background: var(--ok); }
  .chips { margin-top: 6px; min-height: 24px; }
  .chip {
    display: inline-block; background: #eaeef2; border: 1px solid var(--border);
    border-radius: 999px; padding: 0 9px; margin: 2px 3px 0 0; font-size: 12px;
  }
  .chip.ok { border-color: var(--ok); color: var(--ok); }
  .status { margin-top: 4px; font-size: 13px; color: var(--dim); min-height: 20px; }
  .status.ok { color: var(--ok); }
  .controls { margin-top: 10px; display: flex; gap: 12px; align-items: center; }
  button {
    background: #f6f8fa; color: var(--text); border: 1px solid var(--border);
    border-radius: 6px; padding: 5px 14px; font-size: 14px; cursor: pointer;
  }
  button:hover { border-color: var(--accent); }
  input[type=range] { width: 240px; accent-color: var(--accent); }
  table { border-collapse: collapse; width: 100%; }
  th, td { padding: 7px 14px; text-align: left; border-bottom: 1px solid var(--border); }
  th { color: var(--dim); font-size: 13px; font-weight: 600; }
  td.best { color: var(--accent); font-weight: 600; }
  .posegrid { display: flex; gap: 6px; flex-wrap: wrap; margin: 10px 0; }
  .posecell {
    border: 1px solid var(--border); border-radius: 8px; padding: 8px 12px;
    text-align: center; font-size: 13px; min-width: 64px;
  }
  .mark { font-size: 16px; font-weight: 700; }
  .mark.ok { color: var(--ok); } .mark.bad { color: var(--bad); }
  .readout { min-width: 220px; }
  .readout .lbl { color: var(--dim); font-size: 12px; margin: 10px 0 3px; }
  .minibar { background: #eaeef2; border-radius: 4px; height: 12px; width: 200px; }
  .minibar > div { background: var(--accent); height: 100%; border-radius: 4px; }
  .note { color: var(--dim); font-size: 13px; margin-top: 12px; }
  .explain {
    color: var(--text); font-size: 14px; margin: 0 0 16px;
    padding: 10px 14px; background: #ffffff; border-left: 3px solid var(--accent);
    border-radius: 0 6px 6px 0;
  }
  .intro { max-width: 1100px; margin: 0 auto 24px; color: var(--text); font-size: 15px; }
  footer { max-width: 1100px; margin: 0 auto; color: var(--dim); font-size: 12px; }
</style>
</head>
<body>
<header><h1 id="title"></h1><div class="subtitle">Many cortical columns, each modelling objects in its own reference frame, voting to a consensus.</div></header>
<div class="intro">
  The theory (Hawkins, "A Thousand Brains"): your neocortex is made of thousands of
  near-identical <b>cortical columns</b>. Each one senses only a tiny patch of the world,
  learns objects as little maps in its own reference frame — <i>which feature sits where</i> —
  and they all continuously <b>vote</b> to agree on what they are looking at.
  This page is that system actually running, in miniature: it learned a handful of toy
  objects, and here it has to recognise one by touching it one landmark at a time.
</div>
<main id="app"></main>
<footer>PredictiveCoding.Brain — Thousand Brains demo &middot; generated {{GENERATED_UTC}}</footer>
<script>
const DATA = {{DATA_JSON}};
(function () {
  'use strict';
  document.title = DATA.title;
  document.getElementById('title').textContent = DATA.title;
  const app = document.getElementById('app');
  const PALETTE = ['#0969da', '#1a7f37', '#9a6700', '#cf222e', '#8250df', '#1b7c83', '#bc4c00', '#4d2d00'];
  const el = (tag, cls, parent) => {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (parent) parent.appendChild(e);
    return e;
  };

  // ── Shared playback clock ─────────────────────────────────────────────────
  let step = 0, playing = true, maxStep = 1;
  const tickListeners = [];
  const setStep = s => { step = s; tickListeners.forEach(f => f(step)); };

  // ── Section 1: live episode ───────────────────────────────────────────────
  if (DATA.episode && DATA.episode.singleSteps.length) {
    const ep = DATA.episode;
    maxStep = Math.max(ep.singleSteps.length, ep.votingRounds.length);
    const sec = el('section', null, app);
    el('h2', null, sec).textContent = 'Live episode — object #' + ep.targetId + ' presented to ' +
      '1 column vs ' + ep.columnsVoting + ' voting columns';
    el('div', 'explain', sec).textContent =
      'A recognition race. The dots on the left are the landmarks of the object being touched ' +
      '(the ring is the one being sensed right now). Each touch is a clue: "feature X at place Y" ' +
      'rules out every learned object that has no such landmark — so the green bar, the list of ' +
      'suspects, shrinks. Top track: one column touching one landmark per step. Bottom track: ' +
      ep.columnsVoting + ' columns touch ' + ep.columnsVoting + ' different landmarks at once and ' +
      'keep only the suspects they ALL agree on — that vote is why they finish in a fraction of the steps.';
    const row = el('div', 'row', sec);

    const canvas = el('canvas', null, row);
    canvas.width = canvas.height = 400;
    const ctx = canvas.getContext('2d');
    const target = ep.objects.find(o => o.id === ep.targetId) || ep.objects[0];

    const right = el('div', null, row);
    const mkTrack = label => {
      const t = el('div', 'track', right);
      el('h3', null, t).textContent = label;
      const w = el('div', 'barwrap', t);
      const bar = el('div', 'bar', w);
      const chips = el('div', 'chips', t);
      const status = el('div', 'status', t);
      return { bar, chips, status };
    };
    const tSingle = mkTrack('1 cortical column — one landmark per step');
    const tVoting = mkTrack(ep.columnsVoting + ' columns voting — ' + ep.columnsVoting + ' landmarks per round');

    const controls = el('div', 'controls', sec);
    const btn = el('button', null, controls);
    const slider = el('input', null, controls);
    slider.type = 'range'; slider.min = 0; slider.max = maxStep - 1; slider.value = 0;
    const lbl = el('span', 'status', controls);
    btn.onclick = () => { playing = !playing; btn.textContent = playing ? '⏸' : '▶'; };
    slider.oninput = () => { playing = false; btn.textContent = '▶'; setStep(+slider.value); };

    const drawCloud = s => {
      ctx.clearRect(0, 0, 400, 400);
      const xs = target.points.map(p => p.x), ys = target.points.map(p => p.y);
      const minX = Math.min(...xs), maxX = Math.max(...xs), minY = Math.min(...ys), maxY = Math.max(...ys);
      const span = Math.max(maxX - minX, maxY - minY, 1e-6), pad = 40;
      const sx = x => pad + (x - minX) / span * (400 - 2 * pad);
      const sy = y => pad + (y - minY) / span * (400 - 2 * pad);
      const probe = s % target.points.length;
      target.points.forEach((p, i) => {
        ctx.beginPath();
        ctx.arc(sx(p.x), sy(p.y), i === probe ? 9 : 5, 0, Math.PI * 2);
        ctx.fillStyle = PALETTE[p.featureId % PALETTE.length];
        ctx.fill();
        if (i === probe) {
          ctx.beginPath();
          ctx.arc(sx(p.x), sy(p.y), 14, 0, Math.PI * 2);
          ctx.strokeStyle = '#1f2328'; ctx.lineWidth = 2; ctx.stroke();
        }
      });
      ctx.fillStyle = '#656d76'; ctx.font = '12px sans-serif';
      ctx.fillText('landmarks of object #' + target.id + ' (colour = feature)', 12, 390);
    };

    const drawTrack = (t, seq, s, unit) => {
      const idx = Math.min(s, seq.length - 1);
      const hyp = seq[idx];
      const done = hyp.length === 1;
      t.bar.style.width = (100 * hyp.length / ep.objectCount) + '%';
      t.bar.className = done ? 'bar locked' : 'bar';
      t.chips.innerHTML = '';
      hyp.forEach(id => {
        const c = el('span', done ? 'chip ok' : 'chip', t.chips);
        c.textContent = '#' + id;
      });
      t.status.className = done ? 'status ok' : 'status';
      t.status.textContent = done
        ? 'recognised #' + hyp[0] + ' after ' + (idx + 1) + ' ' + unit
        : hyp.length + ' hypotheses after ' + (idx + 1) + ' ' + unit;
    };

    tickListeners.push(s => {
      // The shared clock also drives the (longer) sensor scan: this section
      // freezes at its own last step instead of counting past it.
      const cs = Math.min(s, maxStep - 1);
      drawCloud(cs);
      drawTrack(tSingle, ep.singleSteps, cs, 'steps');
      drawTrack(tVoting, ep.votingRounds, cs, 'rounds');
      slider.value = cs;
      lbl.textContent = 'step ' + (cs + 1) + ' / ' + maxStep;
    });
    btn.textContent = '⏸';
  }

  // ── Section 2: benchmark ──────────────────────────────────────────────────
  if (DATA.benchmark && DATA.benchmark.length) {
    const sec = el('section', null, app);
    el('h2', null, sec).textContent = 'Benchmark — accuracy and steps-to-recognise per condition';
    el('div', 'explain', sec).textContent =
      'The same race, repeated over many objects and probes, under four conditions: clean view, ' +
      'part of the object hidden (occlusion), noisy sensors, and jittered positions. ' +
      '"Acc" is how often the right object won; "steps" is how many touches it took. ' +
      'The point to notice: voting does not buy accuracy — it buys SPEED (blue = fewer steps).';
    const table = el('table', null, sec);
    table.innerHTML = '<tr><th>condition</th><th>single acc</th><th>single steps</th>' +
      '<th>voting acc</th><th>voting steps</th></tr>';
    DATA.benchmark.forEach(b => {
      const tr = el('tr', null, table);
      const cells = [
        b.condition, Math.round(b.singleAccuracy * 100) + '%', b.singleSteps.toFixed(1),
        Math.round(b.votingAccuracy * 100) + '%', b.votingSteps.toFixed(1),
      ];
      cells.forEach((v, i) => {
        const td = el('td', null, tr);
        td.textContent = v;
        if ((i === 2 && b.singleSteps < b.votingSteps) || (i === 4 && b.votingSteps <= b.singleSteps))
          td.className = 'best';
      });
    });
    el('div', 'note', sec).textContent =
      'Voting matches single-column accuracy but converges in far fewer sensorimotor steps.';
  }

  // ── Section 3: pose invariance ────────────────────────────────────────────
  if (DATA.poseInvariance && DATA.poseInvariance.rows.length) {
    const pi = DATA.poseInvariance;
    const sec = el('section', null, app);
    el('h2', null, sec).textContent = 'Pose invariance — every object rotated ' + pi.thetaDeg +
      '° and translated (' + pi.dx + ', ' + pi.dy + ')';
    el('div', 'explain', sec).textContent =
      'Every learned object was rigidly moved (rotated + shifted), then shown again. ' +
      'The recogniser that memorised absolute positions ("feature X at coordinates Y") fails: ' +
      'all the coordinates changed. The one that memorised the DISTANCES between features still ' +
      'wins: moving an object rigidly never changes its internal distances. That is what ' +
      'displacement cells buy the brain.';
    const mkRow = (label, pick) => {
      el('div', 'lbl note', sec).textContent = label;
      const grid = el('div', 'posegrid', sec);
      let ok = 0;
      pi.rows.forEach(r => {
        const cell = el('div', 'posecell', grid);
        const good = pick(r);
        if (good) ok++;
        cell.innerHTML = '#' + r.id + '<br><span class="mark ' + (good ? 'ok' : 'bad') + '">' +
          (good ? '✓' : '✗') + '</span>';
      });
      el('div', 'note', sec).textContent = ok + ' / ' + pi.rows.length + ' recognised';
    };
    mkRow('absolute cortical column (location matching)', r => r.columnCorrect);
    mkRow('displacement recogniser (pairwise-distance signature)', r => r.displacementCorrect);
    el('div', 'note', sec).textContent =
      'Absolute locations break under rigid motion; displacement structure is invariant to it.';
  }

  // ── Section 4: real-sensor scan ───────────────────────────────────────────
  if (DATA.imageScan && DATA.imageScan.fixations.length) {
    const sc = DATA.imageScan, SCALE = 4;
    const sec = el('section', null, app);
    el('h2', null, sec).textContent = 'Real-sensor scan — a patch sensor exploring an image';
    el('div', 'explain', sec).textContent =
      'The same machinery on real pixels. A small "eye" (a ' + sc.patchSize + '×' + sc.patchSize +
      ' pixel patch, the blue square) walks across an actual image. At every stop it reports what ' +
      'a column would receive: the dominant edge direction it sees (the glyph), how strongly ' +
      'oriented the patch is (edge-energy share), and its brightness. This stream of ' +
      '"feature at location" observations is exactly what the columns above consume.';
    const row = el('div', 'row', sec);

    const canvas = el('canvas', null, row);
    canvas.width = sc.width * SCALE; canvas.height = sc.height * SCALE;
    const ctx = canvas.getContext('2d');
    const off = document.createElement('canvas');
    off.width = sc.width; off.height = sc.height;
    const octx = off.getContext('2d');
    const idata = octx.createImageData(sc.width, sc.height);
    for (let i = 0; i < sc.pixels.length; i++) {
      const v = Math.round(sc.pixels[i] * 255);
      idata.data[i * 4] = idata.data[i * 4 + 1] = idata.data[i * 4 + 2] = v;
      idata.data[i * 4 + 3] = 255;
    }
    octx.putImageData(idata, 0, 0);

    const readout = el('div', 'readout', row);
    el('div', 'lbl', readout).textContent = 'dominant orientation';
    const glyph = el('canvas', null, readout);
    glyph.width = glyph.height = 64;
    const gctx = glyph.getContext('2d');
    el('div', 'lbl', readout).textContent = 'edge-energy share';
    const shareBar = el('div', null, el('div', 'minibar', readout));
    el('div', 'lbl', readout).textContent = 'mean luminance';
    const lumBar = el('div', null, el('div', 'minibar', readout));
    const fixLbl = el('div', 'status', readout);

    tickListeners.push(s => {
      const f = sc.fixations[s % sc.fixations.length];
      ctx.imageSmoothingEnabled = false;
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(off, 0, 0, canvas.width, canvas.height);
      ctx.strokeStyle = '#0969da'; ctx.lineWidth = 2;
      ctx.strokeRect(f.col * sc.stride * SCALE, f.row * sc.stride * SCALE,
        sc.patchSize * SCALE, sc.patchSize * SCALE);

      gctx.clearRect(0, 0, 64, 64);
      if (f.category >= sc.orientationBins) {
        gctx.fillStyle = '#656d76'; gctx.font = '12px sans-serif';
        gctx.fillText('flat', 20, 36);
      } else {
        // The category is the dominant GRADIENT direction; the visible edge runs
        // perpendicular to it.
        const a = (f.category + 0.5) / sc.orientationBins * Math.PI + Math.PI / 2;
        gctx.beginPath();
        gctx.moveTo(32 - 24 * Math.cos(a), 32 - 24 * Math.sin(a));
        gctx.lineTo(32 + 24 * Math.cos(a), 32 + 24 * Math.sin(a));
        gctx.strokeStyle = '#0969da'; gctx.lineWidth = 3; gctx.stroke();
      }
      shareBar.style.width = (100 * f.share) + '%';
      lumBar.style.width = (100 * f.luminance) + '%';
      fixLbl.textContent = 'fixation (' + f.col + ', ' + f.row + ')';
    });
  }

  // ── Clock ────────────────────────────────────────────────────────────────
  const totalTicks = Math.max(maxStep,
    DATA.imageScan && DATA.imageScan.fixations.length ? DATA.imageScan.fixations.length : 1);
  setStep(0);
  setInterval(() => {
    if (playing) setStep((step + 1) % totalTicks);
  }, 700);
})();
</script>
</body>
</html>
""";
}
