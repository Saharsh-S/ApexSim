const $ = (id) => document.getElementById(id);
const pct = (x) => `${(Number(x) * 100).toFixed(1)}%`;
const seconds = (x) => `${Number(x).toFixed(2)}s`;
const titleCase = (s) => String(s || '').replace(/\b\w/g, c => c.toUpperCase());

async function api(path, options = {}) {
  const response = await fetch(path, { ...options, headers: { 'Content-Type': 'application/json', ...(options.headers || {}) } });
  const text = await response.text();
  let data = {};
  try { data = text ? JSON.parse(text) : {}; } catch { throw new Error(`Server returned invalid JSON (${response.status})`); }
  if (!response.ok) throw new Error(data.detail || `Request failed (${response.status})`);
  return data;
}

function setStatus(mode, label) {
  const pill = $('statusPill');
  pill.classList.remove('online', 'offline');
  pill.classList.add(mode === 'online' ? 'online' : 'offline');
  pill.lastElementChild.textContent = label;
}

function toast(message) {
  const el = $('toast'); el.textContent = message; el.classList.add('show');
  clearTimeout(window.__toastTimer); window.__toastTimer = setTimeout(() => el.classList.remove('show'), 2800);
}

function buttonBusy(button, busy, text) {
  button.disabled = busy;
  button.classList.toggle('is-loading', busy);
  button.querySelector('span').textContent = busy ? text : button.dataset.defaultText;
}

async function boot() {
  try {
    const [health, system] = await Promise.all([api('/api/health'), api('/api/system')]);
    setStatus('online', system.cuda ? 'GPU ONLINE' : 'CPU FALLBACK');
    $('heroRecommendation').textContent = system.cuda ? 'GPU PATH AVAILABLE' : 'CPU PATH READY';
    loadBenchmarks();
  } catch (error) {
    setStatus('offline', 'OFFLINE');
    toast('Backend is not running. Start python scripts/run_web.py');
  }
}

$('rain').addEventListener('input', () => { $('rainOut').textContent = `${$('rain').value}%`; $('heroRain').textContent = `${$('rain').value}%`; });
$('sc').addEventListener('input', () => { $('scOut').textContent = `${$('sc').value}%`; });
$('lap').addEventListener('input', () => { $('heroLap').textContent = $('lap').value; });
$('tire').addEventListener('change', () => { $('heroTire').textContent = $('tire').value.toUpperCase(); });

$('strategyForm').addEventListener('submit', async (event) => {
  event.preventDefault();
  const box = $('strategyResult'), button = $('strategyBtn');
  button.dataset.defaultText = 'Evaluate strategy'; buttonBusy(button, true, 'Evaluating…');
  box.className = 'card result-card';
  box.innerHTML = '<div class="result-placeholder"><div class="placeholder-ring">◎</div><div><span>EVALUATING</span><p>Generating candidate futures and ranking the strongest continuations…</p></div></div>';
  try {
    const data = await api('/api/strategy', { method: 'POST', body: JSON.stringify({
      lap: +$('lap').value, position: +$('position').value, tire: $('tire').value, tire_age: +$('tireAge').value,
      gap_ahead: +$('gapAhead').value, gap_behind: +$('gapBehind').value, rain_probability: +$('rain').value / 100,
      safety_car_probability: +$('sc').value / 100, candidate_count: +$('candidates').value,
      backend: $('backend').value, simulations_per_candidate: 4
    }) });
    const x = data.recommendation;
    $('heroRecommendation').textContent = x.strategy.toUpperCase();
    box.innerHTML = `<div class="result-head"><div><div class="eyebrow">RECOMMENDED STRATEGY · ${data.backend.toUpperCase()}</div><h3>${x.strategy}</h3></div><span class="confidence-badge">${pct(x.confidence)} CONFIDENCE</span></div>
      <div class="result-metrics"><div><small>EXPECTED FINISH</small><strong>P${Number(x.expectedFinish).toFixed(1)}</strong></div><div><small>WIN PROBABILITY</small><strong>${pct(x.winProbability)}</strong></div><div><small>PODIUM PROBABILITY</small><strong>${pct(x.podiumProbability)}</strong></div><div><small>DECISION LATENCY</small><strong>${Number(data.timing.total_ms).toFixed(1)}ms</strong></div></div>
      <div class="alternatives"><div class="alternatives-title">ALTERNATIVES · ${data.candidatesEvaluated} CANDIDATES EVALUATED</div>${data.alternatives.map(a => `<div class="alt-row"><span>#${a.rank} · ${a.strategy}</span><b>P${Number(a.expectedFinish).toFixed(1)} · ${pct(a.winProbability)}</b></div>`).join('')}</div>`;
  } catch (error) {
    box.innerHTML = `<div class="error-box"><strong>Could not evaluate strategy</strong><p>${error.message}</p></div>`;
    toast(error.message);
  } finally { buttonBusy(button, false, 'Evaluate strategy'); }
});

$('simForm').addEventListener('submit', async (event) => {
  event.preventDefault();
  const box = $('simResult'), button = $('simBtn'); button.dataset.defaultText = 'Run Monte Carlo'; buttonBusy(button, true, 'Simulating…');
  box.className = 'card sim-result'; box.innerHTML = '<div class="empty-inline">Running Monte Carlo simulations…</div>';
  try {
    const data = await api('/api/simulate', { method: 'POST', body: JSON.stringify({ tire: $('simTire').value, pit_lap: +$('pitLap').value, second_tire: $('secondTire').value, simulations: +$('sims').value, seed: 42 }) });
    box.innerHTML = `<div class="sim-metrics"><div><small>WIN PROBABILITY</small><strong>${pct(data.winProbability)}</strong></div><div><small>PODIUM</small><strong>${pct(data.podiumProbability)}</strong></div><div><small>MEAN RACE TIME</small><strong>${seconds(data.meanRaceTime)}</strong></div><div><small>DNF</small><strong>${pct(data.dnfProbability)}</strong></div></div>`;
  } catch (error) { box.innerHTML = `<div class="error-box"><strong>Simulation failed</strong><p>${error.message}</p></div>`; toast(error.message); }
  finally { buttonBusy(button, false, 'Run Monte Carlo'); }
});

function renderBenchmarks(data) {
  const grid = $('performanceGrid');
  const rows = Array.isArray(data.final) ? data.final : (data.final?.rows || []);
  if (!rows.length) { grid.innerHTML = '<div class="loading-card">No benchmark artifacts found. Run scripts/run_all_benchmarks.py to generate them.</div>'; return; }
  grid.innerHTML = rows.slice(0, 8).map(row => {
    const name = titleCase(String(row.implementation || row.system || row.name || 'Benchmark').replaceAll('_', ' '));
    const measured = row.status === 'measured' && row.throughput_sims_s != null;
    const value = measured ? `${Number(row.throughput_sims_s).toLocaleString(undefined, { maximumFractionDigits: 0 })} /s` : 'NOT RUN';
    const note = measured ? `${Number(row.runtime_s).toFixed(4)}s runtime` : 'Hardware or dependency unavailable';
    return `<div class="perf-card ${measured ? '' : 'not-run'}"><small>${name}</small><strong>${value}</strong><span>${note}</span></div>`;
  }).join('');
}

function renderScaling(data) {
  const rows = data.distributed?.rows || [];
  const note = data.distributed?.cuda_status || 'Distributed study';
  $('scalingNote').textContent = note;
  if (!rows.length) { $('scalingBars').innerHTML = '<div class="scale-label">No distributed measurements available.</div>'; return; }
  const max = Math.max(...rows.map(r => Number(r.throughput || 0)), 1);
  $('scalingBars').innerHTML = rows.map(r => `<div class="scale-col"><span class="scale-value">${Number(r.throughput).toFixed(0)}/s</span><div class="scale-track"><div class="scale-fill" style="height:${Math.max(4, Number(r.throughput) / max * 100)}%"></div></div><span class="scale-label">${r.workers} GPU${r.workers === 1 ? '' : 's'} · ${(Number(r.scaling_efficiency) * 100).toFixed(0)}% eff.</span></div>`).join('');
}

async function loadBenchmarks() {
  try { const data = await api('/api/benchmarks'); renderBenchmarks(data); renderScaling(data); }
  catch { $('performanceGrid').innerHTML = '<div class="loading-card">Benchmark data unavailable. The core application still works without benchmark artifacts.</div>'; }
}

document.querySelectorAll('.btn').forEach(button => { if (!button.dataset.defaultText) button.dataset.defaultText = button.querySelector('span')?.textContent || ''; });
boot();
