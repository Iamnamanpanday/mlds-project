const state = { report: null };

const number = (value) => Number.isFinite(Number(value)) ? Number(value) : 0;
const formatPct = (value) => `${number(value).toFixed(2)}%`;
const formatMs = (value) => `${number(value).toFixed(4)} ms`;
const formatBytes = (value) => `${(number(value) / 1024).toFixed(2)} KiB`;

const metricDefinitions = [
  { label: 'Accuracy', comparisonKey: 'accuracy', format: formatPct, higherIsBetter: true },
  { label: 'Weighted precision', comparisonKey: 'weightedPrecision', format: formatPct, higherIsBetter: true },
  { label: 'Weighted recall', comparisonKey: 'weightedRecall', format: formatPct, higherIsBetter: true },
  { label: 'Weighted F1', comparisonKey: 'weightedF1', format: formatPct, higherIsBetter: true },
  { label: 'Inference time', comparisonKey: 'latency', format: formatMs, higherIsBetter: false },
  { label: 'PCA matrix size', comparisonKey: 'projection', format: formatBytes, higherIsBetter: false }
];

const setStatus = (message, failed = false) => {
  const status = document.getElementById('report-status');
  status.textContent = message;
  status.classList.toggle('failed', failed);
};

const renderReport = () => {
  const features = state.report.distinctiveFeatures || [];
  const rows = state.report.comparisons.map((comparison) => {
    const wins = metricDefinitions.flatMap((metric) => {
      const result = comparison[metric.comparisonKey];
      if (!result) return [];
      const ours = number(result.ourMethod);
      const baseline = number(result.baseline);
      const isStrictWin = metric.higherIsBetter ? ours > baseline : ours < baseline;
      if (!isStrictWin) return [];
      return [`<span class="win-item"><b>✓</b><span><strong>${metric.label}</strong><small>${metric.format(ours)} / ${metric.format(baseline)}</small></span></span>`];
    });
    const featureList = features.map((feature) => `<span class="feature-chip"><b>✓</b>${feature}</span>`).join('');
    const winList = wins.join('') || '<span class="no-wins">No strict measured wins on this baseline.</span>';
    return `<tr><td><strong class="baseline-name">${comparison.name}</strong></td><td><div class="feature-list">${featureList}</div></td><td><div class="win-list">${winList}</div></td></tr>`;
  });

  document.getElementById('model-rows').innerHTML = rows.join('');
  const dataset = state.report.dataset;
  document.getElementById('report-details').textContent = `${dataset.testingFile} · ${dataset.testingSamples.toLocaleString()} held-out samples · only strict wins are listed`;
  document.getElementById('generated-at').textContent = `GENERATED ${state.report.generatedAt}`;
  setStatus(`${rows.length} baselines compared`);
};

const loadReport = async () => {
  setStatus('Loading report');
  try {
    const response = await fetch(`comparison.json?cache=${Date.now()}`, { cache: 'no-store' });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    state.report = await response.json();
    renderReport();
  } catch (error) {
    setStatus('Report unavailable', true);
    document.getElementById('model-rows').innerHTML = '<tr><td colspan="3" class="loading-cell">Run python src/classification_comparison.py, then refresh.</td></tr>';
    console.error('Could not load classification comparison:', error);
  }
};

document.getElementById('refresh-report').addEventListener('click', loadReport);

loadReport();
