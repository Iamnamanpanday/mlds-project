const METHOD_KEYS = ['float', 'naive', 'proposed'];
const METHOD_LABELS = {
  float: 'Float PCA',
  naive: 'Naive INT8',
  proposed: 'Proposed INT8'
};

const STATE = {
  benchmark: null,
  featureReport: null,
  selectedReport: 'proposed'
};

const safeNumber = (value, fallback = 0) => {
  const number = Number(value);
  return Number.isFinite(number) ? number : fallback;
};

const formatPct = (value) => `${safeNumber(value, 0).toFixed(2)}%`;
const formatMs = (value) => `${safeNumber(value, 0).toFixed(2)} ms`;
const formatKiB = (value) => `${safeNumber(value, 0).toFixed(2)} KiB`;
const formatBytes = (bytes) => {
  const size = safeNumber(bytes, 0);
  if (size >= 1024 * 1024) return `${(size / (1024 * 1024)).toFixed(2)} MB`;
  if (size >= 1024) return `${(size / 1024).toFixed(2)} KiB`;
  return `${size} B`;
};

const getMethodEntry = (key) => STATE.benchmark?.methods?.[key] || null;

const setStatus = (text, isActive = true) => {
  const runStatus = document.getElementById('run-status');
  if (!runStatus) return;
  runStatus.innerHTML = `<i></i> ${text}`;
  runStatus.classList.toggle('inactive', !isActive);
};

const renderOverview = () => {
  if (!STATE.benchmark) return;

  const methods = STATE.benchmark.methods;
  const methodKeys = Object.keys(methods);
  const bestMethodKey = methodKeys.reduce((best, key) =>
    safeNumber(methods[key].accuracyPct) > safeNumber(methods[best].accuracyPct) ? key : best,
    methodKeys[0]
  );

  const bestMethod = methods[bestMethodKey];

  document.getElementById('best-method').textContent = bestMethod.name;
  document.getElementById('best-accuracy').textContent = formatPct(bestMethod.accuracyPct);
  document.getElementById('best-latency').textContent = formatMs(bestMethod.latencyMs.mean_ms);
  document.getElementById('best-summary').textContent = `${bestMethod.name} maintains the strongest classification quality at ${formatPct(bestMethod.accuracyPct)} accuracy.`;

  const topAccuracy = Math.max(...methodKeys.map((key) => safeNumber(methods[key].accuracyPct)));
  const fastest = methodKeys.reduce((best, key) =>
    safeNumber(methods[key].latencyMs.mean_ms) < safeNumber(methods[best].latencyMs.mean_ms) ? key : best,
    methodKeys[0]
  );

  document.getElementById('top-accuracy').innerHTML = `${topAccuracy.toFixed(2)}<span>%</span>`;
  document.getElementById('top-accuracy-note').textContent = `${METHOD_LABELS[bestMethodKey]} leads the benchmark`;

  const fastestMethod = methods[fastest];
  document.getElementById('fastest-latency').innerHTML = `${safeNumber(fastestMethod.latencyMs.mean_ms).toFixed(2)}<span> ms</span>`;
  document.getElementById('fastest-method').textContent = `${fastestMethod.name} project time`;

  const floatMemory = safeNumber(methods.float.memoryKiB, 0);
  const smallestMemory = Math.min(...methodKeys.map((key) => safeNumber(methods[key].memoryKiB, 0)));
  document.getElementById('storage-saved').innerHTML = `${smallestMemory.toFixed(2)}<span> KiB</span>`;
  document.getElementById('storage-percent').textContent = `${Math.round((1 - smallestMemory / floatMemory) * 100)}%`;

  document.getElementById('component-count').textContent = safeNumber(STATE.benchmark.dataset.components, 0).toString();
  document.getElementById('feature-count').textContent = `${safeNumber(STATE.benchmark.dataset.features, 0)} input features`;
  document.getElementById('variance-tag').textContent = `${safeNumber(STATE.benchmark.dataset.explainedVariancePct, 0).toFixed(1)}%`;

  const latencyValues = methodKeys.map((key) => safeNumber(methods[key].latencyMs.mean_ms, 0));
  const minLatency = Math.min(...latencyValues);
  const maxLatency = Math.max(...latencyValues);
  document.getElementById('latency-min').textContent = minLatency.toFixed(2);
  document.getElementById('latency-max').textContent = maxLatency.toFixed(2);

  const tradeoffStorage = `${Math.round(safeNumber(methods.proposed.storageReductionPct, 0))}%`;
  const accuracyDelta = `${(safeNumber(methods.proposed.accuracyPct, 0) - safeNumber(methods.float.accuracyPct, 0)).toFixed(2)} pts`;
  document.getElementById('tradeoff-storage').textContent = tradeoffStorage;
  document.getElementById('tradeoff-accuracy').textContent = accuracyDelta;
  document.getElementById('tradeoff-summary').textContent = `${METHOD_LABELS.proposed} preserves the baseline accuracy while shrinking the matrix footprint by ${tradeoffStorage}.`;

  const comparisonBody = document.getElementById('comparison-body');
  comparisonBody.innerHTML = methodKeys.map((key) => {
    const entry = methods[key];
    const reduction = key === 'float' ? '—' : `${safeNumber(entry.storageReductionPct, 0).toFixed(0)}%`;
    const latency = safeNumber(entry.latencyMs.mean_ms, 0).toFixed(2);
    const memory = `${safeNumber(entry.memoryKiB, 0).toFixed(2)} KiB`;
    const approx = `${safeNumber(entry.approximationErrorPct, 0).toFixed(2)}%`;
    return `
      <tr>
        <td>${entry.name}</td>
        <td>${formatPct(entry.accuracyPct)}</td>
        <td>${latency} ms</td>
        <td>${memory}</td>
        <td>${approx}</td>
        <td>${reduction}</td>
      </tr>
    `;
  }).join('');

  const chartMax = Math.max(...latencyValues) || 1;
  const rows = document.querySelectorAll('.bar-row');
  rows.forEach((row) => {
    const methodKey = row.dataset.method;
    const entry = methods[methodKey];
    if (!entry) return;
    const fill = row.querySelector('.bar-fill');
    const value = safeNumber(entry.latencyMs.mean_ms, 0);
    const width = Math.max((value / chartMax) * 100, 12);
    fill.style.width = `${width}%`;
    row.querySelector('strong').textContent = `${value.toFixed(2)} ms`;
  });
};

const renderNovelty = () => {
  const proposed = getMethodEntry('proposed');
  const baseline = getMethodEntry('float');
  if (!proposed || !baseline) return;

  const accuracyDelta = safeNumber(proposed.accuracyPct) - safeNumber(baseline.accuracyPct);
  const storageReduction = safeNumber(proposed.storageReductionPct);
  document.getElementById('novelty-accuracy').textContent = formatPct(proposed.accuracyPct);
  document.getElementById('novelty-accuracy-delta').textContent = `${accuracyDelta >= 0 ? '+' : ''}${accuracyDelta.toFixed(2)} pts`;
  document.getElementById('novelty-int8-storage').textContent = formatKiB(proposed.memoryKiB);
  document.getElementById('novelty-float-storage').textContent = formatKiB(baseline.memoryKiB);
  document.getElementById('novelty-scale').textContent = safeNumber(STATE.benchmark.configuration.quantizationScale).toString();
  document.getElementById('novelty-result-note').textContent = `${storageReduction.toFixed(0)}% less matrix storage than Float PCA, with ${accuracyDelta >= 0 ? 'no measured accuracy loss' : `${Math.abs(accuracyDelta).toFixed(2)} points lower accuracy`} on this test set.`;
};

const renderReport = () => {
  const reportKey = STATE.selectedReport;
  const method = getMethodEntry(reportKey === 'baseline' ? 'float' : reportKey);
  if (!method) return;

  const classificationReport = method.classificationReport || {};
  const accuracy = safeNumber(method.accuracyPct, 0);
  const reportTitle = document.getElementById('report-title');
  const reportScore = document.getElementById('report-score');
  const precision = document.getElementById('precision');
  const recall = document.getElementById('recall');
  const f1 = document.getElementById('f1');
  const matrixTitle = document.getElementById('matrix-title');
  const matrixImage = document.getElementById('matrix-image');
  const matrixLink = document.getElementById('matrix-link');

  reportTitle.textContent = method.name;
  reportScore.innerHTML = `${accuracy.toFixed(2)}<span>%</span>`;
  precision.textContent = classificationReport['weighted avg']?.precision ? safeNumber(classificationReport['weighted avg'].precision).toFixed(3) : '—';
  recall.textContent = classificationReport['weighted avg']?.recall ? safeNumber(classificationReport['weighted avg'].recall).toFixed(3) : '—';
  f1.textContent = classificationReport['weighted avg']?.['f1-score'] ? safeNumber(classificationReport['weighted avg']['f1-score']).toFixed(3) : '—';

  const imageFile = method.confusionMatrixImage || 'baseline_confusion_matrix.png';
  const imagePath = `../results/${imageFile}`;
  matrixTitle.textContent = `${method.name} confusion matrix`;
  matrixImage.src = imagePath;
  matrixImage.alt = `${method.name} confusion matrix`;
  matrixLink.href = imagePath;
  matrixLink.textContent = `Open image ↗`;

  const classReportBody = document.getElementById('class-report-body');
  const classKeys = Object.keys(classificationReport).filter((key) => key !== 'accuracy' && key !== 'macro avg' && key !== 'weighted avg');

  classReportBody.innerHTML = classKeys.map((key) => {
    const entry = classificationReport[key];
    return `
      <tr>
        <td>${key}</td>
        <td>${safeNumber(entry.precision, 0).toFixed(3)}</td>
        <td>${safeNumber(entry.recall, 0).toFixed(3)}</td>
        <td>${safeNumber(entry['f1-score'], 0).toFixed(3)}</td>
        <td>${safeNumber(entry.support, 0).toFixed(0)}</td>
      </tr>
    `;
  }).join('');

  const confusionTable = document.getElementById('confusion-data-table');
  const matrix = method.confusionMatrix || [];
  const matrixHeader = `<thead><tr><th></th>${Array.from({ length: matrix[0]?.length || 0 }, (_, i) => `<th>${i}</th>`).join('')}</tr></thead>`;
  const matrixRows = matrix.map((row, rowIndex) => {
    const cells = row.map((value) => `<td>${value}</td>`).join('');
    return `<tr><th>${rowIndex}</th>${cells}</tr>`;
  }).join('');
  confusionTable.innerHTML = `${matrixHeader}<tbody>${matrixRows}</tbody>`;

  const selectedButtons = document.querySelectorAll('[data-report]');
  selectedButtons.forEach((button) => {
    const isActive = button.dataset.report === reportKey;
    button.classList.toggle('active', isActive);
  });
};

const renderArtifacts = () => {
  const artifacts = STATE.benchmark.artifacts || [];
  const grid = document.getElementById('output-grid');
  grid.innerHTML = artifacts.map((artifact) => {
    const size = formatBytes(artifact.sizeBytes);
    const badge = artifact.extension.toUpperCase();
    return `
      <div class="output-card">
        <div class="output-badge">${badge}</div>
        <strong>${artifact.filename}</strong>
        <span>${size}</span>
      </div>
    `;
  }).join('');

  document.getElementById('artifact-count').textContent = `${artifacts.length} generated files`;
  document.getElementById('generated-time').textContent = `Generated ${STATE.benchmark.generatedAt}`;
  document.getElementById('dataset-note').textContent = STATE.benchmark.dataset.trainingFile;
  document.getElementById('repeat-note').textContent = `${STATE.benchmark.configuration.measuredRuns} runs`;
  document.getElementById('dimensions-note').textContent = `${STATE.benchmark.dataset.features} × ${STATE.benchmark.dataset.components}`;
  document.getElementById('scale-note').textContent = `${STATE.benchmark.configuration.quantizationScale}`;
  document.getElementById('classes-note').textContent = `${STATE.benchmark.dataset.classes.length} classes`;
  document.getElementById('footer-generated').textContent = `Generated: ${STATE.benchmark.generatedAt}`;
};

const renderFeatureReport = () => {
  const report = STATE.featureReport;
  if (!report) return;

  document.getElementById('feature-report-date').textContent = `Generated ${report.generatedAt}`;
  document.getElementById('feature-report-method').textContent = report.methodology.projectionStorage;
  const proposedModel = report.models?.find((model) => model.name.startsWith('Proposed INT8'));
  const proposedRow = proposedModel ? `
    <tr class="our-method-row">
      <td><span class="feature-model-cell"><strong>Proposed INT8</strong><span class="our-method-tag">OUR METHOD</span></span></td>
      <td><span class="feature-own-value">${safeNumber(proposedModel.accuracyPct).toFixed(2)}%</span></td>
      <td><span class="feature-own-value">${safeNumber(proposedModel.latencyMsPerSample).toFixed(4)} ms</span></td>
      <td><span class="feature-own-value">${formatBytes(proposedModel.projectionBytes)}</span></td>
    </tr>
  ` : '';
  const comparisonRows = report.comparisons.map((comparison) => {
    const resultCell = (result, format) => {
      const wins = result.proposedWins;
      const symbol = wins ? '&#10003;' : '&#10007;';
      const value = `${format(result.proposed)} vs ${format(result.baseline)}`;
      return `<td><span class="feature-result ${wins ? 'is-win' : 'is-loss'}" aria-label="${wins ? 'Proposed model ties or wins' : 'Baseline wins'}"><b>${symbol}</b>${value}</span></td>`;
    };
    return `
      <tr>
        <td>${comparison.name}</td>
        ${resultCell(comparison.accuracy, (value) => `${safeNumber(value).toFixed(2)}%`)}
        ${resultCell(comparison.latency, (value) => `${safeNumber(value).toFixed(4)} ms`)}
        ${resultCell(comparison.projection, (value) => formatBytes(value))}
      </tr>
    `;
  }).join('');
  document.getElementById('feature-report-body').innerHTML = `${proposedRow}${comparisonRows}`;
};

const renderBenchmark = () => {
  renderOverview();
  renderNovelty();
  renderReport();
  renderArtifacts();
  renderFeatureReport();
  setStatus('Benchmark loaded', true);
};

const bindControls = () => {
  document.querySelectorAll('[data-report]').forEach((button) => {
    button.addEventListener('click', () => {
      STATE.selectedReport = button.dataset.report;
      renderReport();
    });
  });

  document.getElementById('btn-export-json')?.addEventListener('click', () => {
    const data = JSON.stringify(STATE.benchmark, null, 2);
    const blob = new Blob([data], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement('a');
    anchor.href = url;
    anchor.download = 'benchmark_summary.json';
    anchor.click();
    URL.revokeObjectURL(url);
  });

  document.getElementById('btn-print')?.addEventListener('click', () => {
    window.print();
  });

  const optionsBtn = document.getElementById('options-btn');
  const dropdown = document.getElementById('options-dropdown');

  optionsBtn?.addEventListener('click', (event) => {
    event.stopPropagation();
    const isVisible = dropdown.style.display !== 'none';
    dropdown.style.display = isVisible ? 'none' : 'block';
  });

  document.addEventListener('click', (event) => {
    if (!event.target.closest('#options-btn') && !event.target.closest('#options-dropdown')) {
      dropdown.style.display = 'none';
    }
  });
};

const loadBenchmark = async () => {
  try {
    setStatus('Loading benchmark...', true);
    const response = await fetch('../results/benchmark_summary.json', { cache: 'no-store' });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    STATE.benchmark = await response.json();
    renderBenchmark();
  } catch (error) {
    setStatus('Benchmark unavailable', false);
    console.error('Failed to load benchmark data:', error);
  }
};

const loadFeatureReport = async () => {
  try {
    const response = await fetch('../results/feature_comparison_report.json', { cache: 'no-store' });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    STATE.featureReport = await response.json();
    renderFeatureReport();
  } catch (error) {
    const body = document.getElementById('feature-report-body');
    if (body) body.innerHTML = '<tr><td colspan="4">Run src/feature_comparison_report.py to generate the measured report.</td></tr>';
    console.error('Failed to load feature comparison report:', error);
  }
};

document.addEventListener('DOMContentLoaded', () => {
  bindControls();
  loadBenchmark();
  loadFeatureReport();
});
