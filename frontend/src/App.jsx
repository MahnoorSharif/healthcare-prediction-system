import { useEffect, useMemo, useState, useRef, useCallback } from 'react'

const localHosts = new Set(['localhost', '127.0.0.1', ''])
const apiHost = localHosts.has(window.location.hostname) ? '127.0.0.1' : window.location.hostname
const API_BASE = import.meta.env.VITE_API_URL || `${window.location.protocol}//${apiHost}:8000`

const featureInfo = {
  pregnancies: { label: 'Pregnancies', normal: '0–3 typical', desc: 'Number of times pregnant. More pregnancies can slightly raise diabetes risk due to gestational diabetes history.' },
  glucose: { label: 'Glucose', normal: 'Normal: 70–99 mg/dL', desc: 'Blood sugar level after fasting. Values above 126 mg/dL suggest diabetes. This is the strongest predictor.' },
  bp: { label: 'Blood Pressure', normal: 'Normal: 60–80 mmHg', desc: 'Diastolic blood pressure (lower number). High BP combined with high glucose greatly increases diabetes risk.' },
  skin: { label: 'Skin Thickness', normal: 'Normal: 10–30 mm', desc: 'Triceps skinfold thickness. Used to estimate body fat. Higher values correlate with insulin resistance.' },
  insulin: { label: 'Insulin', normal: 'Normal: 16–166 mu U/ml', desc: '2-Hour serum insulin. High levels may indicate insulin resistance, a precursor to Type 2 diabetes.' },
  bmi: { label: 'BMI', normal: 'Normal: 18.5–24.9 kg/m²', desc: 'Body Mass Index. Obesity (BMI > 30) is a major risk factor for Type 2 diabetes and heart disease.' },
  dpf: { label: 'Diabetes Pedigree', normal: 'Low risk: < 0.5', desc: 'Diabetes Pedigree Function — scores genetic influence based on family history of diabetes. Higher = more risk.' },
  age: { label: 'Age', normal: 'Risk rises after 45', desc: 'Age in years. Diabetes and heart disease risk both increase significantly with age after 45.' },
  sex: { label: 'Sex', normal: '—', desc: 'Biological sex. Men generally have higher heart disease risk at younger ages; women catch up post-menopause.' },
  cp: { label: 'Chest Pain Type', normal: 'Type 0 = typical angina', desc: 'Type 0: typical angina. Type 1: atypical angina. Type 2: non-anginal pain. Type 3: asymptomatic.' },
  trestbps: { label: 'Resting BP', normal: 'Normal: 90–120 mmHg', desc: 'Resting blood pressure on hospital admission. Above 140 mmHg is considered high and increases heart risk.' },
  chol: { label: 'Cholesterol', normal: 'Normal: < 200 mg/dL', desc: 'Serum cholesterol. Above 240 mg/dL is high. High cholesterol leads to plaque buildup in arteries.' },
  fbs: { label: 'Fasting Blood Sugar', normal: 'Normal: < 120 mg/dL', desc: 'Fasting blood sugar > 120 mg/dL = Yes (1). Elevated fasting sugar is a strong indicator of diabetes and heart risk.' },
  restecg: { label: 'Resting ECG', normal: 'Class 0 = normal', desc: 'Resting electrocardiograph results. Class 0: Normal. Class 1: ST-T wave abnormality. Class 2: Left ventricular hypertrophy.' },
  thalach: { label: 'Max Heart Rate', normal: 'Normal: 100–170 bpm', desc: 'Maximum heart rate achieved during exercise. Lower max HR in older patients can indicate heart disease.' },
  exang: { label: 'Exercise Angina', normal: 'No = lower risk', desc: 'Exercise-induced chest pain (angina). If Yes, the heart is not getting enough blood during physical activity.' },
  oldpeak: { label: 'ST Depression', normal: 'Normal: 0–1 mm', desc: 'ST depression induced by exercise relative to rest. Higher values indicate more severe heart stress.' },
  slope: { label: 'ST Slope', normal: 'Slope 2 = upsloping (good)', desc: 'Slope of the peak exercise ST segment. Slope 0: downsloping (bad). Slope 1: flat. Slope 2: upsloping (good).' },
  ca: { label: 'Major Vessels', normal: '0 vessels = normal', desc: 'Number of major vessels (0–3) colored by fluoroscopy. More blocked vessels = higher heart disease severity.' },
  thal: { label: 'Thalassemia', normal: 'Value 2 = normal', desc: 'Blood disorder type. Value 1: fixed defect. Value 2: normal blood flow. Value 3: reversible defect (most serious).' },
}

// Risk thresholds per field
// inverted: true means LOW value = danger (e.g. Max Heart Rate — too low is bad)
const riskThresholds = {
  glucose:   { warn: 100, danger: 126 },
  bp:        { warn: 80,  danger: 100 },
  bmi:       { warn: 25,  danger: 30  },
  insulin:   { warn: 166, danger: 300 },
  age:       { warn: 45,  danger: 65  },
  chol:      { warn: 200, danger: 240 },
  trestbps:  { warn: 120, danger: 140 },
  thalach:   { warn: 100, danger: 70, inverted: true },  // LOW heart rate = bad
  oldpeak:   { warn: 1,   danger: 2   },
}

const diabetesFields = [
  { label: 'Pregnancies', key: 'pregnancies', min: 0, max: 17, step: 1, value: 3, unit: 'count', normal: '0–3 typical', icon: '🤰' },
  { label: 'Glucose', key: 'glucose', min: 50, max: 200, step: 1, value: 120, unit: 'mg/dL', normal: 'Normal: 70–99', icon: '🩸' },
  { label: 'Blood Pressure', key: 'bp', min: 40, max: 130, step: 1, value: 70, unit: 'mmHg', normal: 'Normal: 60–80', icon: '💉' },
  { label: 'Skin Thickness', key: 'skin', min: 0, max: 99, step: 1, value: 20, unit: 'mm', normal: 'Normal: 10–30', icon: '📏' },
  { label: 'Insulin', key: 'insulin', min: 0, max: 846, step: 1, value: 80, unit: 'mu U/ml', normal: 'Normal: 16–166', icon: '💊' },
  { label: 'BMI', key: 'bmi', min: 15, max: 67, step: 0.1, value: 25.0, unit: 'kg/m²', normal: 'Normal: 18.5–24.9', icon: '⚖️' },
  { label: 'Pedigree Function', key: 'dpf', min: 0.0, max: 2.5, step: 0.01, value: 0.5, unit: 'score', normal: 'Low risk: < 0.5', icon: '🧬' },
  { label: 'Age', key: 'age', min: 21, max: 81, step: 1, value: 33, unit: 'years', normal: 'Risk rises after 45', icon: '🎂' },
]

const heartFields = [
  { label: 'Age', key: 'age', min: 29, max: 77, step: 1, value: 54, unit: 'years', normal: 'Risk rises after 45', icon: '🎂' },
  { label: 'Sex', key: 'sex', type: 'select', options: [{ label: 'Male', value: 1 }, { label: 'Female', value: 0 }], value: 1, icon: '👤' },
  { label: 'Chest Pain Type', key: 'cp', type: 'select', options: [0, 1, 2, 3].map((v) => ({ label: `Type ${v}`, value: v })), value: 0, icon: '🫀' },
  { label: 'Resting BP', key: 'trestbps', min: 90, max: 200, step: 1, value: 130, unit: 'mmHg', normal: 'Normal: 90–120', icon: '💉' },
  { label: 'Cholesterol', key: 'chol', min: 120, max: 570, step: 1, value: 240, unit: 'mg/dL', normal: 'Normal: < 200', icon: '🩸' },
  { label: 'Fasting BS', key: 'fbs', type: 'select', options: [{ label: 'No (< 120 mg/dL)', value: 0 }, { label: 'Yes (> 120 mg/dL)', value: 1 }], value: 0, icon: '🍬' },
  { label: 'Resting ECG', key: 'restecg', type: 'select', options: [{ label: 'Class 0 – Normal', value: 0 }, { label: 'Class 1 – ST Abnormality', value: 1 }, { label: 'Class 2 – LV Hypertrophy', value: 2 }], value: 0, icon: '📊' },
  { label: 'Max Heart Rate', key: 'thalach', min: 70, max: 202, step: 1, value: 150, unit: 'bpm', normal: 'Normal: 100–170', icon: '❤️' },
  { label: 'Exercise Angina', key: 'exang', type: 'select', options: [{ label: 'No', value: 0 }, { label: 'Yes', value: 1 }], value: 0, icon: '🏃' },
  { label: 'ST Depression', key: 'oldpeak', min: 0.0, max: 6.2, step: 0.1, value: 1.0, unit: 'mm', normal: 'Normal: 0–1', icon: '📉' },
  { label: 'Slope', key: 'slope', type: 'select', options: [{ label: 'Slope 0 – Downsloping', value: 0 }, { label: 'Slope 1 – Flat', value: 1 }, { label: 'Slope 2 – Upsloping', value: 2 }], value: 0, icon: '📐' },
  { label: 'Major Vessels', key: 'ca', type: 'select', options: [0, 1, 2, 3, 4].map((v) => ({ label: `${v} vessel${v !== 1 ? 's' : ''}`, value: v })), value: 0, icon: '🫁' },
  { label: 'Thalassemia', key: 'thal', type: 'select', options: [{ label: 'Value 0', value: 0 }, { label: 'Value 1 – Fixed Defect', value: 1 }, { label: 'Value 2 – Normal', value: 2 }, { label: 'Value 3 – Reversible Defect', value: 3 }], value: 1, icon: '🧪' },
]

const modelColors = {
  'Logistic Regression': '#2563eb',
  KNN: '#059669',
  'Decision Tree': '#d97706',
}

const views = [
  { id: 'predict', label: '🔬 Predict', desc: 'Patient Assessment' },
  { id: 'history', label: '📋 History', desc: 'Saved Records' },
  { id: 'visuals', label: '📊 Visuals', desc: 'EDA Charts' },
  { id: 'metrics', label: '📈 Metrics', desc: 'Model Eval' },
  { id: 'pca', label: '🔷 PCA', desc: 'Dimensionality' },
]

const visualGroups = [
  {
    title: 'Exploratory Data Analysis',
    intro: 'Distribution, spread, correlation, and class separation visuals generated from the notebook.',
    items: [
      { title: 'Diabetes Histograms Before Cleaning', file: 'plot_hist_diabetes_before.png' },
      { title: 'Diabetes Histograms After Cleaning', file: 'plot_hist_diabetes_after.png' },
      { title: 'Diabetes Boxplot', file: 'plot_boxplot_diabetes.png' },
      { title: 'Heart Disease Boxplot', file: 'plot_boxplot_heart.png' },
      { title: 'Diabetes Correlation Heatmap', file: 'plot_heatmap_diabetes.png' },
      { title: 'Heart Disease Correlation Heatmap', file: 'plot_heatmap_heart.png' },
      { title: 'Diabetes Violin Plots', file: 'plot_violin_diabetes.png' },
      { title: 'Heart Disease Violin Plots', file: 'plot_violin_heart.png' },
      { title: 'Diabetes Pairplot', file: 'plot_pairplot_diabetes.png', span: 'wide' },
      { title: 'Heart Disease Pairplot', file: 'plot_pairplot_heart.png', span: 'wide' },
    ],
  },
  {
    title: 'Model Evaluation',
    intro: 'Saved comparison, confusion matrix, ROC, and feature-importance charts for both datasets.',
    items: [
      { title: 'Diabetes Model Comparison', file: 'plot_comparison_diabetes.png' },
      { title: 'Heart Model Comparison', file: 'plot_comparison_heart.png' },
      { title: 'Diabetes Confusion Matrix', file: 'plot_cm_diabetes.png' },
      { title: 'Heart Confusion Matrix', file: 'plot_cm_heart.png' },
      { title: 'Diabetes ROC Curve', file: 'plot_roc_diabetes.png' },
      { title: 'Heart ROC Curve', file: 'plot_roc_heart.png' },
      { title: 'Diabetes Feature Importance', file: 'plot_feature_imp_diabetes.png' },
      { title: 'Heart Feature Importance', file: 'plot_feature_imp_heart.png' },
      { title: 'Decision Tree Visualization', file: 'plot_decision_tree.png', span: 'wide' },
    ],
  },
  {
    title: 'PCA Analysis',
    intro: 'Dimensionality reduction views showing principal component separation for both datasets.',
    items: [
      { title: 'Diabetes PCA', file: 'plot_pca_diabetes.png' },
      { title: 'Heart Disease PCA', file: 'plot_pca_heart.png' },
    ],
  },
]

function visualUrl(file) { return `${API_BASE}/visuals/${file}` }
function buildDefaultInputs(fields) {
  return fields.reduce((acc, item) => ({ ...acc, [item.key]: item.value }), {})
}
function loadHistory() {
  try { return JSON.parse(localStorage.getItem('hc_history') || '[]') } catch { return [] }
}
function saveHistory(list) {
  try { localStorage.setItem('hc_history', JSON.stringify(list.slice(0, 50))) } catch {}
}

// ── Animated Counter
function AnimatedNumber({ value, decimals = 1, duration = 800 }) {
  const [display, setDisplay] = useState(0)
  const startRef = useRef(null)
  const startValRef = useRef(0)

  useEffect(() => {
    startValRef.current = display
    startRef.current = null
    const target = Number(value)
    const animate = (ts) => {
      if (!startRef.current) startRef.current = ts
      const progress = Math.min((ts - startRef.current) / duration, 1)
      const eased = 1 - Math.pow(1 - progress, 3)
      setDisplay(startValRef.current + (target - startValRef.current) * eased)
      if (progress < 1) requestAnimationFrame(animate)
      else setDisplay(target)
    }
    requestAnimationFrame(animate)
  }, [value])

  return <>{display.toFixed(decimals)}</>
}

// ── Heartbeat ECG line SVG
function HeartbeatLine({ active, color = '#ef4444' }) {
  return (
    <svg viewBox="0 0 200 40" style={{ width: '100%', height: 40, overflow: 'visible' }}>
      <polyline
        points="0,20 30,20 40,5 50,35 60,10 70,30 80,20 200,20"
        fill="none"
        stroke={active ? color : '#e2e8f0'}
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
        style={{
          strokeDasharray: 300,
          strokeDashoffset: active ? 0 : 300,
          transition: 'stroke-dashoffset 1.2s ease, stroke 0.5s',
        }}
      />
    </svg>
  )
}

// ── Risk Gauge SVG — enhanced
function RiskGauge({ probability, status }) {
  const r = 54, cx = 70, cy = 70
  const circ = 2 * Math.PI * r
  const half = circ / 2
  const fill = (probability / 100) * half
  const color = probability < 35 ? '#10b981' : probability < 65 ? '#f59e0b' : '#ef4444'
  const bgColor = probability < 35 ? 'rgba(16,185,129,0.12)' : probability < 65 ? 'rgba(245,158,11,0.12)' : 'rgba(239,68,68,0.12)'
  return (
    <svg viewBox="0 0 140 96" style={{ width: 190, height: 130 }}>
      <defs>
        <filter id="glow">
          <feGaussianBlur stdDeviation="3" result="coloredBlur" />
          <feMerge><feMergeNode in="coloredBlur" /><feMergeNode in="SourceGraphic" /></feMerge>
        </filter>
      </defs>
      {/* Background arc */}
      <path d={`M ${cx - r} ${cy} A ${r} ${r} 0 0 1 ${cx + r} ${cy}`}
        fill="none" stroke="#e2e8f0" strokeWidth="13" strokeLinecap="round" />
      {/* Glow fill arc */}
      <path d={`M ${cx - r} ${cy} A ${r} ${r} 0 0 1 ${cx + r} ${cy}`}
        fill="none" stroke={color} strokeWidth="13" strokeLinecap="round"
        strokeDasharray={`${fill} ${half}`}
        filter="url(#glow)"
        style={{ transition: 'stroke-dasharray 1s cubic-bezier(.4,0,.2,1), stroke 0.5s' }} />
      {/* Center circle */}
      <circle cx={cx} cy={cy} r={28} fill={bgColor} />
      <text x={cx} y={cy - 4} textAnchor="middle" fontSize="17" fontWeight="800" fill={color}>
        {probability.toFixed(1)}%
      </text>
      <text x={cx} y={cy + 13} textAnchor="middle" fontSize="9" fontWeight="700" fill={color} letterSpacing="0.5">
        {status === 'high' ? 'HIGH RISK' : status === 'low' ? 'LOW RISK' : 'CONFIDENCE'}
      </text>
      <text x={cx - r} y={cy + 22} textAnchor="middle" fontSize="9" fill="#94a3b8">0%</text>
      <text x={cx + r} y={cy + 22} textAnchor="middle" fontSize="9" fill="#94a3b8">100%</text>
    </svg>
  )
}

// ── Risk Level Badge
function RiskBadge({ value, fieldKey }) {
  const t = riskThresholds[fieldKey]
  if (!t) return null
  // Inverted fields: LOW value is dangerous (e.g. Max Heart Rate)
  const isHigh = t.inverted ? value <= t.danger : value >= t.danger
  const isWarn = t.inverted
    ? (value > t.danger && value <= t.warn)
    : (value >= t.warn && value < t.danger)
  if (!isHigh && !isWarn) return null
  return (
    <span style={{
      fontSize: 10, fontWeight: 800, padding: '2px 7px', borderRadius: 999,
      background: isHigh ? '#fef2f2' : '#fffbeb',
      color: isHigh ? '#dc2626' : '#d97706',
      border: `1px solid ${isHigh ? '#fecaca' : '#fde68a'}`,
      marginLeft: 6, verticalAlign: 'middle',
      animation: isHigh ? 'pulse-badge 1.5s ease-in-out infinite' : 'none',
    }}>
      {isHigh ? '⚠ HIGH' : '⚠ WATCH'}
    </span>
  )
}

// ── Feature Info Tooltip
function InfoTooltip({ fieldKey }) {
  const [open, setOpen] = useState(false)
  const info = featureInfo[fieldKey]
  if (!info) return null
  return (
    <span style={{ position: 'relative', display: 'inline-block' }}>
      <button
        onClick={() => setOpen(o => !o)}
        className="tooltip-btn"
        title="What is this?"
      >?</button>
      {open && (
        <div className="tooltip-popup">
          <strong style={{ color: '#7dd3fc', display: 'block', marginBottom: 4 }}>{info.label}</strong>
          {info.normal && <span style={{ color: '#86efac', display: 'block', marginBottom: 6, fontSize: 11 }}>📊 {info.normal}</span>}
          <span style={{ lineHeight: 1.6 }}>{info.desc}</span>
          <button onClick={() => setOpen(false)} className="tooltip-close">✕ Close</button>
        </div>
      )}
    </span>
  )
}

// ── Print Report
function printReport(prediction, inputs, disease, fields) {
  const now = new Date().toLocaleString()
  const rows = fields.map(f => `<tr><td>${f.label}</td><td>${inputs[f.key]}${f.unit ? ' ' + f.unit : ''}</td>${f.normal ? `<td style="color:#64748b;font-size:12px">${f.normal}</td>` : '<td></td>'}</tr>`).join('')
  const modelRows = prediction.models.map(m =>
    `<tr><td>${m.name}</td><td>${m.label}</td><td>${m.probability.toFixed(1)}%</td></tr>`
  ).join('')
  const riskColor = prediction.best_model.prediction === 1 ? '#dc2626' : '#059669'
  const html = `<!DOCTYPE html><html><head><title>Healthcare Report</title>
  <style>body{font-family:Arial,sans-serif;padding:32px;color:#182230;max-width:800px;margin:0 auto}
  h1{color:#1e40af;border-bottom:2px solid #1e40af;padding-bottom:8px}
  h2{color:#374151;margin-top:24px}
  table{width:100%;border-collapse:collapse;margin-top:12px}
  th{background:#f1f5f9;text-align:left;padding:8px 12px;font-size:13px}
  td{padding:8px 12px;border-bottom:1px solid #e2e8f0;font-size:13px}
  .risk{font-size:28px;font-weight:bold;color:${riskColor};padding:16px;background:${prediction.best_model.prediction === 1 ? '#fef2f2' : '#f0fdf4'};border-radius:8px;text-align:center;margin:16px 0}
  .footer{margin-top:32px;color:#94a3b8;font-size:11px;border-top:1px solid #e2e8f0;padding-top:12px}
  @media print{button{display:none}}</style></head><body>
  <h1>🏥 Healthcare Prediction Report</h1>
  <p><strong>Date:</strong> ${now} &nbsp;|&nbsp; <strong>Disease:</strong> ${disease} &nbsp;|&nbsp; <strong>Best Model:</strong> ${prediction.best_model.name}</p>
  <div class="risk">Final Result: ${prediction.best_model.label} — ${prediction.best_model.probability.toFixed(1)}% Confidence</div>
  <h2>Patient Input Values</h2>
  <table><thead><tr><th>Feature</th><th>Value</th><th>Normal Range</th></tr></thead><tbody>${rows}</tbody></table>
  <h2>Model Comparison</h2>
  <table><thead><tr><th>Model</th><th>Prediction</th><th>Confidence</th></tr></thead><tbody>${modelRows}</tbody></table>
  <div class="footer">Generated by Healthcare Prediction System &nbsp;|&nbsp; For educational purposes only. Not a medical diagnosis.</div>
  <script>window.onload=()=>window.print()</script></body></html>`
  const w = window.open('', '_blank')
  w.document.write(html)
  w.document.close()
}

// ── Risk Score Preview (live estimate before full prediction)
function LiveRiskMeter({ inputs, fields, disease }) {
  const score = useMemo(() => {
    let pts = 0; let max = 0
    fields.forEach(f => {
      const t = riskThresholds[f.key]
      if (!t) return
      max += 2
      const v = inputs[f.key]
      if (t.inverted) {
        if (v <= t.danger) pts += 2
        else if (v <= t.warn) pts += 1
      } else {
        if (v >= t.danger) pts += 2
        else if (v >= t.warn) pts += 1
      }
    })
    return max > 0 ? Math.round((pts / max) * 100) : 0
  }, [inputs, fields])

  const color = score < 35 ? '#10b981' : score < 65 ? '#f59e0b' : '#ef4444'
  const label = score < 35 ? 'Low' : score < 65 ? 'Moderate' : 'High'

  return (
    <div className="live-risk-card">
      <div className="live-risk-header">
        <span>⚡ Live Risk Preview</span>
        <span className="live-badge" style={{ background: `${color}22`, color }}>
          {label} Risk
        </span>
      </div>
      <div className="live-risk-bar-wrap">
        <div className="live-risk-bar" style={{ '--pct': `${score}%`, '--col': color }} />
      </div>
      <div className="live-risk-labels">
        <span style={{ color: '#10b981' }}>Low</span>
        <span style={{ color: '#f59e0b' }}>Moderate</span>
        <span style={{ color: '#ef4444' }}>High</span>
      </div>
      <p style={{ margin: '8px 0 0', fontSize: 12, color: '#64748b', lineHeight: 1.5 }}>
        📌 This is a <strong>live heuristic estimate</strong> based on known thresholds. Run the full ML prediction for accurate results.
      </p>
    </div>
  )
}

// ── Step Progress indicator
function StepProgress({ step }) {
  const steps = ['Enter Values', 'Run Prediction', 'View Results']
  return (
    <div className="step-progress">
      {steps.map((s, i) => (
        <div key={s} className={`step-item ${i < step ? 'done' : i === step ? 'active' : ''}`}>
          <div className="step-dot">{i < step ? '✓' : i + 1}</div>
          <span>{s}</span>
          {i < steps.length - 1 && <div className={`step-line ${i < step ? 'done' : ''}`} />}
        </div>
      ))}
    </div>
  )
}

// ── Main App
function App() {
  const [booting, setBooting] = useState(true)
  const [bootStep, setBootStep] = useState(0)
  const [loading, setLoading] = useState(false)
  const [apiStatus, setApiStatus] = useState('checking')
  const [disease, setDisease] = useState('Diabetes')
  const [activeView, setActiveView] = useState('predict')
  const [prediction, setPrediction] = useState(null)
  const [error, setError] = useState(null)
  const [inputs, setInputs] = useState(() => buildDefaultInputs(diabetesFields))
  const [history, setHistory] = useState(loadHistory)
  const [darkMode, setDarkMode] = useState(false)
  const [activeField, setActiveField] = useState(null)
  const [predStep, setPredStep] = useState(0)
  const [searchQuery, setSearchQuery] = useState('')
  const [expandedCard, setExpandedCard] = useState(null)
  const [historyFilter, setHistoryFilter] = useState('all')
  const [showCompareTip, setShowCompareTip] = useState(false)

  const fields = useMemo(() => (disease === 'Diabetes' ? diabetesFields : heartFields), [disease])
  const filteredFields = useMemo(() => {
    if (!searchQuery.trim()) return fields
    return fields.filter(f => f.label.toLowerCase().includes(searchQuery.toLowerCase()))
  }, [fields, searchQuery])

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', darkMode ? 'dark' : 'light')
    document.body.style.background = darkMode
      ? 'linear-gradient(180deg,#0f172a 0,#1e293b 100%)'
      : 'linear-gradient(180deg,#dfeef6 0,#f7fbfd 420px,#eef5f8 100%)'
  }, [darkMode])

  // Boot animation sequence
  useEffect(() => {
    const steps = [300, 600, 900, 1200]
    steps.forEach((t, i) => setTimeout(() => setBootStep(i + 1), t))
    const timer = setTimeout(() => setBooting(false), 1600)
    return () => clearTimeout(timer)
  }, [])

  useEffect(() => {
    let ignore = false
    async function checkApi() {
      try {
        const r = await fetch(`${API_BASE}/`)
        if (!ignore) setApiStatus(r.ok ? 'online' : 'offline')
      } catch { if (!ignore) setApiStatus('offline') }
    }
    checkApi()
    return () => { ignore = true }
  }, [])

  useEffect(() => {
    setInputs(buildDefaultInputs(fields))
    setPrediction(null)
    setError(null)
    setPredStep(0)
    setSearchQuery('')
  }, [fields])

  function updateInput(key, value) {
    setInputs(prev => ({ ...prev, [key]: value }))
    if (prediction) { setPrediction(null); setPredStep(0) }
  }

  async function handleRunPrediction() {
    setLoading(true); setPrediction(null); setError(null); setPredStep(1)
    try {
      const response = await fetch(`${API_BASE}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ disease, features: inputs }),
      })
      const data = await response.json()
      if (!response.ok) throw new Error(data.detail || 'Prediction failed')
      setPrediction(data)
      setPredStep(2)
      setApiStatus('online')
      const entry = {
        id: Date.now(),
        date: new Date().toLocaleString(),
        disease,
        result: data.best_model.label,
        confidence: data.best_model.probability.toFixed(1),
        risk: data.best_model.prediction === 1 ? 'high' : 'low',
        model: data.best_model.name,
        inputs: { ...inputs },
        models: data.models,
      }
      const updated = [entry, ...history]
      setHistory(updated)
      saveHistory(updated)
    } catch {
      setApiStatus('offline')
      setError(`Failed to fetch from ${API_BASE}. Start the FastAPI backend on port 8000, then try again.`)
      setPredStep(0)
    } finally { setLoading(false) }
  }

  const finalStatus = useMemo(() => {
    if (!prediction) return 'idle'
    return prediction.best_model.prediction === 1 ? 'high' : 'low'
  }, [prediction])

  const averageConfidence = useMemo(() => {
    if (!prediction) return 0
    return prediction.models.reduce((s, i) => s + i.probability, 0) / prediction.models.length
  }, [prediction])

  const filteredHistory = useMemo(() => {
    if (historyFilter === 'all') return history
    if (historyFilter === 'high') return history.filter(h => h.risk === 'high')
    if (historyFilter === 'low') return history.filter(h => h.risk === 'low')
    return history.filter(h => h.disease === historyFilter)
  }, [history, historyFilter])

  const histStats = useMemo(() => ({
    total: history.length,
    high: history.filter(h => h.risk === 'high').length,
    low: history.filter(h => h.risk === 'low').length,
    diabetes: history.filter(h => h.disease === 'Diabetes').length,
    heart: history.filter(h => h.disease === 'Heart Disease').length,
  }), [history])

  if (booting) return (
    <main className="welcome-screen">
      {/* Animated background blobs */}
      <div className="welcome-bg">
        <div className="wb-blob wb-blob1" />
        <div className="wb-blob wb-blob2" />
        <div className="wb-blob wb-blob3" />
      </div>

      <section className="welcome-card">
        {/* Icon */}
        <div className="welcome-icon-ring">
          <div className="welcome-icon-inner">
            <svg width="38" height="38" viewBox="0 0 38 38" fill="none">
              <path d="M19 6C19 6 8 13 8 21C8 26.5 13 31 19 31C25 31 30 26.5 30 21C30 13 19 6 19 6Z" fill="url(#hg)" />
              <path d="M13 19H17L19 15L21 23L23 19H25" stroke="white" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
              <defs>
                <linearGradient id="hg" x1="8" y1="6" x2="30" y2="31" gradientUnits="userSpaceOnUse">
                  <stop stopColor="#2757d7"/>
                  <stop offset="1" stopColor="#0f9f8f"/>
                </linearGradient>
              </defs>
            </svg>
          </div>
          <div className="welcome-ring r1" />
          <div className="welcome-ring r2" />
        </div>

        <p className="welcome-eyebrow">AI-Powered Clinical Tool</p>
        <h1 className="welcome-title">Healthcare<br />Prediction System</h1>
        <p className="welcome-subtitle">Diabetes & Heart Disease Risk Screening using Machine Learning</p>

        {/* Feature pills */}
        <div className="welcome-pills">
          {['🧠 3 ML Models', '📊 Live Risk Meter', '🩺 13 Heart Fields', '🩸 8 Diabetes Fields'].map(p => (
            <span key={p} className="welcome-pill">{p}</span>
          ))}
        </div>

        {/* Boot steps */}
        <div className="boot-steps">
          {['Initializing models...', 'Loading patient data...', 'Preparing dashboard...', 'Ready!'].map((s, i) => (
            <div key={s} className={`boot-step ${bootStep > i ? 'done' : bootStep === i ? 'active' : ''}`}>
              <span className="boot-dot">{bootStep > i ? '✓' : bootStep === i ? '◉' : '○'}</span>
              <span>{s}</span>
              {bootStep > i && <span className="boot-check-anim">✓</span>}
            </div>
          ))}
        </div>

        {/* Progress bar */}
        <div className="welcome-progress-wrap">
          <div className="welcome-progress-bar" style={{ width: `${(bootStep / 4) * 100}%` }} />
        </div>
        <p className="welcome-progress-label">{Math.round((bootStep / 4) * 100)}% loaded</p>
      </section>
    </main>
  )

  return (
    <main className="app-shell">
      {loading && (
        <div className="data-loader" role="status" aria-live="polite">
          <div className="loader-card">
            <div className="spinner" />
            <strong>Running prediction</strong>
            <span>Analyzing the patient profile with 3 trained ML models.</span>
            <div className="loader-models">
              {['Logistic Regression', 'KNN', 'Decision Tree'].map((m, i) => (
                <div key={m} className="loader-model-row" style={{ animationDelay: `${i * 0.3}s` }}>
                  <div className="loader-model-bar" />
                  <span>{m}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* TOPBAR */}
      <header className="app-topbar">
        <div className="brand">
          <div className="brand-mark">H</div>
          <div>
            <strong>Healthcare Predictor</strong>
            <span>Diabetes and heart disease risk screening</span>
          </div>
        </div>
        <div className="topbar-right">
          <div className="topbar-chip">Clinical ML Studio</div>
          <button className="ghost-button" onClick={() => setDarkMode(d => !d)} style={{ minHeight: 36, padding: '0 13px', fontSize: '0.86rem' }}>
            {darkMode ? '☀️ Light Mode' : '🌙 Dark Mode'}
          </button>
          <div className={`api-pill ${apiStatus}`}>
            <span />
            API {apiStatus === 'online' ? 'Online' : apiStatus === 'checking' ? 'Checking' : 'Offline'}
          </div>
        </div>
      </header>

      {/* NAV */}
      <nav className="app-nav" aria-label="App sections">
        {views.map(view => (
          <button key={view.id} className={activeView === view.id ? 'active' : ''} onClick={() => setActiveView(view.id)}>
            <span className="nav-emoji">{view.label.split(' ')[0]}</span>
            <span className="nav-text">{view.label.split(' ').slice(1).join(' ')}</span>
            <span className="nav-desc">{view.desc}</span>
          </button>
        ))}
      </nav>

      {/* HERO */}
      <section className="hero-panel">
        <div>
          <p className="eyebrow">{activeView === 'predict' ? 'Patient assessment' : 'Notebook insights'}</p>
          <h1>{activeView === 'predict' ? 'Predict health risk from clinical values' : 'Explore saved analysis visuals'}</h1>
        </div>
        <div className="hero-control-card">
          <span>Screening Mode</span>
          <div className="model-switch" aria-label="Disease model">
            {['Diabetes', 'Heart Disease'].map(item => (
              <button key={item} className={disease === item ? 'active' : ''} onClick={() => setDisease(item)}>
                <span style={{ marginRight: 6 }}>{item === 'Diabetes' ? '🩸' : '❤️'}</span>
                {item}
              </button>
            ))}
          </div>
          <div className="hero-mini-stats">
            <div><strong>3</strong><small>models</small></div>
            <div><strong>{fields.length}</strong><small>inputs</small></div>
            <div><strong>{history.length}</strong><small>saved</small></div>
            <div className={finalStatus !== 'idle' ? finalStatus : ''}>
              <strong>{prediction ? `${averageConfidence.toFixed(0)}%` : '--'}</strong>
              <small>avg conf</small>
            </div>
          </div>
        </div>
      </section>

      {/* SUMMARY STRIP */}
      {activeView === 'predict' && (
        <section className="summary-strip">
          <article><span>Selected model</span><strong>{disease}</strong></article>
          <article><span>Inputs</span><strong>{fields.length}</strong></article>
          <article><span>Average signal</span><strong>{prediction ? <><AnimatedNumber value={averageConfidence} />%</> : '--'}</strong></article>
          <article className={finalStatus}><span>Risk status</span><strong>{prediction ? (finalStatus === 'high' ? '🔴 High' : '🟢 Low') : 'Waiting'}</strong></article>
        </section>
      )}

      {/* PREDICT VIEW */}
      {activeView === 'predict' && (
        <section className="app-grid">
          {/* INPUT CARD */}
          <section className="card input-card">
            <div className="card-heading">
              <div>
                <p className="eyebrow">Step 1</p>
                <h2>Enter Patient Values</h2>
                <p style={{ margin: '4px 0 0', fontSize: '0.85rem', color: '#64748b' }}>
                  Hover <strong>?</strong> for medical context. Fields highlighted in <span style={{ color: '#ef4444' }}>red</span> are above danger threshold.
                </p>
              </div>
              <div style={{ display: 'flex', gap: 8 }}>
                <button className="ghost-button" onClick={() => setInputs(buildDefaultInputs(fields))}>↺ Reset</button>
              </div>
            </div>

            {/* Search bar */}
            <div className="search-bar-wrap">
              <span className="search-icon">🔍</span>
              <input
                className="search-bar"
                type="text"
                placeholder="Filter fields by name..."
                value={searchQuery}
                onChange={e => setSearchQuery(e.target.value)}
              />
              {searchQuery && (
                <button className="search-clear" onClick={() => setSearchQuery('')}>✕</button>
              )}
            </div>

            {/* Step Progress */}
            <StepProgress step={predStep} />

            {/* Live risk meter */}
            <LiveRiskMeter inputs={inputs} fields={fields} disease={disease} />

            <div className="fields">
              {filteredFields.length === 0 && (
                <div className="empty-state">No fields match "<strong>{searchQuery}</strong>"</div>
              )}
              {filteredFields.map(field => {
                const isActive = activeField === field.key
                const t = riskThresholds[field.key]
                const v = inputs[field.key]
                const isDanger = t && !field.type && (t.inverted ? v <= t.danger : v >= t.danger)
                const isWarn   = t && !field.type && (t.inverted ? (v > t.danger && v <= t.warn) : (v >= t.warn && v < t.danger))
                return (
                  <label
                    key={field.key}
                    className={`field-row ${isActive ? 'active-field' : ''} ${isDanger ? 'danger-field' : isWarn ? 'warn-field' : ''}`}
                    onMouseEnter={() => setActiveField(field.key)}
                    onMouseLeave={() => setActiveField(null)}
                  >
                    <div className="field-info">
                      <strong>
                        <span className="field-icon">{field.icon}</span>
                        {field.label}
                        <InfoTooltip fieldKey={field.key} />
                        {!field.type && <RiskBadge value={inputs[field.key]} fieldKey={field.key} />}
                      </strong>
                      <span>{field.unit || 'Choose one'}</span>
                      {field.normal && (
                        <span style={{ fontSize: '0.76rem', color: '#059669', fontWeight: 700, marginTop: 2 }}>
                          📏 {field.normal}
                        </span>
                      )}
                    </div>
                    <div className="field-control">
                      {field.type === 'select' ? (
                        <select value={inputs[field.key]} onChange={e => updateInput(field.key, Number(e.target.value))}>
                          {field.options.map(opt => <option key={opt.value} value={opt.value}>{opt.label}</option>)}
                        </select>
                      ) : (
                        <div style={{ position: 'relative' }}>
                          <input type="range" min={field.min} max={field.max} step={field.step}
                            value={inputs[field.key]} onChange={e => updateInput(field.key, Number(e.target.value))}
                            style={{ '--val': `${((inputs[field.key] - field.min) / (field.max - field.min)) * 100}%` }}
                          />
                          {t && !field.type && (
                            <div className="range-thresholds" style={{ '--warn': `${((t.warn - field.min) / (field.max - field.min)) * 100}%`, '--danger': `${((t.danger - field.min) / (field.max - field.min)) * 100}%` }}>
                              <span className="thresh-marker warn" style={{ left: 'var(--warn)' }} title={`Warning: ${t.warn}`} />
                              <span className="thresh-marker danger" style={{ left: 'var(--danger)' }} title={`Danger: ${t.danger}`} />
                            </div>
                          )}
                        </div>
                      )}
                    </div>
                    <output className={isDanger ? 'danger-output' : isWarn ? 'warn-output' : ''}>{inputs[field.key]}</output>
                  </label>
                )
              })}
            </div>

            <button className="primary-button" onClick={handleRunPrediction} disabled={loading}>
              {loading ? <><span className="btn-spinner" /> Predicting...</> : '🔬 Run Full ML Prediction'}
            </button>
            {error && <div className="error-box">⚠️ {error}</div>}
          </section>

          {/* RESULT STACK */}
          <aside className="result-stack">
            <section className={`result-card ${finalStatus}`}>
              <p className="eyebrow">Step 2 — Result</p>
              {/* Heartbeat animation */}
              <div style={{ marginBottom: 8 }}>
                <HeartbeatLine active={!!prediction} color={finalStatus === 'high' ? '#ef4444' : '#10b981'} />
              </div>
              <div className="result-status">
                <span>{prediction ? prediction.best_model.name : 'Ready'}</span>
                <strong>{prediction ? prediction.best_model.label : 'No result yet'}</strong>
                <p>{prediction
                  ? prediction.best_model.explanation
                  : 'Run a prediction to view the risk result and model confidence.'}</p>
              </div>

              <div style={{ display: 'flex', alignItems: 'center', gap: 16, marginTop: 12 }}>
                <RiskGauge
                  probability={prediction ? prediction.best_model.probability : 0}
                  status={finalStatus}
                />
                {prediction && (
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                    <button onClick={() => printReport(prediction, inputs, disease, fields)} className="result-action-btn">
                      🖨️ Print Report
                    </button>
                    <div className="result-consensus">
                      <span style={{ fontSize: 11, color: 'rgba(255,255,255,0.7)' }}>Model Consensus:</span>
                      <strong style={{ fontSize: 13, color: '#fff' }}>
                        {prediction.models.filter(m => m.prediction === prediction.best_model.prediction).length}/3 agree
                      </strong>
                    </div>
                  </div>
                )}
              </div>

              {prediction && (
                <div className="result-tags">
                  <span>Disease: {disease}</span>
                  <span>{new Date().toLocaleDateString()}</span>
                  <span>Conf: {prediction.best_model.probability.toFixed(1)}%</span>
                </div>
              )}
            </section>

            {/* MODEL COMPARISON */}
            <section className="card models-card">
              <div className="card-heading compact">
                <div><p className="eyebrow">Step 3</p><h2>Model Comparison</h2></div>
                {prediction && (
                  <button className="ghost-button" style={{ fontSize: 12 }}
                    onMouseEnter={() => setShowCompareTip(true)} onMouseLeave={() => setShowCompareTip(false)}>
                    ℹ️ About
                  </button>
                )}
              </div>
              {showCompareTip && (
                <div className="compare-tip">
                  Three independent models vote on the result. The best model (highest confidence) is shown as the final result. Consensus across models increases reliability.
                </div>
              )}
              <div className="model-list">
                {prediction ? prediction.models.map((item, idx) => (
                  <article key={item.name} className="model-row" style={{ animationDelay: `${idx * 0.1}s` }}>
                    <div>
                      <strong>{item.name}</strong>
                      <span>{item.prediction === 1 ? '🔴 High risk' : '🟢 Low risk'}</span>
                    </div>
                    <div className="bar-wrap">
                      <div className="bar">
                        <span style={{ width: `${item.probability}%`, backgroundColor: modelColors[item.name] }} />
                      </div>
                      {item.name === prediction.best_model.name && (
                        <span className="best-model-tag">★ Best</span>
                      )}
                    </div>
                    <output>{item.probability.toFixed(1)}%</output>
                  </article>
                )) : (
                  <div className="empty-state">Model confidence bars will appear after prediction.</div>
                )}
              </div>

              {/* Quick re-run hint */}
              {prediction && (
                <div className="rerun-hint">
                  💡 Adjust any input slider above to auto-reset and re-run a new prediction.
                </div>
              )}
            </section>
          </aside>
        </section>
      )}

      {/* HISTORY VIEW */}
      {activeView === 'history' && (
        <section className="analysis-section">
          <div className="analysis-heading">
            <div>
              <p className="eyebrow">Patient Records</p>
              <h2>Prediction History</h2>
              <p>All predictions are saved locally on this device. Last 50 records are kept.</p>
            </div>
            <div style={{ display: 'flex', gap: 10, alignItems: 'center', flexWrap: 'wrap' }}>
              <span className="history-count-badge">{history.length} records</span>
              {history.length > 0 && (
                <button className="ghost-button" style={{ color: '#dc2626' }}
                  onClick={() => { if (confirm('Clear all history?')) { setHistory([]); saveHistory([]) } }}>
                  🗑️ Clear All
                </button>
              )}
            </div>
          </div>

          {/* Stats Bar */}
          {history.length > 0 && (
            <div className="history-stats-bar">
              <div className="hstat"><strong>{histStats.total}</strong><span>Total</span></div>
              <div className="hstat high"><strong style={{ color: '#dc2626' }}>{histStats.high}</strong><span>High Risk</span></div>
              <div className="hstat low"><strong style={{ color: '#059669' }}>{histStats.low}</strong><span>Low Risk</span></div>
              <div className="hstat"><strong style={{ color: '#2563eb' }}>{histStats.diabetes}</strong><span>Diabetes</span></div>
              <div className="hstat"><strong style={{ color: '#ef6f5e' }}>{histStats.heart}</strong><span>Heart</span></div>
            </div>
          )}

          {/* Filter Tabs */}
          {history.length > 0 && (
            <div className="history-filter-tabs">
              {['all', 'high', 'low', 'Diabetes', 'Heart Disease'].map(f => (
                <button key={f} className={historyFilter === f ? 'active' : ''}
                  onClick={() => setHistoryFilter(f)}>
                  {f === 'all' ? 'All' : f === 'high' ? '🔴 High Risk' : f === 'low' ? '🟢 Low Risk' : f}
                </button>
              ))}
            </div>
          )}

          {filteredHistory.length === 0 ? (
            <div className="empty-state" style={{ padding: 32, textAlign: 'center', marginTop: 16 }}>
              {history.length === 0
                ? 'No predictions yet. Run a prediction to see history here.'
                : `No ${historyFilter} records found.`}
            </div>
          ) : (
            <div style={{ display: 'grid', gap: 12, marginTop: 16 }}>
              {filteredHistory.map(entry => (
                <div
                  key={entry.id}
                  className={`history-card ${entry.risk} ${expandedCard === entry.id ? 'expanded' : ''}`}
                  onClick={() => setExpandedCard(expandedCard === entry.id ? null : entry.id)}
                >
                  <div className="history-card-main">
                    <div>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 6, flexWrap: 'wrap' }}>
                        <span className={`risk-pill ${entry.risk}`}>
                          {entry.risk === 'high' ? '🔴' : '🟢'} {entry.result}
                        </span>
                        <span className="disease-pill">{entry.disease}</span>
                        <span style={{ fontSize: 13, color: '#64748b' }}>{entry.model}</span>
                      </div>
                      <div style={{ fontSize: 13, color: '#64748b' }}>
                        📅 {entry.date} &nbsp;|&nbsp; 🎯 Confidence: <strong>{entry.confidence}%</strong>
                      </div>
                    </div>
                    <div style={{ textAlign: 'right' }}>
                      <div className={`history-confidence ${entry.risk}`}>{entry.confidence}%</div>
                      <div style={{ fontSize: 11, color: '#94a3b8' }}>confidence</div>
                      <div style={{ fontSize: 11, color: '#94a3b8', marginTop: 4 }}>
                        {expandedCard === entry.id ? '▲ collapse' : '▼ expand'}
                      </div>
                    </div>
                  </div>

                  {expandedCard === entry.id && (
                    <div className="history-expanded">
                      <div className="history-models">
                        {entry.models.map(m => (
                          <div key={m.name} className="history-model-row">
                            <span>{m.name}</span>
                            <div className="bar" style={{ flex: 1, margin: '0 10px' }}>
                              <span style={{ width: `${m.probability}%`, backgroundColor: modelColors[m.name] }} />
                            </div>
                            <span style={{ fontWeight: 800, minWidth: 48, textAlign: 'right' }}>{m.probability.toFixed(1)}%</span>
                          </div>
                        ))}
                      </div>
                      <div className="history-inputs-grid">
                        {Object.entries(entry.inputs).map(([k, v]) => (
                          <div key={k} className="history-input-item">
                            <span>{k}</span>
                            <strong>{typeof v === 'number' ? v.toFixed ? v.toFixed(1) : v : v}</strong>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}
        </section>
      )}

      {activeView === 'visuals' && <VisualGallery group={visualGroups[0]} />}
      {activeView === 'metrics' && <VisualGallery group={visualGroups[1]} />}
      {activeView === 'pca' && <VisualGallery group={visualGroups[2]} />}
    </main>
  )
}

// ── Visual Gallery with lightbox
function VisualGallery({ group }) {
  const [lightbox, setLightbox] = useState(null)
  const [filter, setFilter] = useState('all')

  const categories = useMemo(() => {
    const all = ['all']
    group.items.forEach(i => {
      if (i.title.toLowerCase().includes('diabetes') && !all.includes('Diabetes')) all.push('Diabetes')
      if (i.title.toLowerCase().includes('heart') && !all.includes('Heart')) all.push('Heart')
    })
    return all
  }, [group])

  const filtered = useMemo(() => {
    if (filter === 'all') return group.items
    return group.items.filter(i => i.title.toLowerCase().includes(filter.toLowerCase()))
  }, [filter, group])

  return (
    <section className="analysis-section">
      {lightbox && (
        <div className="lightbox" onClick={() => setLightbox(null)}>
          <div className="lightbox-inner" onClick={e => e.stopPropagation()}>
            <button className="lightbox-close" onClick={() => setLightbox(null)}>✕ Close</button>
            <img src={visualUrl(lightbox.file)} alt={lightbox.title} />
            <p>{lightbox.title}</p>
          </div>
        </div>
      )}

      <div className="analysis-heading">
        <div>
          <p className="eyebrow">Analysis</p>
          <h2>{group.title}</h2>
          <p>{group.intro}</p>
        </div>
        <span className="analysis-count-badge">{group.items.length} visuals</span>
      </div>

      {categories.length > 1 && (
        <div className="visual-filter-tabs">
          {categories.map(c => (
            <button key={c} className={filter === c ? 'active' : ''} onClick={() => setFilter(c)}>
              {c === 'all' ? 'All' : c}
            </button>
          ))}
        </div>
      )}

      <div className="visual-grid">
        {filtered.map(item => (
          <article key={item.file} className={item.span === 'wide' ? 'visual-card wide' : 'visual-card'}>
            <div className="visual-title">
              <strong>{item.title}</strong>
              <div style={{ display: 'flex', gap: 8 }}>
                <button className="visual-zoom-btn" onClick={() => setLightbox(item)}>🔍 Zoom</button>
                <a href={visualUrl(item.file)} target="_blank" rel="noreferrer" className="visual-open-btn">Open ↗</a>
              </div>
            </div>
            <div className="visual-img-wrap" onClick={() => setLightbox(item)}>
              <img src={visualUrl(item.file)} alt={item.title} loading="lazy" />
              <div className="visual-overlay">Click to zoom</div>
            </div>
          </article>
        ))}
      </div>
    </section>
  )
}

export default App
