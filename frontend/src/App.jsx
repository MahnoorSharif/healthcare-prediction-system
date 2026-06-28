import React, { useEffect, useMemo, useState, useRef } from 'react'
import {
  translations,
  getFieldLabel,
  getFeatureInfo,
  localizeFields,
  getModelName,
  translateDiseaseName,
  translateResultLabel,
  localizePrediction,
  getVisualGroups,
  loadLang,
  saveLang,
} from './i18n'

const localHosts = new Set(['localhost', '127.0.0.1', ''])
const apiHost = localHosts.has(window.location.hostname) ? '127.0.0.1' : window.location.hostname
const API_BASE = import.meta.env.VITE_API_URL || `${window.location.protocol}//${apiHost}:8000`

const riskThresholds = {
  glucose: { warn: 100, danger: 126 },
  bp: { warn: 80, danger: 100 },
  bmi: { warn: 25, danger: 30 },
  insulin: { warn: 166, danger: 300 },
  age: { warn: 45, danger: 65 },
  chol: { warn: 200, danger: 240 },
  trestbps: { warn: 120, danger: 140 },
  thalach: { warn: 100, danger: 70, inverted: true },
  oldpeak: { warn: 1, danger: 2 },
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

function RiskGauge({ probability, status, t }) {
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
      <path d={`M ${cx - r} ${cy} A ${r} ${r} 0 0 1 ${cx + r} ${cy}`}
        fill="none" stroke="#e2e8f0" strokeWidth="13" strokeLinecap="round" />
      <path d={`M ${cx - r} ${cy} A ${r} ${r} 0 0 1 ${cx + r} ${cy}`}
        fill="none" stroke={color} strokeWidth="13" strokeLinecap="round"
        strokeDasharray={`${fill} ${half}`}
        filter="url(#glow)"
        style={{ transition: 'stroke-dasharray 1s cubic-bezier(.4,0,.2,1), stroke 0.5s' }} />
      <circle cx={cx} cy={cy} r={28} fill={bgColor} />
      <text x={cx} y={cy - 4} textAnchor="middle" fontSize="17" fontWeight="800" fill={color}>
        {probability.toFixed(1)}%
      </text>
      <text x={cx} y={cy + 13} textAnchor="middle" fontSize="9" fontWeight="700" fill={color} letterSpacing="0.5">
        {status === 'high' ? t.highRiskLabel : status === 'low' ? t.lowRiskLabel : t.confidence}
      </text>
      <text x={cx - r} y={cy + 22} textAnchor="middle" fontSize="9" fill="#94a3b8">0%</text>
      <text x={cx + r} y={cy + 22} textAnchor="middle" fontSize="9" fill="#94a3b8">100%</text>
    </svg>
  )
}

function RiskBadge({ value, fieldKey, t }) {
  const thr = riskThresholds[fieldKey]
  if (!thr) return null
  const isHigh = thr.inverted ? value <= thr.danger : value >= thr.danger
  const isWarn = thr.inverted
    ? (value > thr.danger && value <= thr.warn)
    : (value >= thr.warn && value < thr.danger)
  if (!isHigh && !isWarn) return null
  return (
    <span style={{
      fontSize: 10, fontWeight: 800, padding: '2px 7px', borderRadius: 999,
      background: isHigh ? '#fef2f2' : '#fffbeb',
      color: isHigh ? '#dc2626' : '#d97706',
      border: `1px solid ${isHigh ? '#fecaca' : '#fde68a'}`,
      marginInlineStart: 6, verticalAlign: 'middle',
      animation: isHigh ? 'pulse-badge 1.5s ease-in-out infinite' : 'none',
    }}>
      {isHigh ? t.highBadge : t.watchBadge}
    </span>
  )
}

function InfoTooltip({ fieldKey, lang }) {
  const [open, setOpen] = useState(false)
  const info = getFeatureInfo(fieldKey, lang)
  const t = translations[lang]
  if (!info) return null
  return (
    <span style={{ position: 'relative', display: 'inline-block' }}>
      <button
        onClick={() => setOpen(o => !o)}
        className="tooltip-btn"
        title={t.whatIsThis}
      >?</button>
      {open && (
        <div className="tooltip-popup">
          <strong style={{ color: '#7dd3fc', display: 'block', marginBottom: 4 }}>{info.label}</strong>
          {info.normal && <span style={{ color: '#86efac', display: 'block', marginBottom: 6, fontSize: 11 }}>📊 {info.normal}</span>}
          <span style={{ lineHeight: 1.6 }}>{info.desc}</span>
          <button onClick={() => setOpen(false)} className="tooltip-close">{t.close}</button>
        </div>
      )}
    </span>
  )
}

function printReport(prediction, inputs, disease, fields, t, lang) {
  const now = new Date().toLocaleString(lang === 'ur' ? 'ur-PK' : undefined)
  const diseaseLabel = translateDiseaseName(disease, t)
  const rows = fields.map(f => `<tr><td>${f.label}</td><td>${inputs[f.key]}${f.unit ? ' ' + f.unit : ''}</td>${f.normal ? `<td style="color:#64748b;font-size:12px">${f.normal}</td>` : '<td></td>'}</tr>`).join('')
  const modelRows = prediction.models.map(m => `<tr><td>${m.name}</td><td>${m.label}</td><td>${m.probability.toFixed(1)}%</td></tr>`).join('')
  const riskColor = prediction.best_model.prediction === 1 ? '#dc2626' : '#059669'
  const html = `<!DOCTYPE html><html lang="${lang}" dir="${lang === 'ur' ? 'rtl' : 'ltr'}"><head><title>${t.reportTitle}</title>
  <style>body{font-family:Arial,sans-serif;padding:32px;color:#182230;max-width:800px;margin:0 auto}
  h1{color:#1e40af;border-bottom:2px solid #1e40af;padding-bottom:8px}
  h2{color:#374151;margin-top:24px}
  table{width:100%;border-collapse:collapse;margin-top:12px}
  th{background:#f1f5f9;text-align:left;padding:8px 12px;font-size:13px}
  td{padding:8px 12px;border-bottom:1px solid #e2e8f0;font-size:13px}
  .risk{font-size:28px;font-weight:bold;color:${riskColor};padding:16px;background:${prediction.best_model.prediction === 1 ? '#fef2f2' : '#f0fdf4'};border-radius:8px;text-align:center;margin:16px 0}
  .footer{margin-top:32px;color:#94a3b8;font-size:11px;border-top:1px solid #e2e8f0;padding-top:12px}
  @media print{button{display:none}}</style></head><body>
  <h1>🏥 ${t.reportTitle}</h1>
  <p><strong>${t.reportDate}:</strong> ${now} &nbsp;|&nbsp; <strong>${t.reportDisease}:</strong> ${diseaseLabel} &nbsp;|&nbsp; <strong>${t.reportBestModel}:</strong> ${prediction.best_model.name}</p>
  <div class="risk">${t.reportFinalResult}: ${prediction.best_model.label} — ${prediction.best_model.probability.toFixed(1)}% ${t.reportConfidence}</div>
  <h2>${t.reportPatientInputs}</h2>
  <table><thead><tr><th>${t.reportFeature}</th><th>${t.reportValue}</th><th>${t.reportNormalRange}</th></tr></thead><tbody>${rows}</tbody></table>
  <h2>${t.reportModelComparison}</h2>
  <table><thead><tr><th>${t.reportModel}</th><th>${t.reportPrediction}</th><th>${t.reportConfidence}</th></tr></thead><tbody>${modelRows}</tbody></table>
  <div class="footer">${t.reportFooter}</div>
  <script>window.onload=()=>window.print()</script></body></html>`
  const w = window.open('', '_blank')
  w.document.write(html)
  w.document.close()
}

function LiveRiskMeter({ inputs, fields, t }) {
  const score = useMemo(() => {
    let pts = 0
    let max = 0
    fields.forEach((f) => {
      const thr = riskThresholds[f.key]
      if (!thr) return
      max += 2
      const v = inputs[f.key]
      if (thr.inverted) {
        if (v <= thr.danger) pts += 2
        else if (v <= thr.warn) pts += 1
      } else {
        if (v >= thr.danger) pts += 2
        else if (v >= thr.warn) pts += 1
      }
    })
    return max > 0 ? Math.round((pts / max) * 100) : 0
  }, [inputs, fields])

  const color = score < 35 ? '#10b981' : score < 65 ? '#f59e0b' : '#ef4444'
  const labelKey = score < 35 ? 'lowLabel' : score < 65 ? 'moderateLabel' : 'highLabel'
  const riskTextKey = score < 35 ? 'lowRiskLabel' : score < 65 ? 'moderateLabel' : 'highRiskLabel'

  return (
    <div className="live-risk-card">
      <div className="live-risk-header">
        <span>{t.liveRisk}</span>
        <span className="live-badge" style={{ background: `${color}22`, color }}>
          {t[riskTextKey]} {t.riskSignal}
        </span>
      </div>
      <div className="live-risk-bar-wrap">
        <div className="live-risk-bar" style={{ '--pct': `${score}%`, '--col': color }} />
      </div>
      <div className="live-risk-labels">
        <span style={{ color: '#10b981' }}>{t.lowLabel}</span>
        <span style={{ color: '#f59e0b' }}>{t.moderateLabel}</span>
        <span style={{ color: '#ef4444' }}>{t.highLabel}</span>
      </div>
      <p style={{ margin: '8px 0 0', fontSize: 12, color: '#64748b', lineHeight: 1.5 }}>{t.liveRiskNote}</p>
    </div>
  )
}

function StepProgress({ step, t }) {
  const steps = [t.enterValues, t.runPred, t.viewResults]
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

function VisualGallery({ group, t }) {
  const [lightbox, setLightbox] = useState(null)
  const [filter, setFilter] = useState('all')

  const categories = useMemo(() => {
    const all = ['all']
    group.items.forEach((i) => {
      if (i.title.toLowerCase().includes('diabetes') && !all.includes('Diabetes')) all.push('Diabetes')
      if (i.title.toLowerCase().includes('heart') && !all.includes('Heart')) all.push('Heart')
      if (i.title.includes('ذیابیطس') && !all.includes('Diabetes')) all.push('Diabetes')
      if ((i.title.includes('دل') || i.title.includes('Heart')) && !all.includes('Heart')) all.push('Heart')
    })
    return all
  }, [group])

  const filtered = useMemo(() => {
    if (filter === 'all') return group.items
    if (filter === 'Diabetes') return group.items.filter((i) => i.title.toLowerCase().includes('diabetes') || i.title.includes('ذیابیطس'))
    return group.items.filter((i) => i.title.toLowerCase().includes('heart') || i.title.includes('دل'))
  }, [filter, group])

  const catLabel = (c) => {
    if (c === 'all') return t.all
    if (c === 'Diabetes') return t.diabetesLabel
    if (c === 'Heart') return t.heartLabel
    return c
  }

  return (
    <section className="analysis-section">
      {lightbox && (
        <div className="lightbox" onClick={() => setLightbox(null)}>
          <div className="lightbox-inner" onClick={(e) => e.stopPropagation()}>
            <button className="lightbox-close" onClick={() => setLightbox(null)}>
              {t.close}
            </button>
            <img src={visualUrl(lightbox.file)} alt={lightbox.title} />
            <p>{lightbox.title}</p>
          </div>
        </div>
      )}

      <div className="analysis-heading">
        <div>
          <p className="eyebrow">{t.analysisLabel}</p>
          <h2>{group.title}</h2>
          <p>{group.intro}</p>
        </div>
        <span className="analysis-count-badge">{group.items.length} {t.visualsCount}</span>
      </div>

      {categories.length > 1 && (
        <div className="visual-filter-tabs">
          {categories.map((c) => (
            <button key={c} className={filter === c ? 'active' : ''} onClick={() => setFilter(c)}>
              {catLabel(c)}
            </button>
          ))}
        </div>
      )}

      <div className="visual-grid">
        {filtered.map((item) => (
          <article key={item.file} className={item.span === 'wide' ? 'visual-card wide' : 'visual-card'}>
            <div className="visual-title">
              <strong>{item.title}</strong>
              <div style={{ display: 'flex', gap: 8 }}>
                <button className="visual-zoom-btn" onClick={() => setLightbox(item)}>{t.zoomBtn}</button>
                <a href={visualUrl(item.file)} target="_blank" rel="noreferrer" className="visual-open-btn">{t.openBtn}</a>
              </div>
            </div>
            <div className="visual-img-wrap" onClick={() => setLightbox(item)}>
              <img src={visualUrl(item.file)} alt={item.title} loading="lazy" />
              <div className="visual-overlay">{t.clickZoom}</div>
            </div>
          </article>
        ))}
      </div>
    </section>
  )
}

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
  const [lang, setLang] = useState(loadLang)

  const t = translations[lang]
  const visualGroups = useMemo(() => getVisualGroups(lang, t), [lang, t])

  const baseFields = useMemo(() => (disease === 'Diabetes' ? diabetesFields : heartFields), [disease])
  const fields = useMemo(() => localizeFields(baseFields, lang, t), [baseFields, lang, t])
  const localizedPrediction = useMemo(
    () => localizePrediction(prediction, disease, t),
    [prediction, disease, t],
  )
  const [activeField, setActiveField] = useState(null)
  const [predStep, setPredStep] = useState(0)
  const [searchQuery, setSearchQuery] = useState('')
  const [expandedCard, setExpandedCard] = useState(null)
  const [historyFilter, setHistoryFilter] = useState('all')
  const [showCompareTip, setShowCompareTip] = useState(false)

  const filteredFields = useMemo(() => {
    if (!searchQuery.trim()) return fields
    return fields.filter((f) => f.label.toLowerCase().includes(searchQuery.toLowerCase()))
  }, [fields, searchQuery])

  useEffect(() => {
    document.documentElement.lang = lang
    document.documentElement.dir = lang === 'ur' ? 'rtl' : 'ltr'
    document.documentElement.setAttribute('data-lang', lang)
    saveLang(lang)
  }, [lang])

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', darkMode ? 'dark' : 'light')
    document.body.style.background = darkMode
      ? 'linear-gradient(180deg,#0f172a 0,#1e293b 100%)'
      : 'linear-gradient(180deg,#dfeef6 0,#f7fbfd 420px,#eef5f8 100%)'
  }, [darkMode])

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
    setInputs(buildDefaultInputs(baseFields))
    setPrediction(null)
    setError(null)
    setPredStep(0)
    setSearchQuery('')
  }, [baseFields])

  function updateInput(key, value) {
    setInputs((prev) => ({ ...prev, [key]: value }))
    if (prediction) { setPrediction(null); setPredStep(0) }
  }

  async function handleRunPrediction() {
    setLoading(true)
    setPrediction(null)
    setError(null)
    setPredStep(1)
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
      setError(t.apiError)
      setPredStep(0)
    } finally {
      setLoading(false)
    }
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
    if (historyFilter === 'high') return history.filter((h) => h.risk === 'high')
    if (historyFilter === 'low') return history.filter((h) => h.risk === 'low')
    return history.filter((h) => h.disease === historyFilter)
  }, [history, historyFilter])

  const histStats = useMemo(() => ({
    total: history.length,
    high: history.filter((h) => h.risk === 'high').length,
    low: history.filter((h) => h.risk === 'low').length,
    diabetes: history.filter((h) => h.disease === 'Diabetes').length,
    heart: history.filter((h) => h.disease === 'Heart Disease').length,
  }), [history])

  if (booting) {
    return (
      <main className="welcome-screen">
        <div className="welcome-bg">
          <div className="wb-blob wb-blob1" />
          <div className="wb-blob wb-blob2" />
          <div className="wb-blob wb-blob3" />
        </div>

        <section className="welcome-card">
          <div className="welcome-icon-ring">
            <div className="welcome-icon-inner">
              <svg width="38" height="38" viewBox="0 0 38 38" fill="none">
                <path d="M19 6C19 6 8 13 8 21C8 26.5 13 31 19 31C25 31 30 26.5 30 21C30 13 19 6 19 6Z" fill="url(#hg)" />
                <path d="M13 19H17L19 15L21 23L23 19H25" stroke="white" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                <defs>
                  <linearGradient id="hg" x1="8" y1="6" x2="30" y2="31" gradientUnits="userSpaceOnUse">
                    <stop stopColor="#2757d7" />
                    <stop offset="1" stopColor="#0f9f8f" />
                  </linearGradient>
                </defs>
              </svg>
            </div>
            <div className="welcome-ring r1" />
            <div className="welcome-ring r2" />
          </div>

          <p className="welcome-eyebrow">{t.aiTool}</p>
          <h1 className="welcome-title">
            {t.welcomeTitle.split('\n').map((line, idx) => (
              <React.Fragment key={idx}>
                {line}
                {idx === 0 ? <br /> : null}
              </React.Fragment>
            ))}
          </h1>
          <p className="welcome-subtitle">{t.welcomeSubtitle}</p>

          <div className="welcome-pills">
            {[`🧠 ${t.models} 3`, `📊 ${t.liveRisk}`, `🩺 ${t.heartLabel} 13`, `🩸 ${t.diabetesLabel} 8`].map((p) => (
              <span key={p} className="welcome-pill">{p}</span>
            ))}
          </div>

          <div className="boot-steps">
            {[t.initModels, t.loadingData, t.preparingDash, t.ready2].map((s, i) => (
              <div key={s} className={`boot-step ${bootStep > i ? 'done' : bootStep === i ? 'active' : ''}`}>
                <span className="boot-dot">{bootStep > i ? '✓' : bootStep === i ? '◉' : '○'}</span>
                <span>{s}</span>
                {bootStep > i && <span className="boot-check-anim">✓</span>}
              </div>
            ))}
          </div>

          <div className="welcome-progress-wrap">
            <div className="welcome-progress-bar" style={{ width: `${(bootStep / 4) * 100}%` }} />
          </div>
          <p className="welcome-progress-label">{Math.round((bootStep / 4) * 100)}% {t.loaded}</p>
        </section>
      </main>
    )
  }

  return (
    <main className="app-shell">
      {loading && (
        <div className="data-loader" role="status" aria-live="polite">
          <div className="loader-card">
            <div className="spinner" />
            <strong>{t.runningPred}</strong>
            <span>{t.analyzingProfile}</span>
            <div className="loader-models">
              {[
                { key: 'Logistic Regression', label: t.modelLogistic },
                { key: 'KNN', label: t.modelKnn },
                { key: 'Decision Tree', label: t.modelTree },
              ].map((m, i) => (
                <div key={m.key} className="loader-model-row" style={{ animationDelay: `${i * 0.3}s` }}>
                  <div className="loader-model-bar" />
                  <span>{m.label}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      <header className="app-topbar">
        <div className="brand">
          <div className="brand-mark">H</div>
          <div>
            <strong>{t.appTitle}</strong>
            <span>{t.appSubtitle}</span>
          </div>
        </div>
        <div className="topbar-right">
          <div className="topbar-chip">{t.clinicalStudio}</div>
          <button className="ghost-button" onClick={() => setLang((l) => (l === 'en' ? 'ur' : 'en'))} style={{ minHeight: 36, padding: '0 13px', fontSize: '0.86rem' }} aria-label="Toggle language">
            {t.language}
          </button>
          <button className="ghost-button" onClick={() => setDarkMode((d) => !d)} style={{ minHeight: 36, padding: '0 13px', fontSize: '0.86rem' }}>
            {darkMode ? t.lightMode : t.darkMode}
          </button>
          <div className={`api-pill ${apiStatus}`}>
            <span />
            {apiStatus === 'online' ? t.apiOnline : apiStatus === 'checking' ? t.apiChecking : t.apiOffline}
          </div>
        </div>
      </header>

      <nav className="app-nav" aria-label="App sections">
        {[
          { id: 'predict', label: t.predict, desc: t.predictDesc },
          { id: 'history', label: t.history, desc: t.historyDesc },
          { id: 'visuals', label: t.visuals, desc: t.visualsDesc },
          { id: 'metrics', label: t.metrics, desc: t.metricsDesc },
          { id: 'pca', label: t.pca, desc: t.pcaDesc },
        ].map((view) => (
          <button key={view.id} className={activeView === view.id ? 'active' : ''} onClick={() => setActiveView(view.id)}>
            <span className="nav-emoji">{view.label.split(' ')[0]}</span>
            <span className="nav-text">{view.label.split(' ').slice(1).join(' ')}</span>
            <span className="nav-desc">{view.desc}</span>
          </button>
        ))}
      </nav>

      <section className="hero-panel">
        <div>
          <p className="eyebrow">{activeView === 'predict' ? t.patientAssessment : t.notebookInsights}</p>
          <h1>
            {(activeView === 'predict' ? t.heroTitle : t.heroTitleVisuals).split('\n').map((line, idx, arr) => (
              <React.Fragment key={idx}>
                {line}
                {idx < arr.length - 1 ? <br /> : null}
              </React.Fragment>
            ))}
          </h1>
        </div>

        <div className="hero-control-card">
          <span>{t.screeningMode}</span>
          <div className="model-switch" aria-label="Disease model">
            {['Diabetes', 'Heart Disease'].map((item) => (
              <button key={item} className={disease === item ? 'active' : ''} onClick={() => setDisease(item)}>
                <span style={{ marginInlineEnd: 6 }}>{item === 'Diabetes' ? '🩸' : '❤️'}</span>
                {item === 'Diabetes' ? t.diabetesLabel : t.heartDiseaseLabel}
              </button>
            ))}
          </div>

          <div className="hero-mini-stats">
            <div><strong>3</strong><small>{t.models}</small></div>
            <div><strong>{fields.length}</strong><small>{t.inputs}</small></div>
            <div><strong>{history.length}</strong><small>{t.saved}</small></div>
            <div className={finalStatus !== 'idle' ? finalStatus : ''}>
              <strong>{prediction ? `${averageConfidence.toFixed(0)}%` : '--'}</strong>
              <small>{t.avgConf}</small>
            </div>
          </div>
        </div>
      </section>

      {activeView === 'predict' && (
        <section className="summary-strip">
          <article><span>{t.selectedModel}</span><strong>{translateDiseaseName(disease, t)}</strong></article>
          <article><span>{t.inputsLabel}</span><strong>{fields.length}</strong></article>
          <article><span>{t.averageSignal}</span><strong>{prediction ? <><AnimatedNumber value={averageConfidence} />%</> : '--'}</strong></article>
          <article className={finalStatus}><span>{t.riskStatus}</span><strong>{prediction ? (finalStatus === 'high' ? '🔴 ' + t.highLabel : '🟢 ' + t.lowLabel) : t.waiting}</strong></article>
        </section>
      )}

      {activeView === 'predict' && (
        <section className="app-grid">
          <section className="card input-card">
            <div className="card-heading">
              <div>
                <p className="eyebrow">{t.step1}</p>
                <h2>{t.step1Title}</h2>
                <p style={{ margin: '4px 0 0', fontSize: '0.85rem', color: '#64748b' }}>{t.step1Desc}</p>
              </div>
              <div style={{ display: 'flex', gap: 8 }}>
                <button className="ghost-button" onClick={() => setInputs(buildDefaultInputs(baseFields))}>{t.reset}</button>
              </div>
            </div>

            <div className="search-bar-wrap">
              <span className="search-icon">🔍</span>
              <input
                className="search-bar"
                type="text"
                placeholder={t.filterFields}
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
              />
              {searchQuery && (
                <button className="search-clear" onClick={() => setSearchQuery('')}>✕</button>
              )}
            </div>

            <StepProgress step={predStep} t={t} />
            <LiveRiskMeter inputs={inputs} fields={fields} t={t} />

            <div className="fields">
              {filteredFields.length === 0 && (
                <div className="empty-state">{t.noFieldMatch} "<strong>{searchQuery}</strong>"</div>
              )}
              {filteredFields.map((field) => {
                const isActive = activeField === field.key
                const thr = riskThresholds[field.key]
                const v = inputs[field.key]
                const isDanger = thr && !field.type && (thr.inverted ? v <= thr.danger : v >= thr.danger)
                const isWarn = thr && !field.type && (thr.inverted ? (v > thr.danger && v <= thr.warn) : (v >= thr.warn && v < thr.danger))

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
                        <InfoTooltip fieldKey={field.key} lang={lang} />
                        {!field.type && <RiskBadge value={inputs[field.key]} fieldKey={field.key} t={t} />}
                      </strong>
                      <span>{field.unit || t.chooseOne}</span>
                      {field.normal && (
                        <span style={{ fontSize: '0.76rem', color: '#059669', fontWeight: 700, marginTop: 2 }}>
                          📏 {field.normal}
                        </span>
                      )}
                    </div>

                    <div className="field-control">
                      {field.type === 'select' ? (
                        <select value={inputs[field.key]} onChange={(e) => updateInput(field.key, Number(e.target.value))}>
                          {(field.options || []).map((opt) => (
                            <option key={opt.value} value={opt.value}>{opt.label}</option>
                          ))}
                        </select>
                      ) : (
                        <div style={{ position: 'relative' }}>
                          <input
                            type="range"
                            min={field.min}
                            max={field.max}
                            step={field.step}
                            value={inputs[field.key]}
                            onChange={(e) => updateInput(field.key, Number(e.target.value))}
                            style={{ '--val': `${((inputs[field.key] - field.min) / (field.max - field.min)) * 100}%` }}
                          />
                          {thr && !field.type && (
                            <div className="range-thresholds" style={{
                              '--warn': `${((thr.warn - field.min) / (field.max - field.min)) * 100}%`,
                              '--danger': `${((thr.danger - field.min) / (field.max - field.min)) * 100}%`,
                            }}>
                              <span className="thresh-marker warn" style={{ left: 'var(--warn)' }} title={`${t.warning}: ${thr.warn}`} />
                              <span className="thresh-marker danger" style={{ left: 'var(--danger)' }} title={`${t.danger}: ${thr.danger}`} />
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
              {loading ? <><span className="btn-spinner" /> {t.predicting}</> : t.runPrediction}
            </button>
            {error && <div className="error-box">⚠️ {error}</div>}
          </section>

          <aside className="result-stack">
            <section className={`result-card ${finalStatus}`}>
              <p className="eyebrow">{t.step2}</p>
              <div style={{ marginBottom: 8 }}>
                <HeartbeatLine active={!!prediction} color={finalStatus === 'high' ? '#ef4444' : '#10b981'} />
              </div>
              <div className="result-status">
                <span>{localizedPrediction ? localizedPrediction.best_model.name : t.ready}</span>
                <strong>{localizedPrediction ? localizedPrediction.best_model.label : t.noResult}</strong>
                <p>{localizedPrediction ? localizedPrediction.best_model.explanation : t.noResultDesc}</p>
              </div>

              <div style={{ display: 'flex', alignItems: 'center', gap: 16, marginTop: 12 }}>
                <RiskGauge probability={localizedPrediction ? localizedPrediction.best_model.probability : 0} status={finalStatus} t={t} />
                {localizedPrediction && (
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                    <button onClick={() => printReport(localizedPrediction, inputs, disease, fields, t, lang)} className="result-action-btn">{t.printReport}</button>
                    <div className="result-consensus">
                      <span style={{ fontSize: 11, color: 'rgba(255,255,255,0.7)' }}>{t.modelConsensus}</span>
                      <strong style={{ fontSize: 13, color: '#fff' }}>{localizedPrediction.models.filter(m => m.prediction === localizedPrediction.best_model.prediction).length}/3 {t.agree}</strong>
                    </div>
                  </div>
                )}
              </div>

              {localizedPrediction && (
                <div className="result-tags">
                  <span>{translateDiseaseName(disease, t)}: ✓</span>
                  <span>{new Date().toLocaleDateString(lang === 'ur' ? 'ur-PK' : undefined)}</span>
                  <span>{t.confidence}: {localizedPrediction.best_model.probability.toFixed(1)}%</span>
                </div>
              )}
            </section>

            <section className="card models-card">
              <div className="card-heading compact">
                <div><p className="eyebrow">{t.step3}</p><h2>{t.step3Title}</h2></div>
                {prediction && (
                  <button className="ghost-button" style={{ fontSize: 12 }} onMouseEnter={() => setShowCompareTip(true)} onMouseLeave={() => setShowCompareTip(false)}>
                    {t.about}
                  </button>
                )}
              </div>
              {showCompareTip && <div className="compare-tip">{t.compareTip}</div>}

              <div className="model-list">
                {localizedPrediction ? localizedPrediction.models.map((item, idx) => (
                  <article key={item.name} className="model-row" style={{ animationDelay: `${idx * 0.1}s` }}>
                    <div>
                      <strong>{item.name}</strong>
                      <span>{item.prediction === 1 ? t.highRisk : t.lowRisk}</span>
                    </div>
                    <div className="bar-wrap">
                      <div className="bar">
                        <span style={{ width: `${item.probability}%`, backgroundColor: modelColors[item.nameKey] }} />
                      </div>
                      {item.name === localizedPrediction.best_model.name && <span className="best-model-tag">{t.best}</span>}
                    </div>
                    <output>{item.probability.toFixed(1)}%</output>
                  </article>
                )) : (
                  <div className="empty-state">{t.emptyModels}</div>
                )}
              </div>

              {prediction && <div className="rerun-hint">{t.rerunHint}</div>}
            </section>
          </aside>
        </section>
      )}

      {activeView === 'history' && (
        <section className="analysis-section">
          <div className="analysis-heading">
            <div>
              <p className="eyebrow">{t.patientRecords}</p>
              <h2>{t.predictionHistory}</h2>
              <p>{t.historyDesc2}</p>
            </div>
            <div style={{ display: 'flex', gap: 10, alignItems: 'center', flexWrap: 'wrap' }}>
              <span className="history-count-badge">{history.length} {t.records}</span>
              {history.length > 0 && (
                <button className="ghost-button" style={{ color: '#dc2626' }} onClick={() => { if (confirm(t.clearConfirm)) { setHistory([]); saveHistory([]) } }}>
                  {t.clearAll}
                </button>
              )}
            </div>
          </div>

          {history.length > 0 && (
            <div className="history-stats-bar">
              <div className="hstat"><strong>{histStats.total}</strong><span>{t.total}</span></div>
              <div className="hstat high"><strong style={{ color: '#dc2626' }}>{histStats.high}</strong><span>{t.highRiskLabel}</span></div>
              <div className="hstat low"><strong style={{ color: '#059669' }}>{histStats.low}</strong><span>{t.lowRiskLabel}</span></div>
              <div className="hstat"><strong style={{ color: '#2563eb' }}>{histStats.diabetes}</strong><span>{t.diabetesLabel}</span></div>
              <div className="hstat"><strong style={{ color: '#ef6f5e' }}>{histStats.heart}</strong><span>{t.heartLabel}</span></div>
            </div>
          )}

          {history.length > 0 && (
            <div className="history-filter-tabs">
              {['all', 'high', 'low', 'Diabetes', 'Heart Disease'].map((f) => (
                <button key={f} className={historyFilter === f ? 'active' : ''} onClick={() => setHistoryFilter(f)}>
                  {f === 'all' ? t.all : f === 'high' ? '🔴 ' + t.highRiskLabel : f === 'low' ? '🟢 ' + t.lowRiskLabel : f === 'Diabetes' ? t.diabetesLabel : t.heartLabel}
                </button>
              ))}
            </div>
          )}

          {filteredHistory.length === 0 ? (
            <div className="empty-state" style={{ padding: 32, textAlign: 'center', marginTop: 16 }}>
              {history.length === 0 ? t.noHistory : t.noFilter}
            </div>
          ) : (
            <div style={{ display: 'grid', gap: 12, marginTop: 16 }}>
              {filteredHistory.map((entry) => (
                <div
                  key={entry.id}
                  className={`history-card ${entry.risk} ${expandedCard === entry.id ? 'expanded' : ''}`}
                  onClick={() => setExpandedCard(expandedCard === entry.id ? null : entry.id)}
                >
                  <div className="history-card-main">
                    <div>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 6, flexWrap: 'wrap' }}>
                        <span className={`risk-pill ${entry.risk}`}>{entry.risk === 'high' ? '🔴' : '🟢'} {translateResultLabel(entry.risk === 'high' ? 1 : 0, entry.disease, t)}</span>
                        <span className="disease-pill">{entry.disease === 'Diabetes' ? t.diabetesLabel : t.heartDiseaseLabel}</span>
                        <span style={{ fontSize: 13, color: '#64748b' }}>{getModelName(entry.model, t)}</span>
                      </div>
                      <div style={{ fontSize: 13, color: '#64748b' }}>{t.date} {entry.date} &nbsp;|&nbsp; {t.confidenceLabel} <strong>{entry.confidence}%</strong></div>
                    </div>
                    <div style={{ textAlign: 'right' }}>
                      <div className={`history-confidence ${entry.risk}`}>{entry.confidence}%</div>
                      <div style={{ fontSize: 11, color: '#94a3b8' }}>{t.confidence}</div>
                      <div style={{ fontSize: 11, color: '#94a3b8', marginTop: 4 }}>{expandedCard === entry.id ? t.collapse : t.expand}</div>
                    </div>
                  </div>

                  {expandedCard === entry.id && (
                    <div className="history-expanded">
                      <div className="history-models">
                        {entry.models.map((m) => (
                          <div key={m.name} className="history-model-row">
                            <span>{getModelName(m.name, t)}</span>
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
                            <span>{getFieldLabel(k, lang)}</span>
                            <strong>{typeof v === 'number' ? (v.toFixed ? v.toFixed(1) : v) : v}</strong>
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

      {activeView === 'visuals' && <VisualGallery group={visualGroups[0]} t={t} />}
      {activeView === 'metrics' && <VisualGallery group={visualGroups[1]} t={t} />}
      {activeView === 'pca' && <VisualGallery group={visualGroups[2]} t={t} />}
    </main>
  )
}

export default App

