import { useEffect, useState } from 'react'

const API_BASE = ''

const METRIC_HELP = {
  MAE: 'Mean absolute error: average dollar difference between predicted and actual sale prices on the test set. Lower is better.',
  MAPE: 'Mean absolute percentage error: average percent error versus actual prices. Lower is better.',
  R2: 'R² (coefficient of determination): how much of the variation in price the model explains, from 0 to 1 (higher is better).',
}

function MetricWithTip({ label, value, format, helpKey }) {
  const h = METRIC_HELP[helpKey]
  return (
    <span className="metric-term" title={h}>
      <abbr className="metric-abbr">{label}</abbr>
      <span className="metric-value">{format(value)}</span>
    </span>
  )
}

function trainProgressPercent(state) {
  if (!state || state.status === 'idle') return 0
  if (state.status === 'complete') return 100
  if (state.status === 'error') return 100
  const ph = state.phase
  if (ph === 'listing') return 12
  if (ph === 'fetching' && state.total > 0) {
    return 20 + (state.fetched / state.total) * 45
  }
  if (ph === 'fetching') return 22
  if (ph === 'cleaning') return 68
  if (ph === 'loading_db') return 78
  if (ph === 'training') return 88
  return 8
}

function phaseLabel(state) {
  if (!state) return ''
  if (state.status === 'complete') return 'Complete'
  if (state.status === 'error') return 'Error'
  const ph = state.phase
  if (ph === 'listing') return 'Discovering auctions'
  if (ph === 'fetching') return 'Fetching auction pages'
  if (ph === 'cleaning') return 'Cleaning data'
  if (ph === 'loading_db') return 'Loading database'
  if (ph === 'training') return 'Training models'
  if (ph === 'starting') return 'Starting'
  return state.phase || ''
}

function isValidSearchPath(value) {
  return /^[a-z0-9-]+\/[a-z0-9-]+$/i.test((value || '').trim())
}

function isValidCnbAuctionUrl(value) {
  try {
    const u = new URL((value || '').trim())
    const host = u.hostname.toLowerCase()
    const hostOk = host === 'carsandbids.com' || host === 'www.carsandbids.com'
    return hostOk && u.pathname.startsWith('/auctions/')
  } catch {
    return false
  }
}

export default function App() {
  const [models, setModels] = useState([])
  const [summary, setSummary] = useState(null)
  const [charts, setCharts] = useState([])

  const [searchPath, setSearchPath] = useState('')
  const [trainMessage, setTrainMessage] = useState('')
  const [trainState, setTrainState] = useState(null)
  const [pollTrain, setPollTrain] = useState(false)

  const [auctionUrl, setAuctionUrl] = useState('')
  const [predictResult, setPredictResult] = useState(null)
  const [isPredicting, setIsPredicting] = useState(false)
  const [isPredictModalOpen, setIsPredictModalOpen] = useState(false)
  const [isFullRetraining, setIsFullRetraining] = useState(false)

  const [selectedModel, setSelectedModel] = useState(null)
  const [modelDetails, setModelDetails] = useState(null)
  const [isLoadingDetails, setIsLoadingDetails] = useState(false)
  const [lastUpdated, setLastUpdated] = useState(Date.now())
  const [chartModal, setChartModal] = useState(null)

  const fetchModels = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/models`)
      const data = await res.json()
      setModels(data.models || [])
      setSummary(data.summary || null)
      setCharts(data.charts || [])
      setLastUpdated(Date.now())
    } catch (e) {
      console.error('Error fetching models:', e)
    }
  }

  useEffect(() => {
    fetchModels()
  }, [])

  useEffect(() => {
    if (!pollTrain) return undefined
    let cancelled = false
    const tick = async () => {
      try {
        const res = await fetch(`${API_BASE}/api/train/status`)
        const st = await res.json()
        if (cancelled) return
        setTrainState(st)
        setTrainMessage(st.message || '')
        if (st.status === 'complete' || st.status === 'error') {
          setPollTrain(false)
          setIsFullRetraining(false)
          fetchModels()
        }
      } catch (e) {
        if (!cancelled) console.error(e)
      }
    }
    tick()
    const id = setInterval(tick, 1200)
    return () => {
      cancelled = true
      clearInterval(id)
    }
  }, [pollTrain])

  const handleTrain = async (e) => {
    e.preventDefault()
    if (!searchPath) return
    if (!isValidSearchPath(searchPath)) {
      alert('Invalid search path. Use format make/model-slug (example: bmw/e46-m3).')
      return
    }
    setTrainMessage('Starting…')
    setTrainState({ status: 'running', phase: 'starting', message: 'Starting…' })
    try {
      const res = await fetch(`${API_BASE}/api/train`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ search_path: searchPath }),
      })
      if (res.status === 409) {
        const err = await res.json().catch(() => ({}))
        setTrainMessage(err.detail || 'Training already running.')
        setPollTrain(true)
        return
      }
      if (!res.ok) {
        const err = await res.json().catch(() => ({}))
        throw new Error(err.detail || 'Training request failed')
      }
      await res.json()
      setPollTrain(true)
    } catch (err) {
      setTrainMessage(err.message || 'Error triggering training.')
      setTrainState((s) => ({ ...(s || {}), status: 'error', message: err.message }))
    }
  }

  const handleFullRetrain = async () => {
    setIsFullRetraining(true)
    setTrainMessage('Starting full retrain…')
    setTrainState({ status: 'running', phase: 'training', message: 'Starting full retrain…' })
    try {
      const res = await fetch(`${API_BASE}/api/train/full`, { method: 'POST' })
      if (res.status === 409) {
        const err = await res.json().catch(() => ({}))
        setTrainMessage(err.detail || 'Training already running.')
        setPollTrain(true)
        return
      }
      if (!res.ok) {
        const err = await res.json().catch(() => ({}))
        throw new Error(err.detail || 'Full retrain request failed')
      }
      await res.json()
      setPollTrain(true)
    } catch (err) {
      setTrainMessage(err.message || 'Error triggering full retrain.')
      setTrainState((s) => ({ ...(s || {}), status: 'error', message: err.message }))
      setIsFullRetraining(false)
    }
  }

  const handlePredict = async (e) => {
    e.preventDefault()
    if (!auctionUrl) return
    if (!isValidCnbAuctionUrl(auctionUrl)) {
      alert('Invalid URL. Use a Cars & Bids auction URL like https://carsandbids.com/auctions/...')
      return
    }
    setIsPredicting(true)
    setPredictResult(null)
    try {
      const res = await fetch(`${API_BASE}/api/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ url: auctionUrl }),
      })
      if (!res.ok) {
        const error = await res.json()
        throw new Error(error.detail || 'Prediction failed')
      }
      const data = await res.json()
      setPredictResult(data)
      setIsPredictModalOpen(true)
    } catch (err) {
      alert(err.message)
    } finally {
      setIsPredicting(false)
    }
  }

  const openModelDetails = async (model) => {
    setSelectedModel(model)
    setIsLoadingDetails(true)
    setModelDetails(null)
    try {
      const res = await fetch(`${API_BASE}/api/models/${encodeURIComponent(model.id)}`)
      const data = await res.json()
      setModelDetails(data)
    } catch (err) {
      console.error(err)
    } finally {
      setIsLoadingDetails(false)
    }
  }

  const deleteModel = async (model) => {
    const ok = window.confirm(`Delete model "${model.name}"? This removes the saved model + related files.`)
    if (!ok) return
    try {
      const res = await fetch(`${API_BASE}/api/models/${encodeURIComponent(model.id)}`, { method: 'DELETE' })
      if (!res.ok) {
        const err = await res.json().catch(() => ({}))
        throw new Error(err.detail || 'Delete failed')
      }
      await res.json()
      if (selectedModel?.id === model.id) setSelectedModel(null)
      fetchModels()
    } catch (err) {
      alert(err.message)
    }
  }

  const pct = trainProgressPercent(trainState)
  const busyTraining = pollTrain || trainState?.status === 'running'

  return (
    <div className="app-container">
      <header>
        <h1>Car Auction Price Predictor</h1>
        <p className="subtitle">Machine learning powered estimations for enthusiast vehicles</p>
      </header>

      <div className="grid-container">
        <div className="glass-panel">
          <h2>Train new model</h2>
          <p className="subtitle panel-hint">
            Enter a Cars &amp; Bids search path (e.g. <code>mazda/nd-miata</code>) to scrape, load new rows, train, and refresh charts.
          </p>
          <form onSubmit={handleTrain}>
            <input
              type="text"
              placeholder="e.g. bmw/e46-m3"
              value={searchPath}
              onChange={(e) => setSearchPath(e.target.value)}
            />
            <button type="submit" disabled={busyTraining || !searchPath}>
              {busyTraining ? (
                <>
                  <span className="loader" /> Training…
                </>
              ) : (
                'Start training'
              )}
            </button>
            {(busyTraining || trainMessage) && (
              <div className="train-progress-wrap">
                <div className="train-progress-meta">
                  <span className="train-phase">{phaseLabel(trainState)}</span>
                  {trainState?.phase === 'fetching' && trainState.total > 0 && (
                    <span className="train-count">
                      {trainState.fetched}/{trainState.total} auctions
                    </span>
                  )}
                </div>
                <div className="train-progress-bar">
                  <div
                    className="train-progress-fill"
                    style={{
                      width: `${Math.min(100, pct)}%`,
                      opacity: trainState?.phase === 'training' ? 0.85 : 1,
                    }}
                  />
                </div>
                {trainMessage && (
                  <p
                    className={`train-msg ${
                      trainState?.status === 'error' ? 'train-msg-err' : ''
                    }`}
                  >
                    {trainMessage}
                  </p>
                )}
              </div>
            )}
          </form>
        </div>

        <div className="glass-panel">
          <h2>Predict auction price</h2>
          <p className="subtitle panel-hint">
            Paste a Cars &amp; Bids auction URL.
          </p>
          <form onSubmit={handlePredict}>
            <input
              type="text"
              placeholder="https://carsandbids.com/auctions/..."
              value={auctionUrl}
              onChange={(e) => setAuctionUrl(e.target.value)}
            />
            <button type="submit" disabled={isPredicting || !auctionUrl}>
              {isPredicting ? (
                <>
                  <span className="loader" /> Predicting…
                </>
              ) : (
                'Predict sale price'
              )}
            </button>
          </form>
        </div>
      </div>

      {charts.length > 0 && (
        <div className="glass-panel">
          <h2>Model visualizations</h2>
          <div className="charts-grid">
            {charts.map((c) => (
              <figure
                key={c.id}
                className="chart-card chart-clickable"
                role="presentation"
                onClick={() => setChartModal(c)}
              >
                <figcaption>{c.title}</figcaption>
                <img src={`${c.url}?t=${lastUpdated}`} alt={c.title} className="chart-img" />
              </figure>
            ))}
          </div>
        </div>
      )}

      <div className="glass-panel">
        <div className="panel-header-row">
          <h2>Trained models</h2>
          <div className="btn-row">
            <button type="button" className="btn-secondary" onClick={fetchModels}>
              Refresh
            </button>
            <button
              type="button"
              className="btn-danger"
              disabled={busyTraining}
              onClick={handleFullRetrain}
              title="Retrain every model using existing database rows"
            >
              {isFullRetraining ? (
                <>
                  <span className="loader" /> Full retraining…
                </>
              ) : (
                'Full retrain (all models)'
              )}
            </button>
          </div>
        </div>

        {summary && summary.MAE != null && (
          <div className="aggregate-banner">
            <strong>Aggregate test performance:</strong>{' '}
            <MetricWithTip
              label="MAE"
              helpKey="MAE"
              value={summary.MAE}
              format={(v) => `$${Number(v).toFixed(2)}`}
            />
            <span className="sep">|</span>
            <MetricWithTip
              label="MAPE"
              helpKey="MAPE"
              value={summary.MAPE}
              format={(v) => `${(Number(v) * 100).toFixed(2)}%`}
            />
            <span className="sep">|</span>
            <MetricWithTip
              label="R²"
              helpKey="R2"
              value={summary.R2}
              format={(v) => Number(v).toFixed(3)}
            />
          </div>
        )}

        {models.length === 0 ? (
          <p>No models trained yet. Train one above.</p>
        ) : (
          <div className="models-grid">
            {models.map((m) => (
              <div
                key={m.id}
                className="model-card"
                role="button"
                tabIndex={0}
                onClick={() => openModelDetails(m)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' || e.key === ' ') {
                    e.preventDefault()
                    openModelDetails(m)
                  }
                }}
              >
                <button
                  type="button"
                  className="model-delete"
                  onClick={(e) => {
                    e.stopPropagation()
                    deleteModel(m)
                  }}
                  title="Delete this trained model"
                >
                  Delete
                </button>
                {m.image_url && (
                  <img src={m.image_url} alt={m.name} className="model-hero" />
                )}
                <h3>{m.name}</h3>
                {m.metrics ? (
                  <div className="model-metrics">
                    <MetricWithTip
                      label="MAE"
                      helpKey="MAE"
                      value={m.metrics.MAE}
                      format={(v) => `$${Number(v).toFixed(0)}`}
                    />
                    <MetricWithTip
                      label="MAPE"
                      helpKey="MAPE"
                      value={m.metrics.MAPE}
                      format={(v) => `${(Number(v) * 100).toFixed(1)}%`}
                    />
                    <MetricWithTip
                      label="R²"
                      helpKey="R2"
                      value={m.metrics.R2}
                      format={(v) => Number(v).toFixed(2)}
                    />
                  </div>
                ) : (
                  <p className="stats-muted">No test metrics yet (train to refresh).</p>
                )}
                <span className="badge">View details</span>
              </div>
            ))}
          </div>
        )}
      </div>

      {selectedModel && (
        <div
          className="modal-overlay"
          role="presentation"
          onClick={(e) => {
            if (e.target.className === 'modal-overlay') setSelectedModel(null)
          }}
        >
          <div className="modal-content modal-wide">
            <button
              type="button"
              className="close-btn"
              onClick={() => setSelectedModel(null)}
            >
              ×
            </button>
            <h2>{selectedModel.name}</h2>
            <div className="modal-actions">
              <button
                type="button"
                className="btn-danger"
                onClick={() => deleteModel(selectedModel)}
              >
                Delete model
              </button>
            </div>
            {selectedModel.metrics && (
              <p className="subtitle modal-metrics">
                <MetricWithTip
                  label="MAE"
                  helpKey="MAE"
                  value={selectedModel.metrics.MAE}
                  format={(v) => `$${Number(v).toFixed(2)}`}
                />
                <span className="sep">|</span>
                <MetricWithTip
                  label="MAPE"
                  helpKey="MAPE"
                  value={selectedModel.metrics.MAPE}
                  format={(v) => `${(Number(v) * 100).toFixed(2)}%`}
                />
                <span className="sep">|</span>
                <MetricWithTip
                  label="R²"
                  helpKey="R2"
                  value={selectedModel.metrics.R2}
                  format={(v) => Number(v).toFixed(3)}
                />
              </p>
            )}
            <p className="subtitle">Test-set accuracy for this model</p>
            {modelDetails?.accuracy_chart_url && (
              <a
                href={modelDetails.accuracy_chart_url}
                target="_blank"
                rel="noreferrer"
                className="modal-chart-link"
              >
                <img
                  src={`${modelDetails.accuracy_chart_url}?t=${lastUpdated}`}
                  alt={`Accuracy chart for ${selectedModel.name}`}
                  className="modal-chart"
                />
              </a>
            )}
            <p className="subtitle feature-section-title">
              Feature importances
            </p>

            {isLoadingDetails ? (
              <div className="modal-loading">
                <span
                  className="loader"
                  style={{
                    borderColor: 'var(--accent)',
                    borderTopColor: 'transparent',
                  }}
                />
              </div>
            ) : modelDetails && modelDetails.importances?.length > 0 ? (
              <div className="feature-bar-container">
                {modelDetails.importances.map((imp, idx) => {
                  const maxImp = modelDetails.importances[0].Importance
                  const pctBar = maxImp > 0 ? (imp.Importance / maxImp) * 100 : 0
                  const label = imp.Feature.replace('cat__', '').replace('num__', '')
                  return (
                    <div className="feature-bar" key={`${label}-${idx}`}>
                      <div className="feature-name" title={label}>
                        {label}
                      </div>
                      <div className="bar-wrapper">
                        <div className="bar-fill" style={{ width: `${pctBar}%` }} />
                      </div>
                      <div className="feature-val">
                        {(imp.Importance * 100).toFixed(2)}%
                      </div>
                    </div>
                  )
                })}
              </div>
            ) : (
              <p className="muted">No feature importance rows above the 0.1% threshold.</p>
            )}
          </div>
        </div>
      )}

      {chartModal && (
        <div
          className="modal-overlay"
          role="presentation"
          onClick={(e) => {
            if (e.target.className === 'modal-overlay') setChartModal(null)
          }}
        >
          <div className="modal-content modal-xxl">
            <button type="button" className="close-btn" onClick={() => setChartModal(null)}>
              ×
            </button>
            <h2>{chartModal.title}</h2>
            <img src={`${chartModal.url}?t=${lastUpdated}`} alt={chartModal.title} className="modal-chart modal-chart-large" />
          </div>
        </div>
      )}

      {isPredictModalOpen && predictResult && (
        <div
          className="modal-overlay"
          role="presentation"
          onClick={(e) => {
            if (e.target.className === 'modal-overlay') setIsPredictModalOpen(false)
          }}
        >
          <div className="modal-content modal-wide">
            <button type="button" className="close-btn" onClick={() => setIsPredictModalOpen(false)}>
              ×
            </button>
            <h2>Predicted auction price</h2>
            <div className="prediction-result">
              <p>Estimated value</p>
              <div className="prediction-price">
                $
                {predictResult.prediction.toLocaleString(undefined, {
                  minimumFractionDigits: 2,
                  maximumFractionDigits: 2,
                })}
              </div>
              {predictResult.details?.image_url && (
                <img className="prediction-hero" src={predictResult.details.image_url} alt="" />
              )}
              {predictResult.details_display?.fields?.length > 0 && (
                <dl className="prediction-details">
                  {predictResult.details_display.fields.map((row) => (
                    <div key={row.key} className="prediction-dl-row">
                      <dt>{row.label}</dt>
                      <dd>{row.value}</dd>
                    </div>
                  ))}
                </dl>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
