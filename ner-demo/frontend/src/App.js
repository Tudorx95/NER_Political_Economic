import React, { useState, useCallback, useRef, useEffect } from 'react';

// ── Config ──
const API_BASE = '/api';

const LABEL_COLORS = {
  POLITICIAN: '#FF6B6B', POLITICAL_PARTY: '#4ECDC4', POLITICAL_ORG: '#45B7D1',
  FINANCIAL_ORG: '#96CEB4', ECONOMIC_INDICATOR: '#FFEAA7', POLICY: '#DDA0DD',
  LEGISLATION: '#98D8C8', MARKET_EVENT: '#F7DC6F', CURRENCY: '#BB8FCE',
  TRADE_AGREEMENT: '#85C1E9', GPE: '#F0B27A',
};

const SAMPLE_TEXTS = {
  'United States': 'The Federal Reserve raised interest rates by 50 basis points after the FOMC meeting, citing persistent inflation pressures. President Biden signed the Inflation Reduction Act into law.',
  'United Kingdom': 'The Bank of England held rates steady as Prime Minister Starmer announced new trade agreements with the European Union. The pound sterling weakened against the dollar.',
  'Germany': 'The Bundesbank warned of recession risks as Chancellor Scholz met with NATO allies to discuss defense spending. The European Central Bank maintained its quantitative easing program.',
  'France': 'President Macron addressed the G7 summit in Paris, calling for stronger climate legislation. The CAC 40 index fell sharply after new EU tariff announcements.',
  'Japan': 'The Bank of Japan ended its negative interest rate policy as Prime Minister Kishida pushed for economic reform. The yen surged against major currencies after the policy shift.',
  'China': 'The People\'s Bank of China cut reserve requirements to stimulate growth. The Communist Party announced new trade agreements with ASEAN nations at the Belt and Road Forum.',
  'Brazil': 'President Lula signed new environmental legislation as the Brazilian real gained strength. Petrobras announced record quarterly profits amid rising commodity prices.',
  'India': 'The Reserve Bank of India maintained its benchmark rate as Prime Minister Modi launched new economic indicators for digital payments. The rupee traded near historic lows.',
  'Russia': 'The Central Bank of Russia raised rates dramatically as Western sanctions intensified. President Putin addressed the Shanghai Cooperation Organisation summit.',
  'Romania': 'The National Bank of Romania adjusted monetary policy as inflation concerns grew. Parliament debated new legislation on renewable energy subsidies.',
};

// ── Highlighted Text Component ──
function HighlightedText({ text, entities, label }) {
  if (!entities || entities.length === 0) return <span>{text}</span>;
  const sorted = [...entities].sort((a, b) => a.start - b.start);
  const parts = [];
  let lastEnd = 0;

  sorted.forEach((ent, i) => {
    if (ent.start > lastEnd) {
      parts.push(<span key={`t${i}`}>{text.slice(lastEnd, ent.start)}</span>);
    }
    const color = LABEL_COLORS[ent.label] || '#ccc';
    parts.push(
      <span key={`e${i}`} style={{
        backgroundColor: color + '33',
        borderBottom: `2px solid ${color}`,
        padding: '1px 4px', borderRadius: '3px',
        position: 'relative', cursor: 'pointer',
      }} title={`${ent.label}${ent.score ? ` (${ent.score})` : ''}`}>
        {text.slice(ent.start, ent.end)}
        <sup style={{ fontSize: '0.6em', color, fontWeight: 700, marginLeft: 2 }}>
          {ent.label}
        </sup>
      </span>
    );
    lastEnd = ent.end;
  });
  if (lastEnd < text.length) parts.push(<span key="last">{text.slice(lastEnd)}</span>);
  return <>{parts}</>;
}

// ── Entity Table ──
function EntityTable({ entities, modelName }) {
  if (!entities || entities.length === 0) {
    return <p style={{ color: '#999', fontStyle: 'italic' }}>Nu s-au detectat entitati.</p>;
  }
  return (
    <div style={{ overflowX: 'auto' }}>
      <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.85rem' }}>
        <thead>
          <tr style={{ borderBottom: '2px solid #333' }}>
            <th style={thStyle}>Entity</th>
            <th style={thStyle}>Label</th>
            {modelName === 'gliner' && <th style={thStyle}>Score</th>}
          </tr>
        </thead>
        <tbody>
          {entities.map((e, i) => (
            <tr key={i} style={{ borderBottom: '1px solid #222' }}>
              <td style={tdStyle}>{e.text}</td>
              <td style={tdStyle}>
                <span style={{
                  backgroundColor: (LABEL_COLORS[e.label] || '#666') + '33',
                  color: LABEL_COLORS[e.label] || '#fff',
                  padding: '2px 8px', borderRadius: '12px',
                  fontSize: '0.75rem', fontWeight: 600,
                }}>{e.label}</span>
              </td>
              {modelName === 'gliner' && <td style={tdStyle}>{e.score || '—'}</td>}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

const thStyle = { textAlign: 'left', padding: '8px 12px', color: '#aaa', fontWeight: 500 };
const tdStyle = { padding: '8px 12px', color: '#ddd' };

// ── Main App ──
export default function App() {
  const [selectedCountry, setSelectedCountry] = useState(null);
  const [inputText, setInputText] = useState('');
  const [extraLabels, setExtraLabels] = useState('');
  const [threshold, setThreshold] = useState(0.5);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [showPanel, setShowPanel] = useState(false);
  const [backendStatus, setBackendStatus] = useState(null);

  // Check backend health
  useEffect(() => {
    fetch(`${API_BASE}/health`)
      .then(r => r.json())
      .then(setBackendStatus)
      .catch(() => setBackendStatus({ status: 'offline', gliner_loaded: false, spacy_loaded: false }));
  }, []);

  const selectCountry = useCallback((country) => {
    setSelectedCountry(country);
    setInputText(SAMPLE_TEXTS[country] || `Enter text about ${country}...`);
    setShowPanel(true);
    setResults(null);
  }, []);

  const runInference = useCallback(async () => {
    if (!inputText.trim()) return;
    setLoading(true);
    setResults(null);
    try {
      const extra = extraLabels.split(',').map(s => s.trim()).filter(Boolean);
      const res = await fetch(`${API_BASE}/predict_both`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          text: inputText,
          model: 'gliner',
          threshold,
          extra_labels: extra.length > 0 ? extra : null,
        }),
      });
      const data = await res.json();
      setResults(data);
    } catch (err) {
      console.error(err);
      setResults({ error: err.message });
    }
    setLoading(false);
  }, [inputText, extraLabels, threshold]);

  const countries = Object.keys(SAMPLE_TEXTS);

  return (
    <div style={containerStyle}>
      {/* Header */}
      <header style={headerStyle}>
        <h1 style={{ margin: 0, fontSize: '1.8rem', fontWeight: 700 }}>
          <span style={{ color: '#4ECDC4' }}>NER</span> Entity Compare
        </h1>
        <p style={{ margin: '4px 0 0', color: '#888', fontSize: '0.9rem' }}>
          Political & Economic Named Entity Recognition — GLiNER vs spaCy
        </p>
        {backendStatus && (
          <div style={{ display: 'flex', gap: 12, marginTop: 8 }}>
            <StatusBadge label="GLiNER" ok={backendStatus.gliner_loaded} />
            <StatusBadge label="spaCy" ok={backendStatus.spacy_loaded} />
          </div>
        )}
      </header>

      <div style={mainLayout}>
        {/* Left: Globe / Country Selector */}
        <div style={globeSection}>
          <div style={globeContainer}>
            <div style={globeVisual}>
              <div style={globeInner}>
                {/* Animated Globe CSS */}
                <div style={globeSphere}></div>
                <div style={globeGrid}></div>
              </div>
            </div>
            <h3 style={{ color: '#4ECDC4', margin: '16px 0 8px', textAlign: 'center' }}>
              Select a Country
            </h3>
            <div style={countryGrid}>
              {countries.map(c => (
                <button
                  key={c}
                  onClick={() => selectCountry(c)}
                  style={{
                    ...countryBtn,
                    backgroundColor: selectedCountry === c ? '#4ECDC4' : 'rgba(255,255,255,0.05)',
                    color: selectedCountry === c ? '#000' : '#ddd',
                    borderColor: selectedCountry === c ? '#4ECDC4' : 'rgba(255,255,255,0.1)',
                  }}
                >
                  {c}
                </button>
              ))}
            </div>
          </div>

          {/* Label Legend */}
          <div style={legendBox}>
            <h4 style={{ margin: '0 0 8px', color: '#aaa' }}>NER Labels</h4>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
              {Object.entries(LABEL_COLORS).map(([label, color]) => (
                <span key={label} style={{
                  fontSize: '0.7rem', padding: '3px 8px', borderRadius: 10,
                  backgroundColor: color + '22', color: color,
                  border: `1px solid ${color}44`,
                }}>{label}</span>
              ))}
            </div>
          </div>
        </div>

        {/* Right: Input + Results */}
        <div style={panelSection}>
          {!showPanel ? (
            <div style={placeholderStyle}>
              <p style={{ fontSize: '1.2rem', color: '#555' }}>
                ← Selecteaza o tara pentru a incepe
              </p>
              <p style={{ color: '#444', fontSize: '0.9rem' }}>
                Textul generat automat va putea fi editat sau inlocuit cu textul tau.
              </p>
            </div>
          ) : (
            <>
              {/* Input Area */}
              <div style={cardStyle}>
                <h3 style={{ color: '#4ECDC4', margin: '0 0 12px' }}>
                  {selectedCountry && `📍 ${selectedCountry}`} — Input Text
                </h3>
                <textarea
                  value={inputText}
                  onChange={e => setInputText(e.target.value)}
                  rows={5}
                  style={textareaStyle}
                  placeholder="Introdu text in engleza..."
                />

                {/* Zero-shot labels */}
                <div style={{ marginTop: 12 }}>
                  <label style={{ color: '#888', fontSize: '0.8rem' }}>
                    Zero-shot extra labels (GLiNER only, comma-separated):
                  </label>
                  <input
                    value={extraLabels}
                    onChange={e => setExtraLabels(e.target.value)}
                    placeholder="e.g. SANCTION, ELECTION, SUMMIT"
                    style={inputStyle}
                  />
                </div>

                {/* Threshold */}
                <div style={{ marginTop: 8, display: 'flex', alignItems: 'center', gap: 12 }}>
                  <label style={{ color: '#888', fontSize: '0.8rem' }}>
                    Threshold: {threshold.toFixed(2)}
                  </label>
                  <input
                    type="range" min="0.1" max="0.95" step="0.05"
                    value={threshold}
                    onChange={e => setThreshold(parseFloat(e.target.value))}
                    style={{ flex: 1 }}
                  />
                </div>

                <button onClick={runInference} disabled={loading} style={runBtnStyle}>
                  {loading ? '⏳ Running inference...' : '🚀 Compare Models'}
                </button>
              </div>

              {/* Results */}
              {results && !results.error && (
                <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap' }}>
                  {/* GLiNER Results */}
                  <div style={{ ...resultCard, flex: 1, minWidth: 300 }}>
                    <h3 style={{ color: '#FF6B6B', margin: '0 0 12px' }}>
                      🔍 GLiNER
                      <span style={badgeStyle}>fine-tuned</span>
                    </h3>
                    {results.gliner?.error ? (
                      <p style={{ color: '#FF6B6B' }}>{results.gliner.error}</p>
                    ) : (
                      <>
                        <div style={highlightBox}>
                          <HighlightedText
                            text={inputText}
                            entities={results.gliner?.entities || []}
                          />
                        </div>
                        <EntityTable
                          entities={results.gliner?.entities || []}
                          modelName="gliner"
                        />
                      </>
                    )}
                  </div>

                  {/* spaCy Results */}
                  <div style={{ ...resultCard, flex: 1, minWidth: 300 }}>
                    <h3 style={{ color: '#45B7D1', margin: '0 0 12px' }}>
                      🏷️ spaCy
                      <span style={badgeStyle}>fine-tuned</span>
                    </h3>
                    {results.spacy?.error ? (
                      <p style={{ color: '#FF6B6B' }}>{results.spacy.error}</p>
                    ) : (
                      <>
                        <div style={highlightBox}>
                          <HighlightedText
                            text={inputText}
                            entities={results.spacy?.entities || []}
                          />
                        </div>
                        <EntityTable
                          entities={results.spacy?.entities || []}
                          modelName="spacy"
                        />
                      </>
                    )}
                  </div>
                </div>
              )}

              {results?.error && (
                <div style={{ ...cardStyle, borderColor: '#FF6B6B' }}>
                  <p style={{ color: '#FF6B6B' }}>Error: {results.error}</p>
                </div>
              )}
            </>
          )}
        </div>
      </div>

      {/* Footer */}
      <footer style={footerStyle}>
        <p>
          Lepadatu Tudor — Academia Tehnica Militara "Ferdinand I", 2026 |
          Dataset: <a href="https://huggingface.co/datasets/Tudorx95/NER_Political_Economic"
            style={{ color: '#4ECDC4' }} target="_blank" rel="noreferrer">HuggingFace</a>
        </p>
      </footer>
    </div>
  );
}

function StatusBadge({ label, ok }) {
  return (
    <span style={{
      fontSize: '0.75rem', padding: '3px 10px', borderRadius: 12,
      backgroundColor: ok ? '#2d6a4f22' : '#6b000022',
      color: ok ? '#4ECDC4' : '#FF6B6B',
      border: `1px solid ${ok ? '#4ECDC444' : '#FF6B6B44'}`,
    }}>
      {ok ? '●' : '○'} {label}
    </span>
  );
}

// ── Styles ──
const containerStyle = {
  minHeight: '100vh', backgroundColor: '#0a0a0f',
  fontFamily: "'Inter', sans-serif", color: '#eee',
  display: 'flex', flexDirection: 'column',
};
const headerStyle = {
  padding: '20px 32px', borderBottom: '1px solid #1a1a2e',
  background: 'linear-gradient(180deg, #0f0f1a 0%, #0a0a0f 100%)',
};
const mainLayout = {
  display: 'flex', flex: 1, gap: 0,
  flexWrap: 'wrap',
};
const globeSection = {
  width: 340, padding: 24,
  borderRight: '1px solid #1a1a2e',
  display: 'flex', flexDirection: 'column', gap: 16,
};
const globeContainer = {
  background: 'rgba(255,255,255,0.02)', borderRadius: 16,
  padding: 20, border: '1px solid #1a1a2e',
};
const globeVisual = {
  width: '100%', aspectRatio: '1', display: 'flex',
  alignItems: 'center', justifyContent: 'center',
};
const globeInner = {
  width: 200, height: 200, position: 'relative',
};
const globeSphere = {
  position: 'absolute', inset: 0, borderRadius: '50%',
  background: 'radial-gradient(circle at 30% 30%, #1a3a4a, #0a1520)',
  border: '2px solid #4ECDC444',
  animation: 'pulse 4s ease-in-out infinite',
  boxShadow: '0 0 60px #4ECDC422, inset 0 0 40px #4ECDC411',
};
const globeGrid = {
  position: 'absolute', inset: 10, borderRadius: '50%',
  border: '1px solid #4ECDC422',
  background: `
    repeating-conic-gradient(#4ECDC411 0deg 2deg, transparent 2deg 30deg),
    repeating-linear-gradient(0deg, transparent, transparent 25%, #4ECDC411 25%, #4ECDC411 26%)
  `,
  animation: 'spin 20s linear infinite',
};
const countryGrid = {
  display: 'grid', gridTemplateColumns: '1fr 1fr',
  gap: 6, marginTop: 8,
};
const countryBtn = {
  padding: '8px 12px', borderRadius: 8, border: '1px solid',
  cursor: 'pointer', fontSize: '0.8rem', fontWeight: 500,
  transition: 'all 0.2s',
};
const legendBox = {
  background: 'rgba(255,255,255,0.02)', borderRadius: 12,
  padding: 16, border: '1px solid #1a1a2e',
};
const panelSection = {
  flex: 1, padding: 24, minWidth: 0,
  display: 'flex', flexDirection: 'column', gap: 16,
};
const placeholderStyle = {
  flex: 1, display: 'flex', flexDirection: 'column',
  alignItems: 'center', justifyContent: 'center',
};
const cardStyle = {
  background: 'rgba(255,255,255,0.03)', borderRadius: 16,
  padding: 20, border: '1px solid #1a1a2e',
};
const textareaStyle = {
  width: '100%', background: '#0f0f1a', border: '1px solid #1a1a2e',
  borderRadius: 10, padding: 14, color: '#eee', fontSize: '0.9rem',
  fontFamily: 'Inter, sans-serif', resize: 'vertical', outline: 'none',
  boxSizing: 'border-box',
};
const inputStyle = {
  width: '100%', background: '#0f0f1a', border: '1px solid #1a1a2e',
  borderRadius: 8, padding: '8px 12px', color: '#eee', fontSize: '0.85rem',
  fontFamily: 'Inter, sans-serif', outline: 'none', marginTop: 4,
  boxSizing: 'border-box',
};
const runBtnStyle = {
  marginTop: 16, width: '100%', padding: '14px 0',
  background: 'linear-gradient(135deg, #4ECDC4, #45B7D1)',
  border: 'none', borderRadius: 10, color: '#000',
  fontSize: '1rem', fontWeight: 700, cursor: 'pointer',
  transition: 'transform 0.1s',
};
const resultCard = {
  background: 'rgba(255,255,255,0.03)', borderRadius: 16,
  padding: 20, border: '1px solid #1a1a2e',
};
const highlightBox = {
  background: '#0f0f1a', borderRadius: 10, padding: 16,
  lineHeight: 1.8, fontSize: '0.9rem', marginBottom: 16,
  border: '1px solid #1a1a2e',
};
const badgeStyle = {
  fontSize: '0.65rem', marginLeft: 8, padding: '2px 8px',
  borderRadius: 8, backgroundColor: 'rgba(255,255,255,0.08)',
  color: '#888', verticalAlign: 'middle',
};
const footerStyle = {
  padding: '16px 32px', borderTop: '1px solid #1a1a2e',
  textAlign: 'center', color: '#555', fontSize: '0.8rem',
};

// Inject CSS animations
const styleSheet = document.createElement('style');
styleSheet.textContent = `
  @keyframes pulse {
    0%, 100% { box-shadow: 0 0 60px #4ECDC422, inset 0 0 40px #4ECDC411; }
    50% { box-shadow: 0 0 80px #4ECDC444, inset 0 0 60px #4ECDC422; }
  }
  @keyframes spin {
    from { transform: rotate(0deg); }
    to { transform: rotate(360deg); }
  }
  * { box-sizing: border-box; }
  body { margin: 0; padding: 0; }
  textarea:focus, input:focus { border-color: #4ECDC4 !important; }
  button:hover:not(:disabled) { transform: scale(1.02); opacity: 0.95; }
  button:disabled { opacity: 0.5; cursor: not-allowed; }
  ::-webkit-scrollbar { width: 6px; }
  ::-webkit-scrollbar-track { background: #0a0a0f; }
  ::-webkit-scrollbar-thumb { background: #333; border-radius: 3px; }
`;
document.head.appendChild(styleSheet);
