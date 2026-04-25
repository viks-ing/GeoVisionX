import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Satellite, 
  Zap, 
  Droplets, 
  Leaf, 
  RefreshCcw, 
  AlertTriangle,
  LogOut,
  ArrowRight,
  ArrowLeft
} from 'lucide-react';

const API_BASE = 'http://localhost:8000/api';

const MetricCard = ({ label, value }) => (
  <div className="metric-card">
    <div className="metric-label">{label}</div>
    <div className="metric-value">{value}</div>
  </div>
);

const ImagePanel = ({ label, src, scanning }) => (
  <div className="image-panel">
    <div className="panel-label">{label}</div>
    <div className="image-wrapper">
      <img src={src} alt={label} />
      {scanning && <div className="scanning-line" />}
    </div>
  </div>
);

const ProgressCard = ({ label, value, icon }) => (
  <div className="progress-card">
    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.5rem' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '0.8rem', fontWeight: 600 }}>
        {icon}
        {label}
      </div>
      <div style={{ fontSize: '0.8rem', fontWeight: 800, color: 'var(--accent-primary)' }}>
        {(((value || 0) * 100)).toFixed(1)}%
      </div>
    </div>
    <div className="progress-bar-bg">
      <div className="progress-bar-fill" style={{ width: `${(value || 0) * 100}%` }} />
    </div>
  </div>
);

const Splash = ({ onEnter }) => {
  // Memoized stars to prevent re-calculation lag
  const stars = React.useMemo(() => 
    Array.from({ length: 150 }).map((_, i) => ({
      id: i,
      top: `${Math.random() * 100}%`,
      left: `${Math.random() * 100}%`,
      size: `${1 + Math.random() * 3}px`,
      delay: `${Math.random() * 5}s`,
      duration: `${3 + Math.random() * 5}s`
    })), []);

  return (
    <motion.div 
      className="splash-container"
      initial={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      transition={{ duration: 0.8 }}
    >
      <div className="starfield">
        {stars.map(star => (
          <div 
            key={star.id}
            className="star-warp"
            style={{
              top: star.top,
              left: star.left,
              width: star.size,
              height: star.size,
              animationDelay: star.delay,
              animationDuration: star.duration
            }}
          />
        ))}
      </div>

      <div className="logo-static">
        GeoVisionX
      </div>

      <motion.button 
        className="enter-btn"
        whileHover={{ scale: 1.05 }}
        whileTap={{ scale: 0.95 }}
        onClick={onEnter}
      >
        Begin Uplink
      </motion.button>
    </motion.div>
  );
};

const ConfigView = ({ onBack, onRun, loading, sceneType, setSceneType, task, setTask, useTim, setUseTim, useScorer, setUseScorer, uploadFile, setUploadFile, onPortalReturn }) => {
  const staticStars = React.useMemo(() => 
    Array.from({ length: 50 }).map((_, i) => ({
      id: i,
      top: `${Math.random() * 100}%`,
      left: `${Math.random() * 100}%`,
    })), []);

  return (
    <motion.div 
      key="config"
      className="config-container"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      transition={{ duration: 0.5 }}
    >
      <button className="corner-back-btn" onClick={onBack}>
        <ArrowLeft size={16} /> BACK
      </button>
      <div className="starfield">
        {staticStars.map(star => (
          <div 
            key={star.id}
            className="star-static"
            style={{
              position: 'absolute',
              top: star.top,
              left: star.left,
              width: '1px',
              height: '1px',
              background: '#fff',
              opacity: 0.2
            }}
          />
        ))}
      </div>

      <div className="config-card">
        <div className="config-header">
          <h2>Mission Setup</h2>
          <p>Calibrate on-orbit logic and scene parameters</p>
        </div>

        <div className="config-grid">
          <div className="control-group">
            <label className="control-label">Scene Type</label>
            <select value={sceneType} onChange={(e) => setSceneType(e.target.value)}>
              <option>Agricultural</option>
              <option>Urban / Coastal</option>
              <option>Forest / Wildfire</option>
            </select>
          </div>

          <div className="control-group">
            <label className="control-label">Inference Task</label>
            <select value={task} onChange={(e) => setTask(e.target.value)}>
              <option>Multi-Task (All)</option>
              <option>Flood Detection</option>
              <option>Crop Stress Detection</option>
              <option>Change Detection</option>
            </select>
          </div>

          <div className="control-group" style={{ flexDirection: 'row', gap: '20px' }}>
            <label style={{ fontSize: '0.8rem', display: 'flex', alignItems: 'center', gap: '8px', cursor: 'pointer' }}>
              <input type="checkbox" checked={useTim} onChange={() => setUseTim(!useTim)} /> 
              Enable TiM
            </label>
            <label style={{ fontSize: '0.8rem', display: 'flex', alignItems: 'center', gap: '8px', cursor: 'pointer' }}>
              <input type="checkbox" checked={useScorer} onChange={() => setUseScorer(!useScorer)} /> 
              Enable Scorer
            </label>
          </div>

          <div className="control-group">
            <label className="control-label">Satellite Source</label>
            <div className="custom-file-upload">
              <input 
                type="file" 
                id="satellite-upload"
                onChange={(e) => setUploadFile(e.target.files[0])}
              />
              <label htmlFor="satellite-upload" className="file-trigger">
                {uploadFile ? uploadFile.name : 'CHOOSE MISSION FILE'}
              </label>
            </div>
          </div>

          <button className="run-btn" style={{ padding: '1.25rem' }} onClick={onRun} disabled={loading}>
            {loading ? 'CALIBRATING...' : 'INITIALIZE DOWNLINK'}
          </button>

          <button className="back-btn" onClick={onPortalReturn}>
            <LogOut size={14} />
            Return to Portal
          </button>
        </div>
      </div>
    </motion.div>
  );
};
function App() {
  const [view, setView] = useState('splash');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [sceneType, setSceneType] = useState('Agricultural');
  const [task, setTask] = useState('Multi-Task (All)');
  const [useTim, setUseTim] = useState(true);
  const [useScorer, setUseScorer] = useState(true);
  const [uploadFile, setUploadFile] = useState(null);

  const runInference = async () => {
    setLoading(true);
    const formData = new FormData();
    formData.append('scene_type', sceneType);
    formData.append('task', task);
    formData.append('use_tim', useTim);
    formData.append('use_scorer', useScorer);
    if (uploadFile) formData.append('file', uploadFile);

    try {
      const response = await axios.post(`${API_BASE}/inference`, formData);
      setResult(response.data);
      setView('telemetry');
    } catch {
      console.error("Inference failed");
      alert("System error: Satellite downlink failed.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <AnimatePresence mode="wait">
      {view === 'splash' && (
        <Splash key="splash" onEnter={() => setView('config')} />
      )}

      {view === 'config' && (
        <ConfigView 
          onBack={() => setView('splash')}
          onRun={runInference}
          loading={loading}
          sceneType={sceneType}
          setSceneType={setSceneType}
          task={task}
          setTask={setTask}
          useTim={useTim}
          setUseTim={setUseTim}
          useScorer={useScorer}
          setUseScorer={setUseScorer}
          uploadFile={uploadFile}
          setUploadFile={setUploadFile}
          onPortalReturn={() => setView('splash')}
        />
      )}
      {view === 'telemetry' && (
        <motion.div 
          key="telemetry"
          className="app-container"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.4 }}
        >
          <button className="corner-back-btn" onClick={() => setView('config')}>
            <ArrowLeft size={16} /> BACK
          </button>
          <main className="dashboard" style={{ maxWidth: '1200px', margin: '0 auto', width: '95%', padding: '20px' }}>
            <header style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '30px', width: '100%' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '15px' }}>
                <Satellite size={30} color="var(--accent-primary)" />
                <span className="brand-gradient" style={{ fontSize: '22px', whiteSpace: 'nowrap' }}>VISUAL TELEMETRY</span>
              </div>
              <button className="new-analysis-btn" style={{ padding: '10px 20px', background: 'var(--accent-primary)', color: '#000', fontSize: '12px' }} onClick={() => setView('analysis')}>
                PROCEED <ArrowRight size={14} />
              </button>
            </header>

            {result && (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '4rem' }}>
                <div className="hud-row" style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '1.5rem' }}>
                  <MetricCard label="MISSION EVENT" value={result?.prediction?.event} />
                  <MetricCard label="SYSTEM CONFIDENCE" value={`${((result?.prediction?.confidence || 0) * 100).toFixed(1)}%`} />
                  <MetricCard label="DOWNLINK GAIN" value={`${result?.bandwidth?.saving_pct?.toFixed(2)}%`} />
                  <MetricCard label="LATENCY" value={`${result?.latency_ms} ms`} />
                </div>

                <section>
                  <h3 style={{ marginBottom: '2rem', fontSize: '0.8rem', color: 'var(--accent-primary)', textTransform: 'uppercase', letterSpacing: '3px' }}>ORBITAL IMAGERY DOWNLINK</h3>
                  <div className="imagery-grid" style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '2rem' }}>
                    <ImagePanel label="RGB STREAM" src={`data:image/png;base64,${result?.images?.input}`} scanning />
                    <ImagePanel label="TiM FEATURE MAP" src={`data:image/png;base64,${result?.images?.ndvi}`} />
                    <ImagePanel label="CHANGE DELTA" src={`data:image/png;base64,${result?.images?.change}`} />
                  </div>
                </section>

                <button className="back-btn" style={{ alignSelf: 'flex-start', padding: '1rem 2.5rem' }} onClick={() => setView('config')}>
                  RE-CALIBRATE SYSTEM
                </button>
              </div>
            )}
          </main>
        </motion.div>
      )}

      {view === 'analysis' && (
        <motion.div 
          key="analysis"
          className="app-container"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.4 }}
        >
          <button className="corner-back-btn" onClick={() => setView('telemetry')}>
            <ArrowLeft size={16} /> BACK
          </button>
          <main className="dashboard" style={{ maxWidth: '1200px', margin: '0 auto', width: '95%', padding: '20px' }}>
            <header style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '30px', width: '100%' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '15px' }}>
                <Zap size={30} color="var(--accent-primary)" />
                <span className="brand-gradient" style={{ fontSize: '22px', whiteSpace: 'nowrap' }}>SEMANTIC INTELLIGENCE</span>
              </div>
              <div style={{ display: 'flex', gap: '10px' }}>
                <button className="back-btn" style={{ padding: '10px 20px', fontSize: '12px' }} onClick={() => setView('telemetry')}>
                   BACK
                </button>
                <button className="new-analysis-btn" style={{ padding: '10px 20px', background: 'var(--accent-primary)', color: '#000', fontSize: '12px' }} onClick={() => setView('config')}>
                   NEW MISSION
                </button>
              </div>
            </header>

            {result && (
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1.6fr', gap: '4rem' }}>
                {/* Left Side: Multi-Head Signals */}
                <section style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
                  <h3 style={{ fontSize: '0.8rem', color: 'var(--accent-primary)', textTransform: 'uppercase', letterSpacing: '4px' }}>MULTI-HEAD SIGNALS</h3>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
                    <ProgressCard label="Flood Risk" value={result?.multi_head?.flood || 0} icon={<Droplets size={24} />} />
                    <ProgressCard label="Vegetation Stress" value={result?.multi_head?.crop_stress || 0} icon={<Leaf size={24} />} />
                    <ProgressCard label="Activity Delta" value={result?.multi_head?.change || 0} icon={<RefreshCcw size={24} />} />
                  </div>
                </section>

                {/* Right Side: Explanation & JSON Payload */}
                <div style={{ display: 'flex', flexDirection: 'column', gap: '3.5rem' }}>
                  <div className="analysis-card" style={{ borderLeft: '5px solid var(--accent-primary)', padding: '2.5rem' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '15px', marginBottom: '2rem' }}>
                      <AlertTriangle size={30} color="var(--accent-primary)" />
                      <h4 style={{ margin: 0, fontSize: '1.1rem', textTransform: 'uppercase', letterSpacing: '3px' }}>ON-ORBIT INTELLIGENCE EXPLANATION</h4>
                    </div>
                    <p style={{ lineHeight: 2.2, fontSize: '1.3rem', color: '#e0e6ed', fontStyle: 'italic', fontWeight: 500 }}>
                      "{result?.prediction?.explanation}"
                    </p>
                  </div>

                  <div className="terminal-window" style={{ padding: '2rem' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '2rem', borderBottom: '1px solid #222', paddingBottom: '1.5rem' }}>
                      <span style={{ fontSize: '0.75rem', color: '#666', letterSpacing: '3px' }}>SEMANTIC DOWNLINK PAYLOAD [ENCRYPTED]</span>
                      <div style={{ display: 'flex', gap: '10px' }}>
                         <div style={{ width: '8px', height: '8px', borderRadius: '50%', background: '#ff4b4b' }}></div>
                         <div style={{ width: '8px', height: '8px', borderRadius: '50%', background: '#ffa500' }}></div>
                         <div style={{ width: '8px', height: '8px', borderRadius: '50%', background: '#21c354' }}></div>
                      </div>
                    </div>
                    <pre style={{ 
                      fontSize: '0.85rem', 
                      color: '#00ff41', 
                      fontFamily: 'monospace', 
                      maxHeight: '400px', 
                      overflowY: 'auto', 
                      overflowX: 'hidden', 
                      whiteSpace: 'pre-wrap', 
                      wordBreak: 'break-all',
                      padding: '1rem' 
                    }}>
                      {JSON.stringify(result?.output_json, null, 2)}
                    </pre>
                  </div>
                </div>
              </div>
            )}
          </main>
        </motion.div>
      )}
    </AnimatePresence>
  );
}

export default App;
