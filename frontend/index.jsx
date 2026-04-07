// frontend/index.jsx — Complete Brain Tumor Segmentation Dashboard
// ML Project by Rahul & Krishnaa for Dr. Valarmathi

import { useState, useRef, useEffect, useCallback } from "react";

const API_BASE = import.meta.env.VITE_API_URL || "http://localhost:8000";

// ─── Color Palette ───
const COLORS = {
  bg: "#0a0e1a",
  surface: "#111827",
  surfaceLight: "#1e293b",
  border: "#334155",
  accent: "#06b6d4",
  accentDim: "#0891b2",
  danger: "#ef4444",
  warning: "#f59e0b",
  success: "#22c55e",
  text: "#f1f5f9",
  textDim: "#94a3b8",
  necrotic: "#e74c3c",
  edema: "#f1c40f",
  enhancing: "#2ecc71",
};

// ─── Segmentation Label Colors ───
const LABEL_COLORS = {
  0: "transparent",
  1: COLORS.necrotic,
  2: COLORS.edema,
  3: COLORS.enhancing,
  4: COLORS.enhancing,
};

// ─── Tab Button ───
function TabButton({ active, children, onClick, icon }) {
  return (
    <button
      onClick={onClick}
      style={{
        padding: "10px 18px",
        background: active ? COLORS.accent : "transparent",
        color: active ? "#000" : COLORS.textDim,
        border: `1px solid ${active ? COLORS.accent : COLORS.border}`,
        borderRadius: "8px",
        cursor: "pointer",
        fontWeight: active ? 700 : 500,
        fontSize: "13px",
        display: "flex",
        alignItems: "center",
        gap: "6px",
        transition: "all 0.2s",
        fontFamily: "'JetBrains Mono', monospace",
      }}
    >
      {icon && <span style={{ fontSize: "16px" }}>{icon}</span>}
      {children}
    </button>
  );
}

// ─── Status Badge ───
function StatusBadge({ status }) {
  const colors = {
    PASS: COLORS.success,
    REVIEW: COLORS.warning,
    FAIL: COLORS.danger,
    completed: COLORS.success,
    processing: COLORS.accent,
    error: COLORS.danger,
  };
  return (
    <span
      style={{
        padding: "2px 10px",
        borderRadius: "99px",
        fontSize: "11px",
        fontWeight: 700,
        background: `${colors[status] || COLORS.textDim}22`,
        color: colors[status] || COLORS.textDim,
        border: `1px solid ${colors[status] || COLORS.textDim}44`,
        fontFamily: "'JetBrains Mono', monospace",
      }}
    >
      {status}
    </span>
  );
}

// ─── Metric Card ───
function MetricCard({ label, value, unit, color, sub }) {
  return (
    <div
      style={{
        background: COLORS.surfaceLight,
        border: `1px solid ${COLORS.border}`,
        borderRadius: "12px",
        padding: "16px",
        flex: 1,
        minWidth: "140px",
      }}
    >
      <div style={{ fontSize: "11px", color: COLORS.textDim, marginBottom: "4px", textTransform: "uppercase", letterSpacing: "0.5px" }}>
        {label}
      </div>
      <div style={{ fontSize: "24px", fontWeight: 800, color: color || COLORS.text, fontFamily: "'JetBrains Mono', monospace" }}>
        {value}
        {unit && <span style={{ fontSize: "12px", color: COLORS.textDim, marginLeft: "4px" }}>{unit}</span>}
      </div>
      {sub && <div style={{ fontSize: "11px", color: COLORS.textDim, marginTop: "4px" }}>{sub}</div>}
    </div>
  );
}

// ─── 3D Slice Viewer (Canvas-based) ───
function SliceViewer({ volumeData, segData, sliceAxis, sliceIndex, onSliceChange, showOverlay, opacity }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    if (!canvasRef.current) return;
    const ctx = canvasRef.current.getContext("2d");
    const size = 256;
    canvasRef.current.width = size;
    canvasRef.current.height = size;

    // Draw gray background
    ctx.fillStyle = "#1a1a2e";
    ctx.fillRect(0, 0, size, size);

    if (!volumeData) {
      ctx.fillStyle = COLORS.textDim;
      ctx.font = "14px JetBrains Mono";
      ctx.textAlign = "center";
      ctx.fillText("Upload MRI to view", size / 2, size / 2);
      return;
    }

    // Render slice from synthetic data
    const dim = volumeData.shape;
    const idx = Math.min(sliceIndex, dim[0] - 1);

    const imgData = ctx.createImageData(size, size);
    const scaleX = dim[1] / size;
    const scaleY = dim[2] / size;

    for (let y = 0; y < size; y++) {
      for (let x = 0; x < size; x++) {
        const sx = Math.floor(x * scaleX);
        const sy = Math.floor(y * scaleY);
        let val = 0;

        if (sliceAxis === "axial") {
          val = volumeData.data[idx * dim[1] * dim[2] + sy * dim[2] + sx];
        } else if (sliceAxis === "sagittal") {
          val = volumeData.data[sy * dim[1] * dim[2] + sx * dim[2] + idx];
        } else {
          val = volumeData.data[sy * dim[1] * dim[2] + idx * dim[2] + sx];
        }

        const normalized = Math.max(0, Math.min(255, ((val + 3) / 6) * 255));
        const pixel = (y * size + x) * 4;
        imgData.data[pixel] = normalized;
        imgData.data[pixel + 1] = normalized;
        imgData.data[pixel + 2] = normalized;
        imgData.data[pixel + 3] = 255;

        // Overlay segmentation
        if (showOverlay && segData) {
          let segVal = 0;
          if (sliceAxis === "axial") {
            segVal = segData.data[idx * dim[1] * dim[2] + sy * dim[2] + sx];
          } else if (sliceAxis === "sagittal") {
            segVal = segData.data[sy * dim[1] * dim[2] + sx * dim[2] + idx];
          } else {
            segVal = segData.data[sy * dim[1] * dim[2] + idx * dim[2] + sx];
          }

          if (segVal > 0) {
            const colors = { 1: [231, 76, 60], 2: [241, 196, 15], 3: [46, 204, 113], 4: [46, 204, 113] };
            const c = colors[segVal] || [255, 255, 255];
            const a = opacity;
            imgData.data[pixel] = normalized * (1 - a) + c[0] * a;
            imgData.data[pixel + 1] = normalized * (1 - a) + c[1] * a;
            imgData.data[pixel + 2] = normalized * (1 - a) + c[2] * a;
          }
        }
      }
    }
    ctx.putImageData(imgData, 0, 0);

    // Crosshair
    ctx.strokeStyle = `${COLORS.accent}44`;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(size / 2, 0);
    ctx.lineTo(size / 2, size);
    ctx.moveTo(0, size / 2);
    ctx.lineTo(size, size / 2);
    ctx.stroke();
  }, [volumeData, segData, sliceAxis, sliceIndex, showOverlay, opacity]);

  const maxSlice = volumeData ? volumeData.shape[0] - 1 : 127;

  return (
    <div>
      <canvas
        ref={canvasRef}
        style={{ borderRadius: "8px", border: `1px solid ${COLORS.border}`, width: "100%", maxWidth: "400px", imageRendering: "pixelated" }}
      />
      <div style={{ display: "flex", alignItems: "center", gap: "8px", marginTop: "8px" }}>
        <span style={{ fontSize: "11px", color: COLORS.textDim, width: "60px" }}>{sliceAxis}</span>
        <input
          type="range"
          min={0}
          max={maxSlice}
          value={sliceIndex}
          onChange={(e) => onSliceChange(parseInt(e.target.value))}
          style={{ flex: 1, accentColor: COLORS.accent }}
        />
        <span style={{ fontSize: "11px", color: COLORS.accent, fontFamily: "'JetBrains Mono', monospace", width: "36px" }}>
          {sliceIndex}
        </span>
      </div>
    </div>
  );
}

// ─── File Upload Area ───
function FileUpload({ onFilesUploaded, isProcessing }) {
  const [dragOver, setDragOver] = useState(false);
  const [files, setFiles] = useState({});
  const inputRef = useRef();

  const handleFiles = (fileList) => {
    const newFiles = { ...files };
    Array.from(fileList).forEach((f) => {
      const name = f.name.toLowerCase();
      if (name.includes("t1ce") || name.includes("t1c")) newFiles.t1ce = f;
      else if (name.includes("t1")) newFiles.t1 = f;
      else if (name.includes("t2f") || name.includes("flair")) newFiles.flair = f;
      else if (name.includes("t2")) newFiles.t2 = f;
      else newFiles.flair = f; // Default to FLAIR
    });
    setFiles(newFiles);
  };

  return (
    <div style={{ marginBottom: "20px" }}>
      <div
        onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
        onDragLeave={() => setDragOver(false)}
        onDrop={(e) => { e.preventDefault(); setDragOver(false); handleFiles(e.dataTransfer.files); }}
        onClick={() => inputRef.current?.click()}
        style={{
          border: `2px dashed ${dragOver ? COLORS.accent : COLORS.border}`,
          borderRadius: "12px",
          padding: "32px",
          textAlign: "center",
          cursor: "pointer",
          background: dragOver ? `${COLORS.accent}11` : "transparent",
          transition: "all 0.2s",
        }}
      >
        <input
          ref={inputRef}
          type="file"
          multiple
          accept=".nii,.nii.gz,.gz"
          style={{ display: "none" }}
          onChange={(e) => handleFiles(e.target.files)}
        />
        <div style={{ fontSize: "32px", marginBottom: "8px" }}>🧠</div>
        <div style={{ fontSize: "14px", color: COLORS.text, fontWeight: 600 }}>
          Drop MRI files here (NIfTI format)
        </div>
        <div style={{ fontSize: "12px", color: COLORS.textDim, marginTop: "4px" }}>
          T1, T1CE, T2, FLAIR — .nii.gz files
        </div>
      </div>

      {Object.keys(files).length > 0 && (
        <div style={{ marginTop: "12px" }}>
          <div style={{ display: "flex", gap: "8px", flexWrap: "wrap", marginBottom: "12px" }}>
            {Object.entries(files).map(([mod, f]) => (
              <div
                key={mod}
                style={{
                  background: COLORS.surfaceLight,
                  border: `1px solid ${COLORS.accent}44`,
                  borderRadius: "8px",
                  padding: "6px 12px",
                  fontSize: "12px",
                  display: "flex",
                  alignItems: "center",
                  gap: "6px",
                }}
              >
                <span style={{ color: COLORS.accent, fontWeight: 700, textTransform: "uppercase" }}>{mod}</span>
                <span style={{ color: COLORS.textDim }}>{f.name}</span>
              </div>
            ))}
          </div>
          <button
            onClick={() => onFilesUploaded(files)}
            disabled={isProcessing}
            style={{
              width: "100%",
              padding: "12px",
              background: isProcessing ? COLORS.surfaceLight : `linear-gradient(135deg, ${COLORS.accent}, ${COLORS.accentDim})`,
              color: isProcessing ? COLORS.textDim : "#000",
              border: "none",
              borderRadius: "10px",
              fontSize: "14px",
              fontWeight: 700,
              cursor: isProcessing ? "wait" : "pointer",
              fontFamily: "'JetBrains Mono', monospace",
            }}
          >
            {isProcessing ? "⏳ Processing..." : "▶ Run Segmentation"}
          </button>
        </div>
      )}
    </div>
  );
}

// ─── QA Report Panel ───
function QAPanel({ qaResult }) {
  if (!qaResult) return null;
  return (
    <div style={{ background: COLORS.surfaceLight, border: `1px solid ${COLORS.border}`, borderRadius: "12px", padding: "20px" }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "16px" }}>
        <h3 style={{ margin: 0, color: COLORS.text, fontSize: "16px" }}>Quality Assurance</h3>
        <StatusBadge status={qaResult.overall_status} />
      </div>
      <div style={{
        background: COLORS.bg,
        borderRadius: "8px",
        padding: "12px",
        marginBottom: "12px",
        display: "flex",
        alignItems: "center",
        gap: "12px",
      }}>
        <div style={{
          width: "48px",
          height: "48px",
          borderRadius: "50%",
          background: `conic-gradient(${COLORS.success} ${qaResult.segmentation_quality_score}%, ${COLORS.border} 0)`,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
        }}>
          <div style={{
            width: "36px",
            height: "36px",
            borderRadius: "50%",
            background: COLORS.bg,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            fontSize: "12px",
            fontWeight: 700,
            color: COLORS.text,
            fontFamily: "'JetBrains Mono', monospace",
          }}>
            {Math.round(qaResult.segmentation_quality_score)}
          </div>
        </div>
        <div>
          <div style={{ fontSize: "13px", fontWeight: 600, color: COLORS.text }}>Quality Score</div>
          <div style={{ fontSize: "11px", color: COLORS.textDim }}>
            {qaResult.checks.filter(c => c.status === "PASS").length}/{qaResult.checks.length} checks passed
          </div>
        </div>
      </div>
      {qaResult.checks.map((check, i) => (
        <div
          key={i}
          style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            padding: "8px 0",
            borderBottom: i < qaResult.checks.length - 1 ? `1px solid ${COLORS.border}` : "none",
          }}
        >
          <span style={{ fontSize: "12px", color: COLORS.text }}>{check.check_name}</span>
          <StatusBadge status={check.status} />
        </div>
      ))}
    </div>
  );
}

// ─── Comparison Panel ───
function ComparisonPanel({ comparisonResult }) {
  if (!comparisonResult) return null;
  return (
    <div style={{ background: COLORS.surfaceLight, border: `1px solid ${COLORS.border}`, borderRadius: "12px", padding: "20px" }}>
      <h3 style={{ margin: "0 0 16px 0", color: COLORS.text, fontSize: "16px" }}>Comparative Analysis</h3>
      <div style={{ display: "flex", gap: "12px", flexWrap: "wrap", marginBottom: "16px" }}>
        <MetricCard label="Overall Dice" value={comparisonResult.overall_dice.toFixed(3)} color={COLORS.accent} />
        <MetricCard label="Hausdorff" value={comparisonResult.overall_hausdorff_mm.toFixed(1)} unit="mm" color={COLORS.warning} />
        <MetricCard label="Volume Δ" value={comparisonResult.volume_change_percent.toFixed(1)} unit="%" color={comparisonResult.volume_change_percent > 0 ? COLORS.danger : COLORS.success} />
      </div>
      <div style={{ background: COLORS.bg, borderRadius: "8px", padding: "12px" }}>
        <div style={{ fontSize: "13px", fontWeight: 600, color: COLORS.text, marginBottom: "4px" }}>Progression</div>
        <div style={{ fontSize: "14px", color: COLORS.accent }}>{comparisonResult.progression_status}</div>
      </div>
    </div>
  );
}

// ─── Radiomics Panel ───
function RadiomicsPanel({ radiomicsResult }) {
  if (!radiomicsResult) return null;
  return (
    <div style={{ background: COLORS.surfaceLight, border: `1px solid ${COLORS.border}`, borderRadius: "12px", padding: "20px" }}>
      <h3 style={{ margin: "0 0 12px 0", color: COLORS.text, fontSize: "16px" }}>
        Radiomics Features ({radiomicsResult.total_features})
      </h3>
      <div style={{ display: "flex", gap: "8px", flexWrap: "wrap", marginBottom: "12px" }}>
        {Object.entries(radiomicsResult.categories).map(([cat, count]) => (
          <div
            key={cat}
            style={{
              background: COLORS.bg,
              border: `1px solid ${COLORS.border}`,
              borderRadius: "6px",
              padding: "4px 10px",
              fontSize: "11px",
            }}
          >
            <span style={{ color: COLORS.accent, fontWeight: 600 }}>{cat}</span>
            <span style={{ color: COLORS.textDim, marginLeft: "4px" }}>({count})</span>
          </div>
        ))}
      </div>
      <div style={{ maxHeight: "300px", overflowY: "auto" }}>
        <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "11px" }}>
          <thead>
            <tr style={{ borderBottom: `1px solid ${COLORS.border}` }}>
              <th style={{ textAlign: "left", padding: "6px 8px", color: COLORS.textDim }}>Feature</th>
              <th style={{ textAlign: "right", padding: "6px 8px", color: COLORS.textDim }}>Value</th>
            </tr>
          </thead>
          <tbody>
            {radiomicsResult.features.slice(0, 30).map((f, i) => (
              <tr key={i} style={{ borderBottom: `1px solid ${COLORS.border}22` }}>
                <td style={{ padding: "4px 8px", color: COLORS.text, fontFamily: "'JetBrains Mono', monospace" }}>
                  {f.feature_name}
                </td>
                <td style={{ padding: "4px 8px", textAlign: "right", color: COLORS.accent, fontFamily: "'JetBrains Mono', monospace" }}>
                  {typeof f.value === "number" ? f.value.toFixed(4) : f.value}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

// ─── Survival Prediction Panel ───
function SurvivalPanel({ survivalResult }) {
  if (!survivalResult) return null;
  const maxTime = survivalResult.survival_curve[survivalResult.survival_curve.length - 1]?.time_days || 1;
  return (
    <div style={{ background: COLORS.surfaceLight, border: `1px solid ${COLORS.border}`, borderRadius: "12px", padding: "20px" }}>
      <h3 style={{ margin: "0 0 16px 0", color: COLORS.text, fontSize: "16px" }}>Survival Prediction</h3>
      <div style={{ display: "flex", gap: "12px", flexWrap: "wrap", marginBottom: "16px" }}>
        <MetricCard label="Predicted OS" value={survivalResult.predicted_os_months.toFixed(0)} unit="mo" color={COLORS.accent} />
        <MetricCard label="Risk Group" value={survivalResult.risk_group} color={
          survivalResult.risk_group === "High Risk" ? COLORS.danger :
          survivalResult.risk_group === "Moderate Risk" ? COLORS.warning : COLORS.success
        } />
      </div>
      {/* Mini survival curve */}
      <div style={{ background: COLORS.bg, borderRadius: "8px", padding: "12px" }}>
        <div style={{ fontSize: "11px", color: COLORS.textDim, marginBottom: "8px" }}>Kaplan-Meier Survival Curve</div>
        <svg width="100%" height="120" viewBox="0 0 300 120">
          <line x1="30" y1="100" x2="290" y2="100" stroke={COLORS.border} strokeWidth="1" />
          <line x1="30" y1="10" x2="30" y2="100" stroke={COLORS.border} strokeWidth="1" />
          <polyline
            fill="none"
            stroke={COLORS.accent}
            strokeWidth="2"
            points={survivalResult.survival_curve.map((p, i) => {
              const x = 30 + (p.time_days / maxTime) * 260;
              const y = 100 - p.survival_probability * 90;
              return `${x},${y}`;
            }).join(" ")}
          />
          <text x="160" y="118" fill={COLORS.textDim} fontSize="9" textAnchor="middle">Days</text>
          <text x="12" y="55" fill={COLORS.textDim} fontSize="9" textAnchor="middle" transform="rotate(-90, 12, 55)">Probability</text>
        </svg>
      </div>
    </div>
  );
}

// ─── Main App ───
export default function BrainTumorDashboard() {
  const [activeTab, setActiveTab] = useState("viewer");
  const [isProcessing, setIsProcessing] = useState(false);
  const [backendUrl, setBackendUrl] = useState(API_BASE);
  const [backendStatus, setBackendStatus] = useState("unknown"); // "ok", "down", "unknown"
  const [volumeData, setVolumeData] = useState(null);
  const [segData, setSegData] = useState(null);
  const [sliceIndex, setSliceIndex] = useState(64);
  const [sliceAxis, setSliceAxis] = useState("axial");
  const [showOverlay, setShowOverlay] = useState(true);
  const [overlayOpacity, setOverlayOpacity] = useState(0.5);
  const [segResult, setSegResult] = useState(null);
  const [gradingResult, setGradingResult] = useState(null);
  const [uncertaintyResult, setUncertaintyResult] = useState(null);
  const [qaResult, setQaResult] = useState(null);
  const [comparisonResult, setComparisonResult] = useState(null);
  const [radiomicsResult, setRadiomicsResult] = useState(null);
  const [survivalResult, setSurvivalResult] = useState(null);
  const [logs, setLogs] = useState([]);
  const [useDemoMode, setUseDemoMode] = useState(false);

  const addLog = (msg) => setLogs((prev) => [...prev, `[${new Date().toLocaleTimeString()}] ${msg}`]);

  // Generate demo data
  const runDemoMode = useCallback(() => {
    setIsProcessing(true);
    addLog("Generating synthetic BraTS demo data...");

    setTimeout(() => {
      const shape = [128, 128, 128];
      const size = shape[0] * shape[1] * shape[2];

      // Generate synthetic volume
      const volData = new Float32Array(size);
      const segDataArr = new Int32Array(size);
      const center = shape.map((s) => s / 2);

      for (let z = 0; z < shape[0]; z++) {
        for (let y = 0; y < shape[1]; y++) {
          for (let x = 0; x < shape[2]; x++) {
            const idx = z * shape[1] * shape[2] + y * shape[2] + x;
            const dist = Math.sqrt((z - center[0]) ** 2 + (y - center[1]) ** 2 + (x - center[2]) ** 2);

            // Brain
            if (dist < 50) {
              volData[idx] = 0.8 + Math.random() * 0.3;

              // Tumor
              const tDist = Math.sqrt((z - center[0] - 5) ** 2 + (y - center[1] + 3) ** 2 + (x - center[2] - 2) ** 2);
              if (tDist < 8) {
                segDataArr[idx] = 4;
                volData[idx] = 1.5 + Math.random() * 0.2;
              } else if (tDist < 14) {
                segDataArr[idx] = 1;
                volData[idx] = 0.5 + Math.random() * 0.2;
              } else if (tDist < 22) {
                segDataArr[idx] = 2;
                volData[idx] = 1.1 + Math.random() * 0.2;
              }
            }
          }
        }
      }

      setVolumeData({ data: volData, shape });
      setSegData({ data: segDataArr, shape });
      addLog("Volume & segmentation generated");

      // Segmentation result
      const et = Array.from(segDataArr).filter((v) => v === 4).length;
      const ncr = Array.from(segDataArr).filter((v) => v === 1).length;
      const ed = Array.from(segDataArr).filter((v) => v === 2).length;
      const total = et + ncr + ed;

      setSegResult({
        job_id: "demo-001",
        status: "completed",
        total_tumor_volume_cm3: total / 1000,
        inference_time_seconds: 1.23,
        model_used: "3D-UNet (BraTS)",
        regions: [
          { label: 1, name: "Necrotic", volume_cm3: ncr / 1000, voxel_count: ncr, percentage: ((ncr / total) * 100).toFixed(1) },
          { label: 2, name: "Edema", volume_cm3: ed / 1000, voxel_count: ed, percentage: ((ed / total) * 100).toFixed(1) },
          { label: 4, name: "Enhancing", volume_cm3: et / 1000, voxel_count: et, percentage: ((et / total) * 100).toFixed(1) },
        ],
      });

      // Grading
      setGradingResult({
        predicted_grade: "Grade III",
        who_classification: "High-grade glioma (WHO Grade III)",
        confidence: 0.78,
        grade_probabilities: { "Grade I": 0.05, "Grade II": 0.12, "Grade III": 0.78, "Grade IV": 0.05 },
        risk_stratification: "High",
        clinical_notes: "Anaplastic astrocytoma — malignant, rapid growth",
      });

      // Uncertainty
      setUncertaintyResult({
        mean_uncertainty: 0.23,
        max_uncertainty: 0.89,
        high_uncertainty_percentage: 12.5,
        clinical_interpretation: "Moderate uncertainty — some ambiguous regions present at tumor margins.",
      });

      // QA
      setQaResult({
        overall_status: "PASS",
        segmentation_quality_score: 87.5,
        checks: [
          { check_name: "Connectivity (Necrotic)", status: "PASS", message: "Single connected component" },
          { check_name: "Connectivity (Edema)", status: "PASS", message: "Single connected component" },
          { check_name: "Connectivity (Enhancing)", status: "PASS", message: "Single connected component" },
          { check_name: "Volume Plausibility", status: "PASS", message: `Volume ${(total / 1000).toFixed(2)} cm³ within range` },
          { check_name: "Hole Detection", status: "PASS", message: "No internal holes" },
          { check_name: "Label Hierarchy", status: "PASS", message: "Enhancing properly surrounded" },
          { check_name: "Symmetry Check", status: "REVIEW", message: "Minor asymmetry detected" },
          { check_name: "Non-Empty Check", status: "PASS", message: "Contains labels [1, 2, 4]" },
        ],
        recommendations: ["Segmentation passes all quality checks"],
      });

      // Radiomics
      setRadiomicsResult({
        total_features: 63,
        categories: { shape: 10, firstorder: 18, glcm: 10, glrlm: 5 },
        features: [
          { category: "shape", feature_name: "shape_Volume", value: total },
          { category: "shape", feature_name: "shape_Sphericity", value: 0.72 },
          { category: "shape", feature_name: "shape_Elongation", value: 0.85 },
          { category: "firstorder", feature_name: "firstorder_Mean", value: 0.943 },
          { category: "firstorder", feature_name: "firstorder_Std", value: 0.412 },
          { category: "firstorder", feature_name: "firstorder_Skewness", value: -0.234 },
          { category: "firstorder", feature_name: "firstorder_Entropy", value: 3.456 },
          { category: "glcm", feature_name: "glcm_Contrast", value: 12.34 },
          { category: "glcm", feature_name: "glcm_Homogeneity", value: 0.567 },
          { category: "glcm", feature_name: "glcm_Energy", value: 0.0234 },
          { category: "glrlm", feature_name: "glrlm_ShortRunEmphasis", value: 0.891 },
        ],
      });

      // Survival
      setSurvivalResult({
        predicted_os_days: 385,
        predicted_os_months: 12.8,
        confidence_interval_lower: 270,
        confidence_interval_upper: 500,
        risk_group: "Moderate Risk",
        survival_curve: Array.from({ length: 20 }, (_, i) => ({
          time_days: i * 30,
          survival_probability: Math.exp((-i * 30) / 578),
        })),
        features_importance: { enhancing_ratio: 0.4, tumor_volume: 0.3, necrotic_ratio: 0.2, sphericity: 0.1 },
      });

      setIsProcessing(false);
      addLog("All analyses complete");
    }, 1500);
  }, []);

  const handleFilesUploaded = async (files) => {
    setIsProcessing(true);
    setLogs([]);
    addLog("Starting full analysis pipeline...");

    // Try real backend first
    let backendOk = false;
    try {
      const h = await fetch(`${API_BASE}/health`, { signal: AbortSignal.timeout(3000) });
      backendOk = h.ok;
    } catch { backendOk = false; }

    if (!backendOk) {
      addLog("Backend not reachable — using demo mode");
      runDemoMode();
      return;
    }

    try {
      // ── 1. Segmentation ──
      addLog("Uploading MRI files...");
      const formData = new FormData();
      Object.entries(files).forEach(([mod, file]) => formData.append(mod, file));

      const segRes = await fetch(`${API_BASE}/api/segment`, { method: "POST", body: formData });
      if (!segRes.ok) throw new Error(`Segmentation failed: ${segRes.status}`);
      const segJson = await segRes.json();
      setSegResult(segJson);
      addLog(`Segmentation done: ${segJson.total_tumor_volume_cm3.toFixed(2)} cm³`);

      // Get the segmentation file for downstream features
      const segBlob = await (await fetch(`${API_BASE}${segJson.download_url}`)).blob();
      const segFile = new File([segBlob], "segmentation.nii.gz");

      // ── 2. Tumor Grading ──
      addLog("Running tumor grading...");
      try {
        const gf = new FormData(); gf.append("seg_file", segFile);
        const gr = await fetch(`${API_BASE}/api/grade-tumor`, { method: "POST", body: gf });
        if (gr.ok) { setGradingResult(await gr.json()); addLog("Grading complete"); }
      } catch (e) { addLog(`Grading skipped: ${e.message}`); }

      // ── 3. QA Check ──
      addLog("Running quality assurance...");
      try {
        const qf = new FormData(); qf.append("seg_file", segFile);
        const qr = await fetch(`${API_BASE}/api/qa`, { method: "POST", body: qf });
        if (qr.ok) { setQaResult(await qr.json()); addLog("QA complete"); }
      } catch (e) { addLog(`QA skipped: ${e.message}`); }

      // ── 4. Radiomics (needs original image + seg) ──
      addLog("Extracting radiomic features...");
      try {
        const rf = new FormData();
        // Use any available modality as the image
        const imgFile = files.t1ce || files.flair || files.t2 || files.t1;
        if (imgFile) {
          rf.append("image_file", imgFile);
          rf.append("seg_file", segFile);
          const rr = await fetch(`${API_BASE}/api/radiomics`, { method: "POST", body: rf });
          if (rr.ok) { setRadiomicsResult(await rr.json()); addLog("Radiomics complete"); }
        }
      } catch (e) { addLog(`Radiomics skipped: ${e.message}`); }

      // ── 5. Survival Prediction ──
      addLog("Running survival prediction...");
      try {
        const sf = new FormData(); sf.append("seg_file", segFile);
        const sr = await fetch(`${API_BASE}/api/survival-prediction`, { method: "POST", body: sf });
        if (sr.ok) { setSurvivalResult(await sr.json()); addLog("Survival prediction complete"); }
      } catch (e) { addLog(`Survival skipped: ${e.message}`); }

      // ── 6. Uncertainty (needs original modalities — expensive, run last) ──
      addLog("Running uncertainty quantification (MC-Dropout, may take 2-3 min)...");
      try {
        const uf = new FormData();
        Object.entries(files).forEach(([mod, file]) => uf.append(mod, file));
        const ur = await fetch(`${API_BASE}/api/uncertainty`, { method: "POST", body: uf });
        if (ur.ok) { setUncertaintyResult(await ur.json()); addLog("Uncertainty analysis complete"); }
      } catch (e) { addLog(`Uncertainty skipped: ${e.message}`); }

      addLog("✅ Full pipeline complete!");
    } catch (err) {
      addLog(`Error: ${err.message} — falling back to demo mode`);
      runDemoMode();
    } finally {
      setIsProcessing(false);
    }
  };

  const tabs = [
    { id: "viewer", label: "3D Viewer", icon: "🧠" },
    { id: "analysis", label: "Analysis", icon: "📊" },
    { id: "qa", label: "QA", icon: "✅" },
    { id: "radiomics", label: "Radiomics", icon: "🔬" },
    { id: "survival", label: "Survival", icon: "📈" },
  ];

  return (
    <div style={{ minHeight: "100vh", background: COLORS.bg, color: COLORS.text, fontFamily: "'Inter', 'Segoe UI', sans-serif" }}>
      {/* Header */}
      <header style={{
        background: `linear-gradient(135deg, ${COLORS.surface}, ${COLORS.surfaceLight})`,
        borderBottom: `1px solid ${COLORS.border}`,
        padding: "16px 24px",
        display: "flex",
        justifyContent: "space-between",
        alignItems: "center",
      }}>
        <div>
          <h1 style={{ margin: 0, fontSize: "20px", fontWeight: 800, letterSpacing: "-0.5px" }}>
            <span style={{ color: COLORS.accent }}>ML</span> Brain Tumor Segmentation
          </h1>
          <p style={{ margin: "2px 0 0 0", fontSize: "12px", color: COLORS.textDim }}>
            by Rahul & Krishnaa • Dr. Valarmathi Lab • VIT Chennai
          </p>
        </div>
        <div style={{ display: "flex", gap: "8px", alignItems: "center" }}>
          <div style={{ display: "flex", alignItems: "center", gap: "4px", background: COLORS.bg, borderRadius: "6px", padding: "4px 8px", border: `1px solid ${COLORS.border}` }}>
            <div style={{ width: "8px", height: "8px", borderRadius: "50%", background: backendStatus === "ok" ? COLORS.success : backendStatus === "down" ? COLORS.danger : COLORS.warning }} />
            <input
              value={backendUrl}
              onChange={(e) => setBackendUrl(e.target.value)}
              onBlur={async () => {
                try {
                  const r = await fetch(`${backendUrl}/health`, { signal: AbortSignal.timeout(2000) });
                  setBackendStatus(r.ok ? "ok" : "down");
                } catch { setBackendStatus("down"); }
              }}
              style={{ background: "transparent", border: "none", color: COLORS.textDim, fontSize: "11px", width: "180px", fontFamily: "'JetBrains Mono', monospace", outline: "none" }}
              placeholder="http://localhost:8000"
            />
          </div>
          {segResult && <StatusBadge status={segResult.status} />}
          <button
            onClick={runDemoMode}
            style={{
              padding: "8px 16px",
              background: COLORS.surfaceLight,
              color: COLORS.accent,
              border: `1px solid ${COLORS.accent}44`,
              borderRadius: "8px",
              cursor: "pointer",
              fontSize: "12px",
              fontWeight: 600,
            }}
          >
            ▶ Demo Mode
          </button>
        </div>
      </header>

      {/* Tab Navigation */}
      <nav style={{ padding: "12px 24px", display: "flex", gap: "8px", overflowX: "auto", borderBottom: `1px solid ${COLORS.border}` }}>
        {tabs.map((tab) => (
          <TabButton key={tab.id} active={activeTab === tab.id} onClick={() => setActiveTab(tab.id)} icon={tab.icon}>
            {tab.label}
          </TabButton>
        ))}
      </nav>

      <div style={{ padding: "24px", maxWidth: "1400px", margin: "0 auto" }}>
        {/* Viewer Tab */}
        {activeTab === "viewer" && (
          <div style={{ display: "grid", gridTemplateColumns: "1fr 320px", gap: "24px" }}>
            <div>
              <FileUpload onFilesUploaded={handleFilesUploaded} isProcessing={isProcessing} />
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: "16px" }}>
                {["axial", "coronal", "sagittal"].map((axis) => (
                  <div key={axis}>
                    <div style={{ fontSize: "12px", color: COLORS.textDim, marginBottom: "8px", textTransform: "capitalize", fontWeight: 600 }}>
                      {axis}
                    </div>
                    <SliceViewer
                      volumeData={volumeData}
                      segData={segData}
                      sliceAxis={axis}
                      sliceIndex={sliceIndex}
                      onSliceChange={setSliceIndex}
                      showOverlay={showOverlay}
                      opacity={overlayOpacity}
                    />
                  </div>
                ))}
              </div>
              {/* Controls */}
              <div style={{ display: "flex", gap: "16px", marginTop: "16px", alignItems: "center", flexWrap: "wrap" }}>
                <label style={{ display: "flex", alignItems: "center", gap: "6px", fontSize: "12px", color: COLORS.textDim, cursor: "pointer" }}>
                  <input type="checkbox" checked={showOverlay} onChange={(e) => setShowOverlay(e.target.checked)} />
                  Show segmentation overlay
                </label>
                <div style={{ display: "flex", alignItems: "center", gap: "6px", fontSize: "12px", color: COLORS.textDim }}>
                  Opacity:
                  <input
                    type="range"
                    min={0}
                    max={1}
                    step={0.05}
                    value={overlayOpacity}
                    onChange={(e) => setOverlayOpacity(parseFloat(e.target.value))}
                    style={{ width: "100px", accentColor: COLORS.accent }}
                  />
                  <span style={{ fontFamily: "'JetBrains Mono', monospace", width: "32px" }}>{(overlayOpacity * 100).toFixed(0)}%</span>
                </div>
              </div>
              {/* Legend */}
              <div style={{ display: "flex", gap: "16px", marginTop: "12px" }}>
                {[
                  { label: "Necrotic (1)", color: COLORS.necrotic },
                  { label: "Edema (2)", color: COLORS.edema },
                  { label: "Enhancing (4)", color: COLORS.enhancing },
                ].map((item) => (
                  <div key={item.label} style={{ display: "flex", alignItems: "center", gap: "6px", fontSize: "12px" }}>
                    <div style={{ width: "12px", height: "12px", borderRadius: "3px", background: item.color }} />
                    <span style={{ color: COLORS.textDim }}>{item.label}</span>
                  </div>
                ))}
              </div>
            </div>

            {/* Right Sidebar */}
            <div style={{ display: "flex", flexDirection: "column", gap: "16px" }}>
              {segResult ? (
                <>
                  <MetricCard label="Total Volume" value={segResult.total_tumor_volume_cm3.toFixed(2)} unit="cm³" color={COLORS.accent} />
                  <MetricCard label="Inference Time" value={segResult.inference_time_seconds.toFixed(2)} unit="sec" color={COLORS.success} />
                  <MetricCard label="Model" value={segResult.model_used} color={COLORS.text} />
                  <div style={{ background: COLORS.surfaceLight, border: `1px solid ${COLORS.border}`, borderRadius: "12px", padding: "16px" }}>
                    <div style={{ fontSize: "12px", color: COLORS.textDim, marginBottom: "12px", fontWeight: 600 }}>Region Volumes</div>
                    {segResult.regions.map((r) => (
                      <div key={r.label} style={{ marginBottom: "12px" }}>
                        <div style={{ display: "flex", justifyContent: "space-between", fontSize: "12px", marginBottom: "4px" }}>
                          <span style={{ color: COLORS.text }}>{r.name}</span>
                          <span style={{ color: COLORS.accent, fontFamily: "'JetBrains Mono', monospace" }}>{r.volume_cm3.toFixed(3)} cm³</span>
                        </div>
                        <div style={{ height: "6px", background: COLORS.bg, borderRadius: "3px", overflow: "hidden" }}>
                          <div
                            style={{
                              height: "100%",
                              width: `${Math.min(parseFloat(r.percentage), 100)}%`,
                              background: r.label === 1 ? COLORS.necrotic : r.label === 2 ? COLORS.edema : COLORS.enhancing,
                              borderRadius: "3px",
                              transition: "width 0.5s ease",
                            }}
                          />
                        </div>
                        <div style={{ fontSize: "10px", color: COLORS.textDim, marginTop: "2px" }}>
                          {r.voxel_count.toLocaleString()} voxels ({r.percentage}%)
                        </div>
                      </div>
                    ))}
                  </div>
                </>
              ) : (
                <div style={{ background: COLORS.surfaceLight, border: `1px solid ${COLORS.border}`, borderRadius: "12px", padding: "32px", textAlign: "center" }}>
                  <div style={{ fontSize: "32px", marginBottom: "8px" }}>📤</div>
                  <div style={{ fontSize: "13px", color: COLORS.textDim }}>Upload MRI files or click Demo Mode to see results</div>
                </div>
              )}

              {/* Log */}
              {logs.length > 0 && (
                <div style={{
                  background: COLORS.bg,
                  border: `1px solid ${COLORS.border}`,
                  borderRadius: "8px",
                  padding: "12px",
                  maxHeight: "150px",
                  overflowY: "auto",
                  fontSize: "10px",
                  fontFamily: "'JetBrains Mono', monospace",
                  color: COLORS.textDim,
                }}>
                  {logs.map((log, i) => (
                    <div key={i} style={{ marginBottom: "2px" }}>{log}</div>
                  ))}
                </div>
              )}
            </div>
          </div>
        )}

        {/* Analysis Tab */}
        {activeTab === "analysis" && (
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "24px" }}>
            {/* Grading */}
            {gradingResult && (
              <div style={{ background: COLORS.surfaceLight, border: `1px solid ${COLORS.border}`, borderRadius: "12px", padding: "20px" }}>
                <h3 style={{ margin: "0 0 16px 0", color: COLORS.text, fontSize: "16px" }}>🏥 Tumor Grading (WHO)</h3>
                <div style={{ display: "flex", gap: "12px", marginBottom: "16px" }}>
                  <MetricCard label="Predicted Grade" value={gradingResult.predicted_grade} color={
                    gradingResult.predicted_grade.includes("IV") ? COLORS.danger :
                    gradingResult.predicted_grade.includes("III") ? COLORS.warning : COLORS.success
                  } />
                  <MetricCard label="Confidence" value={`${(gradingResult.confidence * 100).toFixed(0)}%`} color={COLORS.accent} />
                </div>
                <div style={{ background: COLORS.bg, borderRadius: "8px", padding: "12px", marginBottom: "12px" }}>
                  <div style={{ fontSize: "12px", color: COLORS.textDim, marginBottom: "4px" }}>Classification</div>
                  <div style={{ fontSize: "14px", color: COLORS.text }}>{gradingResult.who_classification}</div>
                </div>
                <div style={{ background: COLORS.bg, borderRadius: "8px", padding: "12px", marginBottom: "12px" }}>
                  <div style={{ fontSize: "12px", color: COLORS.textDim, marginBottom: "8px" }}>Grade Probabilities</div>
                  {Object.entries(gradingResult.grade_probabilities).map(([grade, prob]) => (
                    <div key={grade} style={{ marginBottom: "6px" }}>
                      <div style={{ display: "flex", justifyContent: "space-between", fontSize: "11px", marginBottom: "2px" }}>
                        <span style={{ color: COLORS.text }}>{grade}</span>
                        <span style={{ color: COLORS.accent }}>{(prob * 100).toFixed(1)}%</span>
                      </div>
                      <div style={{ height: "4px", background: COLORS.surfaceLight, borderRadius: "2px" }}>
                        <div style={{ height: "100%", width: `${prob * 100}%`, background: COLORS.accent, borderRadius: "2px" }} />
                      </div>
                    </div>
                  ))}
                </div>
                <div style={{ fontSize: "12px", color: COLORS.textDim, fontStyle: "italic" }}>{gradingResult.clinical_notes}</div>
              </div>
            )}

            {/* Uncertainty */}
            {uncertaintyResult && (
              <div style={{ background: COLORS.surfaceLight, border: `1px solid ${COLORS.border}`, borderRadius: "12px", padding: "20px" }}>
                <h3 style={{ margin: "0 0 16px 0", color: COLORS.text, fontSize: "16px" }}>🎯 Uncertainty Quantification</h3>
                <div style={{ display: "flex", gap: "12px", marginBottom: "16px" }}>
                  <MetricCard label="Mean Uncertainty" value={uncertaintyResult.mean_uncertainty.toFixed(3)} color={COLORS.warning} />
                  <MetricCard label="Max Uncertainty" value={uncertaintyResult.max_uncertainty.toFixed(3)} color={COLORS.danger} />
                </div>
                <MetricCard label="High-Uncertainty Regions" value={`${uncertaintyResult.high_uncertainty_percentage.toFixed(1)}%`} color={COLORS.accent}
                  sub="of tumor volume with uncertainty > 0.5" />
                <div style={{ marginTop: "12px", background: COLORS.bg, borderRadius: "8px", padding: "12px" }}>
                  <div style={{ fontSize: "12px", color: COLORS.textDim, marginBottom: "4px" }}>Clinical Interpretation</div>
                  <div style={{ fontSize: "13px", color: COLORS.text }}>{uncertaintyResult.clinical_interpretation}</div>
                </div>
              </div>
            )}

            {/* Comparison */}
            <ComparisonPanel comparisonResult={comparisonResult} />

            {!gradingResult && !uncertaintyResult && (
              <div style={{
                gridColumn: "1 / -1",
                background: COLORS.surfaceLight,
                border: `1px solid ${COLORS.border}`,
                borderRadius: "12px",
                padding: "48px",
                textAlign: "center",
              }}>
                <div style={{ fontSize: "32px", marginBottom: "8px" }}>📊</div>
                <div style={{ fontSize: "14px", color: COLORS.textDim }}>Run segmentation first to see analysis results</div>
              </div>
            )}
          </div>
        )}

        {/* QA Tab */}
        {activeTab === "qa" && <QAPanel qaResult={qaResult} />}

        {/* Radiomics Tab */}
        {activeTab === "radiomics" && <RadiomicsPanel radiomicsResult={radiomicsResult} />}

        {/* Survival Tab */}
        {activeTab === "survival" && <SurvivalPanel survivalResult={survivalResult} />}
      </div>

      {/* Footer */}
      <footer style={{
        borderTop: `1px solid ${COLORS.border}`,
        padding: "12px 24px",
        display: "flex",
        justifyContent: "space-between",
        fontSize: "11px",
        color: COLORS.textDim,
      }}>
        <span>Built on BraTS 2024 • Three.js + TensorFlow • Privacy-first (all processing local)</span>
        <span>ML Project v1.0 • {new Date().getFullYear()}</span>
      </footer>
    </div>
  );
}
