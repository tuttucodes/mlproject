import React, { useEffect, useRef, useState } from 'react';
import { Niivue } from '@niivue/niivue';
import './Viewer.css';

export function Viewer({ mriData, segmentationData, isLoading }) {
  const canvasRef = useRef(null);
  const niivueRef = useRef(null);
  const [opacity, setOpacity] = useState(0.6);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (!canvasRef.current) return;

    // Initialize Niivue viewer
    const niivue = new Niivue({
      canvas: canvasRef.current,
      logging: false,
    });
    niivueRef.current = niivue;

    return () => {
      if (niivueRef.current) {
        niivueRef.current.dispose();
      }
    };
  }, []);

  useEffect(() => {
    if (!niivueRef.current) return;
    if (!mriData) {
      niivueRef.current.volumes = [];
      return;
    }

    try {
      setError(null);

      // Decode base64 MRI data
      const binaryString = atob(mriData);
      const bytes = new Uint8Array(binaryString.length);
      for (let i = 0; i < binaryString.length; i++) {
        bytes[i] = binaryString.charCodeAt(i);
      }
      const blob = new Blob([bytes], { type: 'application/octet-stream' });
      const url = URL.createObjectURL(blob);

      // Create volume object for MRI
      const mriVolume = {
        url: url,
        colormap: 'gray',
        opacity: 1.0,
      };

      // Load MRI as primary volume
      niivueRef.current.addVolume(mriVolume);

      // Load segmentation as overlay if available
      if (segmentationData) {
        const segBinaryString = atob(segmentationData);
        const segBytes = new Uint8Array(segBinaryString.length);
        for (let i = 0; i < segBinaryString.length; i++) {
          segBytes[i] = segBinaryString.charCodeAt(i);
        }
        const segBlob = new Blob([segBytes], {
          type: 'application/octet-stream',
        });
        const segUrl = URL.createObjectURL(segBlob);

        const segmentationVolume = {
          url: segUrl,
          colormap: 'red',
          opacity: opacity,
        };

        niivueRef.current.addVolume(segmentationVolume);
      }
    } catch (err) {
      setError(`Failed to load volumes: ${err.message}`);
      console.error('Viewer error:', err);
    }
  }, [mriData, segmentationData]);

  // Update segmentation opacity
  useEffect(() => {
    if (!niivueRef.current || !segmentationData) return;
    if (niivueRef.current.volumes.length > 1) {
      niivueRef.current.setOpacity(segmentationData ? opacity : 1.0);
    }
  }, [opacity, segmentationData]);

  return (
    <div className="viewer-container">
      <div className="viewer-header">
        <h2>MRI Visualization</h2>
        {segmentationData && (
          <div className="opacity-control">
            <label htmlFor="opacity-slider">Segmentation Opacity:</label>
            <input
              id="opacity-slider"
              type="range"
              min="0"
              max="1"
              step="0.1"
              value={opacity}
              onChange={(e) => setOpacity(parseFloat(e.target.value))}
              disabled={isLoading}
            />
            <span>{Math.round(opacity * 100)}%</span>
          </div>
        )}
      </div>

      {error && <div className="error-message">{error}</div>}

      <canvas
        ref={canvasRef}
        className={`viewer-canvas ${isLoading ? 'loading' : ''}`}
      />

      {isLoading && (
        <div className="loading-overlay">
          <div className="spinner"></div>
          <p>Processing segmentation...</p>
        </div>
      )}

      <div className="viewer-legend">
        <div className="legend-item">
          <div className="legend-color gray"></div>
          <span>MRI Volume</span>
        </div>
        {segmentationData && (
          <>
            <div className="legend-item">
              <div className="legend-color red"></div>
              <span>Tumor Core (Class 1)</span>
            </div>
            <div className="legend-item">
              <div className="legend-color orange"></div>
              <span>Tumor Edema (Class 2)</span>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
