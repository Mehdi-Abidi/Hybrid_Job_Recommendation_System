import React from 'react';
import { CheckCircle2, XCircle, ArrowRightCircle } from 'lucide-react';
import type { SkillGapResponse } from '../api';

interface SkillGapProps {
  data: SkillGapResponse | null;
}

export const SkillGap: React.FC<SkillGapProps> = ({ data }) => {
  if (!data) return null;

  return (
    <div className="glass-panel" style={{ padding: '2rem', marginBottom: '3rem' }}>
      <h3 style={{ fontSize: '1.5rem', marginBottom: '1.5rem' }}>Skill Analysis</h3>
      
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))', gap: '2rem' }}>
        <div>
          <h4 style={{ color: '#4ade80', display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '1rem' }}>
            <CheckCircle2 size={20} /> Matched Skills ({Math.round(data.match_rate * 100)}%)
          </h4>
          <div className="skills-container">
            {data.matched.length > 0 ? data.matched.map((s, i) => (
              <span key={i} className="skill-tag" style={{ borderColor: 'rgba(74, 222, 128, 0.3)', color: '#4ade80' }}>
                {s}
              </span>
            )) : <span style={{ color: 'var(--text-muted)' }}>No matches</span>}
          </div>
        </div>

        <div>
          <h4 style={{ color: '#f87171', display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '1rem' }}>
            <XCircle size={20} /> Missing Skills
          </h4>
          <div className="skills-container">
            {data.missing.length > 0 ? data.missing.map((s, i) => (
              <span key={i} className="skill-tag" style={{ borderColor: 'rgba(248, 113, 113, 0.3)', color: '#f87171' }}>
                {s}
              </span>
            )) : <span style={{ color: 'var(--text-muted)' }}>None! You're a perfect fit.</span>}
          </div>
        </div>

        {data.adjacent.length > 0 && (
          <div>
            <h4 style={{ color: '#fbbf24', display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '1rem' }}>
              <ArrowRightCircle size={20} /> Transferable Skills
            </h4>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
              {data.adjacent.map((adj, i) => (
                <div key={i} style={{ fontSize: '0.875rem', display: 'flex', alignItems: 'center', gap: '0.5rem', background: 'rgba(255,255,255,0.02)', padding: '0.5rem', borderRadius: '8px' }}>
                  <span style={{ color: '#e2e8f0' }}>{adj[0]}</span>
                  <ArrowRightCircle size={14} style={{ color: 'var(--text-muted)' }}/>
                  <span style={{ color: '#fbbf24' }}>{adj[1]}</span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
