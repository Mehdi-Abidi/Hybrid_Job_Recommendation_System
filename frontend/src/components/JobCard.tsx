import React from 'react';
import { Building2, MapPin, DollarSign, ExternalLink } from 'lucide-react';
import type { Recommendation } from '../api';

interface JobCardProps {
  recommendation: Recommendation;
}

export const JobCard: React.FC<JobCardProps> = ({ recommendation }) => {
  const { job, score } = recommendation;
  if (!job) return null;

  const formatSalary = (min?: number, max?: number) => {
    if (!min && !max) return 'Salary not specified';
    const format = (n: number) => `$${Math.round(n/1000)}k`;
    if (min && !max) return `${format(min)}+`;
    if (!min && max) return `Up to ${format(max)}`;
    return `${format(min!)} - ${format(max!)}`;
  };

  const skills = job.skills ? job.skills.split(',').map(s => s.trim()) : [];

  return (
    <div className="glass-panel job-card animate-fade-in">
      <div className="job-header">
        <div>
          <h3 className="job-title">{job.title}</h3>
          <div className="job-company">
            <Building2 size={16} className="text-muted" />
            {job.category || 'Technology'} • {job.seniority || 'Any'} level
          </div>
        </div>
        <div className="match-score">
          {(score * 100).toFixed(0)}% Match
        </div>
      </div>
      
      <div className="job-meta">
        <div className="meta-item">
          <MapPin size={16} className="text-muted" />
          {job.location || 'Remote'}
        </div>
        <div className="meta-item">
          <DollarSign size={16} className="text-muted" />
          {formatSalary(job.salary_min, job.salary_max)}
        </div>
      </div>

      <div className="skills-container">
        {skills.slice(0, 5).map((skill, idx) => (
          <span key={idx} className="skill-tag">{skill}</span>
        ))}
        {skills.length > 5 && (
          <span className="skill-tag" style={{ background: 'rgba(139, 92, 246, 0.1)', color: '#c4b5fd', borderColor: 'rgba(139, 92, 246, 0.2)' }}>
            +{skills.length - 5} more
          </span>
        )}
      </div>

      <div className="job-actions">
        <button className="btn-apply">
          View Details & Apply <ExternalLink size={16} />
        </button>
      </div>
    </div>
  );
};
