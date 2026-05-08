import React, { useState } from 'react';
import { Search, Sparkles } from 'lucide-react';
import { recommendJobs } from './api';
import type { RecommendationResponse } from './api';
import { JobCard } from './components/JobCard';

function App() {
  const [userId, setUserId] = useState<string>('42');
  const [loading, setLoading] = useState(false);
  const [response, setResponse] = useState<RecommendationResponse | null>(null);

  const handleSearch = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!userId) return;
    
    setLoading(true);
    try {
      const data = await recommendJobs(parseInt(userId), 'hybrid');
      setResponse(data);
    } catch (error) {
      console.error('Failed to fetch recommendations:', error);
      alert('Error fetching recommendations. Make sure the backend is running!');
    } finally {
      setLoading(false);
    }
  };

  return (
    <>
      <div className="bg-animation"></div>
      <div className="app-container">
        <header className="header animate-fade-in">
          <div className="tech-badges">
            <span className="tech-badge"><span className="dot"></span> Two-Tower Neural Retrieval</span>
            <span className="tech-badge"><span className="dot"></span> LambdaMART Ranking</span>
            <span className="tech-badge"><span className="dot"></span> Claude LLM Re-Ranking</span>
          </div>
          <h1>
            Enterprise-Grade <span className="text-gradient">Job Engine</span>
          </h1>
          <p>A multi-stage hybrid recommendation pipeline integrating semantic matching, collaborative filtering, and LLM reasoning to find your perfect role.</p>
        </header>

        <div className="dashboard-panel animate-fade-in" style={{ animationDelay: '0.1s' }}>
          <div className="dashboard-panel-title">Recommendation Control Panel</div>
          <form className="search-bar" onSubmit={handleSearch}>
            <div className="search-input-wrapper">
              <input 
                type="number" 
                className="search-input" 
                placeholder="Enter User ID to generate pipeline recommendations..." 
                value={userId}
                onChange={(e) => setUserId(e.target.value)}
                min="0"
                required
              />
            </div>
            <button type="submit" className="btn-primary" disabled={loading}>
              {loading ? 'Processing...' : <><Search size={20} /> Generate</>}
            </button>
          </form>
        </div>

        {loading && (
          <div className="spinner-container animate-fade-in">
            <div className="spinner"></div>
            <div style={{ fontWeight: 500 }}>Executing Multi-Stage Pipeline...</div>
          </div>
        )}

        {!loading && response && (
          <div className="animate-fade-in" style={{ animationDelay: '0.2s' }}>
            <div className="results-header">
              <Sparkles className="text-gradient" size={28} />
              <h2>
                Optimized Matches for User <span className="text-gradient">#{response.user_id}</span>
              </h2>
            </div>
            
            <div className="job-grid">
              {response.recommendations.map((rec) => (
                <JobCard key={rec.job_id} recommendation={rec} />
              ))}
            </div>
          </div>
        )}
      </div>
    </>
  );
}

export default App;

