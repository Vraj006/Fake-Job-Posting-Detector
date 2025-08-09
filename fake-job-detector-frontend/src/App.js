import React, { useState } from 'react';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faMagnifyingGlass } from '@fortawesome/free-solid-svg-icons';

const FakeJobDetector = () => {
  const [formData, setFormData] = useState({
    title: '',
    description: '',
    company_profile: '',
    requirements: '',
    benefits: ''
  });
  
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [activeTab, setActiveTab] = useState('analyze');

  const handleInputChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value
    });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      // Make API call to Flask backend
      const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000';
      const response = await fetch(`${API_URL}/predict`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(formData)
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
      }

      const result = await response.json();
      setResult(result);
      setActiveTab('results');
    } catch (err) {
      setError(`API Error: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  const handleClearForm = () => {
    setFormData({
      title: '',
      description: '',
      company_profile: '',
      requirements: '',
      benefits: ''
    });
    setResult(null);
    setError(null);
    setActiveTab('analyze');
  };

  const getRiskColorClass = (level) => {
    switch(level) {
      case 'LOW': return 'risk-low';
      case 'MEDIUM': return 'risk-medium';
      case 'HIGH': return 'risk-high';
      default: return 'risk-default';
    }
  };

  const styles = {
    app: {
      minHeight: '100vh',
      background: 'linear-gradient(135deg, #0f0f23 0%, #1a1a2e 25%, #16213e 50%, #0f0f23 100%)',
      fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
      color: '#ffffff',
      position: 'relative',
      overflow: 'hidden'
    },
    backgroundPattern: {
      position: 'absolute',
      top: 0,
      left: 0,
      right: 0,
      bottom: 0,
      backgroundImage: `
        radial-gradient(circle at 25% 25%, rgba(120, 119, 198, 0.15) 0%, transparent 50%),
        radial-gradient(circle at 75% 75%, rgba(255, 107, 107, 0.15) 0%, transparent 50%),
        radial-gradient(circle at 50% 50%, rgba(72, 187, 120, 0.1) 0%, transparent 50%)
      `,
      pointerEvents: 'none'
    },
    container: {
      maxWidth: '1400px',
      margin: '0 auto',
      padding: '2rem',
      position: 'relative',
      zIndex: 1
    },
    header: {
      textAlign: 'center',
      marginBottom: '3rem',
      padding: '2rem 0'
    },
    title: {
      fontSize: '4rem',
      fontWeight: '800',
      background: 'linear-gradient(135deg, #ffffff 0%, #e0e7ff 30%, #c7d2fe 60%, #a5b4fc 100%)',
      WebkitBackgroundClip: 'text',
      WebkitTextFillColor: 'transparent',
      backgroundClip: 'text',
      marginBottom: '1rem',
      textShadow: '0 4px 8px rgba(0,0,0,0.3)'
    },
    subtitle: {
      fontSize: '1.5rem',
      marginBottom: '2rem',
      opacity: 0.9,
      maxWidth: '600px',
      margin: '0 auto 2rem auto',
      lineHeight: 1.6
    },
    stats: {
      display: 'flex',
      justifyContent: 'center',
      gap: '2rem',
      flexWrap: 'wrap',
      marginTop: '2rem'
    },
    statItem: {
      background: 'rgba(255, 255, 255, 0.1)',
      backdropFilter: 'blur(10px)',
      border: '1px solid rgba(255, 255, 255, 0.2)',
      borderRadius: '12px',
      padding: '1rem 1.5rem',
      fontSize: '0.9rem',
      fontWeight: '500',
      transition: 'all 0.3s ease',
      cursor: 'default'
    },
    tabNavigation: {
      display: 'flex',
      justifyContent: 'center',
      marginBottom: '2rem',
      gap: '0.5rem'
    },
    tab: {
      padding: '1rem 2rem',
      background: 'rgba(255, 255, 255, 0.1)',
      border: '1px solid rgba(255, 255, 255, 0.2)',
      borderRadius: '12px',
      color: '#ffffff',
      cursor: 'pointer',
      transition: 'all 0.3s ease',
      fontWeight: '500',
      fontSize: '1rem'
    },
    tabActive: {
      background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
      borderColor: 'rgba(255, 255, 255, 0.3)',
      transform: 'translateY(-2px)',
      boxShadow: '0 8px 25px rgba(102, 126, 234, 0.4)'
    },
    mainContent: {
      display: 'grid',
      gridTemplateColumns: '1fr',
      gap: '2rem',
      maxWidth: '900px',
      margin: '0 auto'
    },
    card: {
      background: 'rgba(255, 255, 255, 0.05)',
      backdropFilter: 'blur(20px)',
      border: '1px solid rgba(255, 255, 255, 0.1)',
      borderRadius: '20px',
      padding: '2rem',
      boxShadow: '0 20px 40px rgba(0, 0, 0, 0.3)',
      transition: 'all 0.3s ease'
    },
    formGroup: {
      marginBottom: '1.5rem'
    },
    label: {
      display: 'block',
      fontSize: '1rem',
      fontWeight: '600',
      color: '#e2e8f0',
      marginBottom: '0.5rem',
      textTransform: 'uppercase',
      letterSpacing: '0.5px',
      fontSize: '0.875rem'
    },
    input: {
      width: '100%',
      padding: '1rem 1.5rem',
      background: 'rgba(255, 255, 255, 0.05)',
      border: '2px solid rgba(255, 255, 255, 0.1)',
      borderRadius: '12px',
      color: '#ffffff',
      fontSize: '1rem',
      transition: 'all 0.3s ease',
      backdropFilter: 'blur(10px)',
      boxSizing: 'border-box'
    },
    inputFocus: {
      borderColor: '#667eea',
      boxShadow: '0 0 0 3px rgba(102, 126, 234, 0.2)',
      background: 'rgba(255, 255, 255, 0.08)'
    },
    textarea: {
      minHeight: '120px',
      resize: 'vertical',
      fontFamily: 'inherit',
      boxSizing: 'border-box'
    },
    buttonGroup: {
      display: 'flex',
      gap: '1rem',
      marginTop: '2rem'
    },
    btnPrimary: {
      flex: 1,
      padding: '1.25rem 2rem',
      background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
      border: 'none',
      borderRadius: '12px',
      color: 'white',
      fontWeight: '600',
      fontSize: '1.1rem',
      cursor: 'pointer',
      transition: 'all 0.3s ease',
      textTransform: 'uppercase',
      letterSpacing: '0.5px',
      position: 'relative',
      overflow: 'hidden'
    },
    btnPrimaryHover: {
      transform: 'translateY(-2px)',
      boxShadow: '0 15px 35px rgba(102, 126, 234, 0.4)'
    },
    btnSecondary: {
      padding: '1.25rem 2rem',
      background: 'rgba(255, 255, 255, 0.1)',
      border: '2px solid rgba(255, 255, 255, 0.2)',
      borderRadius: '12px',
      color: 'white',
      cursor: 'pointer',
      transition: 'all 0.3s ease',
      fontWeight: '500',
      fontSize: '1rem'
    },
    loadingSpinner: {
      display: 'inline-block',
      width: '20px',
      height: '20px',
      border: '2px solid rgba(255, 255, 255, 0.3)',
      borderTop: '2px solid white',
      borderRadius: '50%',
      animation: 'spin 1s linear infinite',
      marginRight: '0.5rem'
    },
    errorBox: {
      background: 'linear-gradient(135deg, rgba(239, 68, 68, 0.2) 0%, rgba(220, 38, 38, 0.1) 100%)',
      border: '1px solid rgba(239, 68, 68, 0.3)',
      borderRadius: '12px',
      padding: '1.5rem',
      marginBottom: '1.5rem',
      color: '#fca5a5',
      backdropFilter: 'blur(10px)'
    },
    predictionBox: {
      borderRadius: '16px',
      padding: '2rem',
      marginBottom: '1.5rem',
      backdropFilter: 'blur(10px)',
      border: '2px solid'
    },
    riskLow: {
      background: 'linear-gradient(135deg, rgba(34, 197, 94, 0.2) 0%, rgba(21, 128, 61, 0.1) 100%)',
      borderColor: 'rgba(34, 197, 94, 0.4)',
      color: '#86efac'
    },
    riskMedium: {
      background: 'linear-gradient(135deg, rgba(245, 158, 11, 0.2) 0%, rgba(217, 119, 6, 0.1) 100%)',
      borderColor: 'rgba(245, 158, 11, 0.4)',
      color: '#fcd34d'
    },
    riskHigh: {
      background: 'linear-gradient(135deg, rgba(239, 68, 68, 0.2) 0%, rgba(220, 38, 38, 0.1) 100%)',
      borderColor: 'rgba(239, 68, 68, 0.4)',
      color: '#fca5a5'
    },
    predictionHeader: {
      display: 'flex',
      justifyContent: 'space-between',
      alignItems: 'center',
      marginBottom: '1rem'
    },
    predictionLabel: {
      fontSize: '2rem',
      fontWeight: '800',
      textTransform: 'uppercase',
      letterSpacing: '1px'
    },
    riskBadge: {
      fontSize: '0.875rem',
      fontWeight: '600',
      padding: '0.5rem 1rem',
      borderRadius: '20px',
      background: 'rgba(255, 255, 255, 0.2)',
      textTransform: 'uppercase',
      letterSpacing: '0.5px'
    },
    scoresSection: {
      background: 'rgba(255, 255, 255, 0.05)',
      borderRadius: '16px',
      padding: '1.5rem',
      marginBottom: '1.5rem',
      backdropFilter: 'blur(10px)'
    },
    sectionTitle: {
      fontSize: '1.25rem',
      fontWeight: '700',
      marginBottom: '1.5rem',
      color: '#e2e8f0',
      textTransform: 'uppercase',
      letterSpacing: '1px'
    },
    scoreItem: {
      marginBottom: '1.5rem'
    },
    scoreHeader: {
      display: 'flex',
      justifyContent: 'space-between',
      alignItems: 'center',
      marginBottom: '0.75rem'
    },
    progressBar: {
      width: '100%',
      height: '12px',
      background: 'rgba(255, 255, 255, 0.1)',
      borderRadius: '6px',
      overflow: 'hidden',
      position: 'relative'
    },
    progressFill: {
      height: '100%',
      borderRadius: '6px',
      transition: 'width 0.8s ease',
      position: 'relative'
    },
    progressFillGlow: {
      position: 'absolute',
      top: 0,
      left: 0,
      right: 0,
      bottom: 0,
      borderRadius: '6px',
      boxShadow: 'inset 0 0 10px rgba(255, 255, 255, 0.3)'
    },
    analysisGrid: {
      display: 'grid',
      gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
      gap: '1rem',
      marginTop: '1rem'
    },
    analysisItem: {
      background: 'rgba(255, 255, 255, 0.05)',
      borderRadius: '12px',
      padding: '1rem',
      display: 'flex',
      justifyContent: 'space-between',
      alignItems: 'center',
      backdropFilter: 'blur(10px)',
      border: '1px solid rgba(255, 255, 255, 0.1)'
    },
    emptyState: {
      textAlign: 'center',
      padding: '4rem 2rem',
      color: '#94a3b8'
    },
    emptyIcon: {
      fontSize: '5rem',
      marginBottom: '2rem',
      background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
      WebkitBackgroundClip: 'text',
      WebkitTextFillColor: 'transparent',
      backgroundClip: 'text'
    },
    footer: {
      textAlign: 'center',
      marginTop: '3rem',
      padding: '2rem',
      color: '#94a3b8',
      fontSize: '0.9rem'
    }
  };

  return (
    <div style={styles.app}>
      <div style={styles.backgroundPattern}></div>
      
      <style>{`
        @keyframes spin {
          0% { transform: rotate(0deg); }
          100% { transform: rotate(360deg); }
        }
        
        input::placeholder,
        textarea::placeholder {
          color: rgba(255, 255, 255, 0.5);
        }
        
        input:focus,
        textarea:focus {
          outline: none;
          border-color: #667eea !important;
          box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.2) !important;
          background: rgba(255, 255, 255, 0.08) !important;
        }
        
        button:hover:not(:disabled) {
          transform: translateY(-2px);
        }
        
        .stat-item:hover {
          transform: translateY(-2px);
          box-shadow: 0 8px 25px rgba(255, 255, 255, 0.1);
        }
        
        .analysis-item:hover {
          background: rgba(255, 255, 255, 0.08);
        }
        
        @media (max-width: 768px) {
          .container {
            padding: 1rem;
          }
          
          .title {
            font-size: 2.5rem !important;
          }
          
          .subtitle {
            font-size: 1.2rem !important;
          }
          
          .stats {
            gap: 1rem !important;
          }
          
          .button-group {
            flex-direction: column;
          }
          
          .analysis-grid {
            grid-template-columns: 1fr !important;
          }
        }
      `}</style>
      
      <div style={styles.container}>
        <div style={styles.header}>
          <h1 style={styles.title}>
            <FontAwesomeIcon icon={faMagnifyingGlass} style={{marginRight: '0.5rem',fontSize:'3.2rem'}} />
            Fake Job Posting Classifier
          </h1>
          <p style={styles.subtitle}>
            Advanced machine learning system to identify and classify fraudulent job postings, protecting job seekers from scams.
          </p>
          <div style={styles.stats}>
            <div style={{...styles.statItem}} className="stat-item">
              <strong>🧠 ML Powered</strong>
            </div>
            <div style={{...styles.statItem}} className="stat-item">
              <strong>📈 97.93% Accuracy</strong>
            </div>
            <div style={{...styles.statItem}} className="stat-item">
              <strong>⚡ Real-time Detection</strong>
            </div>
            <div style={{...styles.statItem}} className="stat-item">
              <strong>🔒 Fraud Prevention</strong>
            </div>
          </div>
        </div>

        <div style={styles.tabNavigation}>
          <button 
            style={{...styles.tab, ...(activeTab === 'analyze' ? styles.tabActive : {})}}
            onClick={() => setActiveTab('analyze')}
          >
            📊 Analyze Job
          </button>
          <button 
            style={{...styles.tab, ...(activeTab === 'results' ? styles.tabActive : {})}}
            onClick={() => setActiveTab('results')}
          >
            📋 Results
          </button>
        </div>

        <div style={styles.mainContent}>
          {activeTab === 'analyze' && (
            <div style={styles.card}>
              <h2 style={styles.sectionTitle}>Job Posting Analysis</h2>
              
              <div>
                <div style={styles.formGroup}>
                  <label style={styles.label}>Job Title *</label>
                  <input
                    style={styles.input}
                    type="text"
                    name="title"
                    value={formData.title}
                    onChange={handleInputChange}
                    placeholder="e.g., Senior Software Engineer, Marketing Manager"
                    required
                  />
                </div>

                <div style={styles.formGroup}>
                  <label style={styles.label}>Job Description *</label>
                  <textarea
                    style={{...styles.input, ...styles.textarea}}
                    name="description"
                    value={formData.description}
                    onChange={handleInputChange}
                    rows={4}
                    placeholder="Describe the job responsibilities, requirements, and details..."
                    required
                  />
                </div>

                <div style={styles.formGroup}>
                  <label style={styles.label}>Company Profile</label>
                  <textarea
                    style={{...styles.input, ...styles.textarea}}
                    name="company_profile"
                    value={formData.company_profile}
                    onChange={handleInputChange}
                    rows={3}
                    placeholder="Information about the company, its mission, and background..."
                  />
                </div>

                <div style={styles.formGroup}>
                  <label style={styles.label}>Requirements</label>
                  <textarea
                    style={{...styles.input, ...styles.textarea}}
                    name="requirements"
                    value={formData.requirements}
                    onChange={handleInputChange}
                    rows={3}
                    placeholder="Job requirements, qualifications, and skills needed..."
                  />
                </div>

                <div style={styles.formGroup}>
                  <label style={styles.label}>Benefits & Compensation</label>
                  <textarea
                    style={{...styles.input, ...styles.textarea}}
                    name="benefits"
                    value={formData.benefits}
                    onChange={handleInputChange}
                    rows={2}
                    placeholder="Salary range, benefits, perks, and compensation details..."
                  />
                </div>

                <div style={styles.buttonGroup}>
                  <button
                    onClick={handleSubmit}
                    disabled={loading || !formData.title || !formData.description}
                    style={{
                      ...styles.btnPrimary,
                      opacity: (loading || !formData.title || !formData.description) ? 0.5 : 1,
                      cursor: (loading || !formData.title || !formData.description) ? 'not-allowed' : 'pointer'
                    }}
                  >
                    {loading && <span style={styles.loadingSpinner}></span>}
                    {loading ? 'Classifying Posting...' : 'Classify Job Posting'}
                  </button>
                  
                  <button
                    onClick={handleClearForm}
                    style={styles.btnSecondary}
                  >
                    Clear Form
                  </button>
                </div>
              </div>
            </div>
          )}

          {activeTab === 'results' && (
            <div style={styles.card}>
              <h2 style={styles.sectionTitle}>ML Classification Results</h2>
              
              {error && (
                <div style={styles.errorBox}>
                  <strong>❌ Classification Error:</strong> {error}
                </div>
              )}

              {result ? (
                <div>
                  <div style={{
                    ...styles.predictionBox,
                    ...(result.risk_assessment.level === 'LOW' ? styles.riskLow :
                        result.risk_assessment.level === 'MEDIUM' ? styles.riskMedium :
                        styles.riskHigh)
                  }}>
                    <div style={styles.predictionHeader}>
                      <span style={styles.predictionLabel}>
                        {result.prediction.label === 'REAL' ? '✅ LEGITIMATE' : '⚠️ SUSPICIOUS'}
                      </span>
                      <span style={styles.riskBadge}>{result.risk_assessment.level} RISK</span>
                    </div>
                    <p>{result.risk_assessment.confidence}</p>
                  </div>

                  <div style={styles.scoresSection}>
                    <h3 style={styles.sectionTitle}>🎯 Confidence Metrics</h3>
                    
                    <div style={styles.scoreItem}>
                      <div style={styles.scoreHeader}>
                        <span style={{fontSize: '1.1rem', fontWeight: '600'}}>Fraud Probability</span>
                        <span style={{color: '#fca5a5', fontWeight: '700', fontSize: '1.2rem'}}>
                          {(result.prediction.fake_probability * 100).toFixed(1)}%
                        </span>
                      </div>
                      <div style={styles.progressBar}>
                        <div 
                          style={{
                            ...styles.progressFill,
                            width: `${result.prediction.fake_probability * 100}%`,
                            background: 'linear-gradient(90deg, #ef4444 0%, #dc2626 100%)'
                          }}
                        >
                          <div style={styles.progressFillGlow}></div>
                        </div>
                      </div>
                    </div>

                    <div style={styles.scoreItem}>
                      <div style={styles.scoreHeader}>
                        <span style={{fontSize: '1.1rem', fontWeight: '600'}}>Legitimacy Probability</span>
                        <span style={{color: '#86efac', fontWeight: '700', fontSize: '1.2rem'}}>
                          {(result.prediction.real_probability * 100).toFixed(1)}%
                        </span>
                      </div>
                      <div style={styles.progressBar}>
                        <div 
                          style={{
                            ...styles.progressFill,
                            width: `${result.prediction.real_probability * 100}%`,
                            background: 'linear-gradient(90deg, #22c55e 0%, #16a34a 100%)'
                          }}
                        >
                          <div style={styles.progressFillGlow}></div>
                        </div>
                      </div>
                    </div>
                  </div>

                  <div style={styles.scoresSection}>
                    <h3 style={styles.sectionTitle}>📈 Content Analysis</h3>
                    <div style={styles.analysisGrid}>
                      <div style={styles.analysisItem} className="analysis-item">
                        <span>📝 Title Length</span>
                        <span style={{fontWeight: '600', color: '#e2e8f0'}}>
                          {result.input_analysis.title_length} chars
                        </span>
                      </div>
                      <div style={styles.analysisItem} className="analysis-item">
                        <span>📄 Description</span>
                        <span style={{fontWeight: '600', color: '#e2e8f0'}}>
                          {result.input_analysis.description_length} chars
                        </span>
                      </div>
                      <div style={styles.analysisItem} className="analysis-item">
                        <span>🔢 Word Count</span>
                        <span style={{fontWeight: '600', color: '#e2e8f0'}}>
                          {result.input_analysis.word_count} words
                        </span>
                      </div>
                      <div style={styles.analysisItem} className="analysis-item">
                        <span>🏢 Company Info</span>
                        <span style={{
                          fontWeight: '600',
                          color: result.input_analysis.has_company_info ? '#86efac' : '#fca5a5'
                        }}>
                          {result.input_analysis.has_company_info ? '✅ Present' : '❌ Missing'}
                        </span>
                      </div>
                    </div>
                  </div>

                  <div style={{textAlign: 'center', marginTop: '2rem', color: '#94a3b8', fontSize: '0.9rem'}}>
                    Classification completed at {new Date(result.timestamp).toLocaleString()}
                  </div>
                </div>
              ) : (
                <div style={styles.emptyState}>
                  <div style={styles.emptyIcon}>🛡️</div>
                  <h3 style={{marginBottom: '1rem', fontSize: '1.5rem'}}>Ready for Classification</h3>
                  <p style={{marginBottom: '0.5rem', fontSize: '1.1rem'}}>Enter a job posting to get started</p>
                  <p style={{color: '#64748b'}}>
                    Our ML classifier will evaluate the posting for fraud indicators using statistical analysis
                  </p>
                </div>
              )}
            </div>
          )}
        </div>

        <div style={styles.footer}>
          <p>Powered by Machine Learning Classification • SVM with TF-IDF Vectorization • 97.93% Accuracy Rate</p>
          <p style={{marginTop: '0.5rem', opacity: 0.7}}>Protecting job seekers from fraudulent postings</p>
        </div>
      </div>
    </div>
  );
};

export default FakeJobDetector;