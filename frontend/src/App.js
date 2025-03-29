import React, { useState } from 'react';
import './App.css';

function App() {
  const [formData, setFormData] = useState({
    ageGroup: 1,  // Default to Adult
    itching: 0,
    nodalSkinEruptions: 0,
    shivering: 0,
    stomachPain: 0,
    vomiting: 0,
    chestPain: 0,
    lossOfAppetite: 0,
    yellowUrine: 0,
    restlessness: 0,
    excessiveHunger: 0,
    highFever: 0,
    diarrhoea: 0,
    redSpotsOverBody: 0,
    breathlessness: 0,
    darkUrine: 0,
    skinRash: 0,
    continuousSneezing: 0,
    chills: 0,
    ulcersOnTongue: 0,
    cough: 0,
    yellowishSkin: 0,
    abdominalPain: 0,
    weightLoss: 0,
    irregularSugarLevel: 0,
    increasedAppetite: 0,
    headache: 0,
    musclePain: 0,
    runnyNose: 0,
    fastHeartRate: 0,
    duration: 0,
    allergies: 0,
    fatigue: 0,
    additionalInformation: 1  // Always set to 1 (Yes)
  });

  const [prediction, setPrediction] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [theme, setTheme] = useState('light'); // 'light', 'dark', or 'eye-care'
  const [showThemeMenu, setShowThemeMenu] = useState(false);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: parseInt(value)
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError('');
    setPrediction('');

    try {
      const response = await fetch(`${process.env.REACT_APP_BACKEND_URL || 'http://localhost:5000'}/predict`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          features: Object.values(formData)
        }),
      });

      const data = await response.json();
      
      if (data.error) {
        setError(data.error);
        return;
      }

      setPrediction(data.prediction);
    } catch (err) {
      setError('Failed to get prediction. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const renderInput = (name, label, type = 'toggle') => {
    if (type === 'toggle') {
      return (
        <div className="form-group">
          <label htmlFor={name}>{label}</label>
          <div className="toggle-container">
            <button
              type="button"
              className={`toggle-button ${formData[name] === 0 ? 'active' : ''}`}
              onClick={() => handleChange({ target: { name, value: 0 } })}
            >
              No
            </button>
            <button
              type="button"
              className={`toggle-button ${formData[name] === 1 ? 'active' : ''}`}
              onClick={() => handleChange({ target: { name, value: 1 } })}
            >
              Yes
            </button>
          </div>
        </div>
      );
    } else if (type === 'number') {
      return (
        <div className="form-group">
          <label htmlFor={name}>{label}</label>
          <input
            type="number"
            id={name}
            name={name}
            value={formData[name]}
            onChange={handleChange}
            min="0"
            className="form-control"
          />
        </div>
      );
    }
  };

  return (
    <div className={`app ${theme}`}>
      <header className="app-header">
        <h1>Medicine Recommendation System</h1>
        <div className="theme-controls">
          <button 
            className="theme-toggle"
            onClick={() => setShowThemeMenu(!showThemeMenu)}
          >
            {theme === 'light' ? '☀️' : theme === 'dark' ? '🌙' : '👁️'}
          </button>
          {showThemeMenu && (
            <div className="theme-menu">
              <button 
                className={`theme-option ${theme === 'light' ? 'active' : ''}`}
                onClick={() => {
                  setTheme('light');
                  setShowThemeMenu(false);
                }}
              >
                Light Mode ☀️
              </button>
              <button 
                className={`theme-option ${theme === 'dark' ? 'active' : ''}`}
                onClick={() => {
                  setTheme('dark');
                  setShowThemeMenu(false);
                }}
              >
                Dark Mode 🌙
              </button>
              <button 
                className={`theme-option ${theme === 'eye-care' ? 'active' : ''}`}
                onClick={() => {
                  setTheme('eye-care');
                  setShowThemeMenu(false);
                }}
              >
                Eye Care Mode 👁️
              </button>
            </div>
          )}
        </div>
        <p className="subtitle">Get personalized medicine recommendations based on your symptoms</p>
      </header>

      <main className="app-main">
        <div className="form-container">
          <form onSubmit={handleSubmit} className="symptom-form">
            <div className="form-section">
              <h2>Personal Information</h2>
              <div className="form-group">
                <label htmlFor="ageGroup">Age Group</label>
                <select
                  id="ageGroup"
                  name="ageGroup"
                  value={formData.ageGroup}
                  onChange={handleChange}
                  className="form-control"
                >
                  <option value="0">Child</option>
                  <option value="1">Adult</option>
                  <option value="2">Senior</option>
                </select>
              </div>
            </div>

            <div className="form-section">
              <h2>Common Symptoms</h2>
              <div className="symptoms-grid">
                {renderInput('highFever', 'High Fever')}
                {renderInput('headache', 'Headache')}
                {renderInput('cough', 'Cough')}
                {renderInput('runnyNose', 'Runny Nose')}
                {renderInput('chestPain', 'Chest Pain')}
                {renderInput('vomiting', 'Vomiting')}
                {renderInput('diarrhoea', 'Diarrhoea')}
                {renderInput('fatigue', 'Fatigue')}
                {renderInput('breathlessness', 'Difficulty Breathing')}
                {renderInput('musclePain', 'Body Pain')}
              </div>
            </div>

            <div className="form-section">
              <h2>Additional Symptoms</h2>
              <div className="symptoms-grid">
                {renderInput('itching', 'Itching')}
                {renderInput('nodalSkinEruptions', 'Nodal Skin Eruptions')}
                {renderInput('shivering', 'Shivering')}
                {renderInput('stomachPain', 'Stomach Pain')}
                {renderInput('lossOfAppetite', 'Loss of Appetite')}
                {renderInput('yellowUrine', 'Yellow Urine')}
                {renderInput('restlessness', 'Restlessness')}
                {renderInput('excessiveHunger', 'Excessive Hunger')}
                {renderInput('redSpotsOverBody', 'Red Spots Over Body')}
                {renderInput('darkUrine', 'Dark Urine')}
                {renderInput('skinRash', 'Skin Rash')}
                {renderInput('continuousSneezing', 'Continuous Sneezing')}
                {renderInput('chills', 'Chills')}
                {renderInput('ulcersOnTongue', 'Ulcers on Tongue')}
                {renderInput('yellowishSkin', 'Yellowish Skin')}
                {renderInput('abdominalPain', 'Abdominal Pain')}
                {renderInput('weightLoss', 'Weight Loss')}
                {renderInput('irregularSugarLevel', 'Irregular Sugar Level')}
                {renderInput('increasedAppetite', 'Increased Appetite')}
                {renderInput('fastHeartRate', 'Fast Heart Rate')}
              </div>
            </div>

            <div className="form-section">
              <h2>Additional Information</h2>
              <div className="additional-info-grid">
                {renderInput('duration', 'Duration (days)', 'number')}
                {renderInput('allergies', 'Allergies')}
              </div>
            </div>

            <button type="submit" className="submit-button" disabled={loading}>
              {loading ? 'Getting Recommendations...' : 'Get Medicine Recommendations'}
            </button>
          </form>
        </div>

        {(prediction || error) && (
          <div className="prediction-container">
            {error ? (
              <div className="error-message">{error}</div>
            ) : (
              <div className="prediction-content">
                {prediction.split('\n').map((line, index) => {
                  if (line.startsWith('Severity Level:')) {
                    return <h3 key={index} className="severity-level">{line}</h3>;
                  } else if (line.startsWith('Recommended Medicines:')) {
                    return <h3 key={index} className="recommendations-title">{line}</h3>;
                  } else if (line.startsWith('•')) {
                    return <div key={index} className="medicine-item">{line}</div>;
                  } else if (line.startsWith('Note:')) {
                    return <div key={index} className="note">{line}</div>;
                  } else if (line === '') {
                    return <hr key={index} className="divider" />;
                  }
                  return <div key={index}>{line}</div>;
                })}
              </div>
            )}
          </div>
        )}
      </main>
    </div>
  );
}

export default App;
