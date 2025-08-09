import React from 'react';

const WeightApplication = ({ config, updateConfig, dataInfo, onNext, onPrev }) => {
  return (
    <div className="space-y-8">
      <div className="text-center">
        <div className="w-20 h-20 bg-gradient-to-br from-green-500 to-green-600 rounded-2xl flex items-center justify-center mx-auto mb-6 shadow-glow-success">
          📊
        </div>
        
        <h2 className="text-3xl font-bold text-gray-900 mb-4">
          Survey Weight Application
        </h2>
        <p className="text-lg text-gray-600 max-w-2xl mx-auto">
          Configure survey weights and estimation variables to produce accurate population estimates.
        </p>
      </div>

      <div className="max-w-6xl mx-auto space-y-8">
        <div className="card p-8">
          <div className="flex items-center space-x-3 mb-6">
            <div className="w-12 h-12 bg-green-100 rounded-xl flex items-center justify-center">
              🧮
            </div>
            <div>
              <h3 className="text-xl font-bold text-gray-900">Weight Configuration</h3>
              <p className="text-gray-600">Select and configure survey weights for accurate estimation</p>
            </div>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            <div>
              <label className="block text-sm font-semibold text-gray-700 mb-3">
                Weight Column
              </label>
              <select
                value={config.weightColumn}
                onChange={(e) => updateConfig('weightColumn', e.target.value)}
                className="input-field"
              >
                <option value="">No weights (equal weights)</option>
                {dataInfo?.columns.map(col => (
                  <option key={col} value={col}>{col}</option>
                ))}
              </select>
              <p className="text-xs text-gray-500 mt-2">
                Leave empty to use equal weights (1.0) for all records
              </p>
            </div>

            <div>
              <label className="block text-sm font-semibold text-gray-700 mb-3">
                Variables to Estimate
              </label>
              <select
                multiple
                value={config.estimateColumns}
                onChange={(e) => updateConfig('estimateColumns', 
                  Array.from(e.target.selectedOptions, option => option.value)
                )}
                className="input-field"
              >
                {dataInfo?.numericColumns.map(col => (
                  <option key={col} value={col}>{col}</option>
                ))}
              </select>
              <p className="text-xs text-gray-500 mt-2">
                Leave empty to estimate all numeric variables
              </p>
            </div>
          </div>
        </div>
      </div>

      <div className="flex justify-between items-center max-w-4xl mx-auto pt-8">
        <button onClick={onPrev} className="btn-secondary">
          ← Previous Step
        </button>
        <button onClick={onNext} className="btn-primary">
          Next: Analysis & Results →
        </button>
      </div>
    </div>
  );
};

export default WeightApplication; 