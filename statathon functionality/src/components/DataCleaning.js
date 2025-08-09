import React from 'react';

const DataCleaning = ({ config, updateConfig, dataInfo, onNext, onPrev }) => {
  const imputationMethods = [
    { value: 'mean', label: 'Mean Imputation', description: 'Replace missing values with column mean' },
    { value: 'median', label: 'Median Imputation', description: 'Replace missing values with column median' },
    { value: 'knn', label: 'K-Nearest Neighbors', description: 'AI-powered imputation using similar records' }
  ];

  const outlierMethods = [
    { value: 'iqr', label: 'Interquartile Range (IQR)', description: 'Detect outliers using Q1-Q3 range' },
    { value: 'zscore', label: 'Z-Score Method', description: 'Detect outliers using standard deviations' }
  ];

  return (
    <div className="space-y-8">
      <div className="text-center">
        <div className="w-20 h-20 bg-gradient-to-br from-yellow-500 to-yellow-600 rounded-2xl flex items-center justify-center mx-auto mb-6 shadow-glow">
          ⚙️
        </div>
        
        <h2 className="text-3xl font-bold text-gray-900 mb-4">
          AI-Enhanced Data Cleaning
        </h2>
        <p className="text-lg text-gray-600 max-w-2xl mx-auto">
          Configure intelligent data cleaning methods to handle missing values and outliers.
        </p>
      </div>

      <div className="max-w-6xl mx-auto space-y-8">
        <div className="card p-8">
          <div className="flex items-center space-x-3 mb-6">
            <div className="w-12 h-12 bg-blue-100 rounded-xl flex items-center justify-center">
              ✨
            </div>
            <div>
              <h3 className="text-xl font-bold text-gray-900">Missing Value Imputation</h3>
              <p className="text-gray-600">Choose how to handle missing data intelligently</p>
            </div>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            <div>
              <label className="block text-sm font-semibold text-gray-700 mb-3">
                Imputation Method
              </label>
              <select
                value={config.imputationMethod}
                onChange={(e) => updateConfig('imputationMethod', e.target.value)}
                className="input-field"
              >
                <option value="">Select method...</option>
                {imputationMethods.map(method => (
                  <option key={method.value} value={method.value}>
                    {method.label}
                  </option>
                ))}
              </select>
            </div>

            <div>
              <label className="block text-sm font-semibold text-gray-700 mb-3">
                Columns to Impute
              </label>
              <select
                multiple
                value={config.imputationColumns}
                onChange={(e) => updateConfig('imputationColumns', 
                  Array.from(e.target.selectedOptions, option => option.value)
                )}
                className="input-field"
              >
                {dataInfo?.columns.map(col => (
                  <option key={col} value={col}>{col}</option>
                ))}
              </select>
              <p className="text-xs text-gray-500 mt-2">
                Leave empty to apply to all numeric columns
              </p>
            </div>
          </div>
        </div>

        <div className="card p-8">
          <div className="flex items-center space-x-3 mb-6">
            <div className="w-12 h-12 bg-yellow-100 rounded-xl flex items-center justify-center">
              🛡️
            </div>
            <div>
              <h3 className="text-xl font-bold text-gray-900">Outlier Detection & Handling</h3>
              <p className="text-gray-600">Identify and handle statistical outliers</p>
            </div>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
            <div>
              <label className="block text-sm font-semibold text-gray-700 mb-3">
                Detection Method
              </label>
              <select
                value={config.outlierMethod}
                onChange={(e) => updateConfig('outlierMethod', e.target.value)}
                className="input-field"
              >
                <option value="">Select method...</option>
                {outlierMethods.map(method => (
                  <option key={method.value} value={method.value}>
                    {method.label}
                  </option>
                ))}
              </select>
            </div>

            <div>
              <label className="block text-sm font-semibold text-gray-700 mb-3">
                Handling Strategy
              </label>
              <select
                value={config.outlierHandling}
                onChange={(e) => updateConfig('outlierHandling', e.target.value)}
                className="input-field"
              >
                <option value="winsorize">Winsorize (Cap at percentiles)</option>
                <option value="remove">Remove Outliers</option>
              </select>
            </div>

            <div>
              <label className="block text-sm font-semibold text-gray-700 mb-3">
                Columns to Check
              </label>
              <select
                multiple
                value={config.outlierColumns}
                onChange={(e) => updateConfig('outlierColumns', 
                  Array.from(e.target.selectedOptions, option => option.value)
                )}
                className="input-field"
              >
                {dataInfo?.numericColumns.map(col => (
                  <option key={col} value={col}>{col}</option>
                ))}
              </select>
            </div>
          </div>
        </div>
      </div>

      <div className="flex justify-between items-center max-w-4xl mx-auto pt-8">
        <button onClick={onPrev} className="btn-secondary">
          ← Previous Step
        </button>
        <button onClick={onNext} className="btn-primary">
          Next: Weight Application →
        </button>
      </div>
    </div>
  );
};

export default DataCleaning; 