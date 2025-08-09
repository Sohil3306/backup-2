import React from 'react';

const AnalysisResults = ({ analysisResults, dataInfo, config, onRunAnalysis, onGenerateReport, loading, onPrev, detailed = false }) => {
  const formatNumber = (num) => {
    if (typeof num !== 'number') return 'N/A';
    return num.toLocaleString('en-US', { 
      minimumFractionDigits: 2, 
      maximumFractionDigits: 4 
    });
  };

  if (!analysisResults && !detailed) {
    return (
      <div className="space-y-8">
        <div className="text-center">
          <div className="w-20 h-20 bg-gradient-to-br from-blue-500 to-blue-600 rounded-2xl flex items-center justify-center mx-auto mb-6 shadow-glow">
            📊
          </div>
          
          <h2 className="text-3xl font-bold text-gray-900 mb-4">
            Analysis & Results
          </h2>
          <p className="text-lg text-gray-600 max-w-2xl mx-auto">
            Run AI-enhanced analysis to generate statistical estimates, 
            detect patterns, and produce professional reports.
          </p>
        </div>

        <div className="max-w-4xl mx-auto">
          <div className="card p-8 text-center">
            <div className="w-16 h-16 bg-blue-100 rounded-xl flex items-center justify-center mx-auto mb-4">
              🧠
            </div>
            <h3 className="text-xl font-bold text-gray-900 mb-2">Ready for Analysis</h3>
            <p className="text-gray-600 mb-6">
              Your data is prepared and configured. Click the button below to start the AI-enhanced analysis.
            </p>
            <button
              onClick={onRunAnalysis}
              disabled={loading}
              className="btn-primary"
            >
              {loading ? (
                <div className="flex items-center space-x-2">
                  <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                  <span>Running Analysis...</span>
                </div>
              ) : (
                <div className="flex items-center space-x-2">
                  ✨
                  <span>Run AI-Enhanced Analysis</span>
                </div>
              )}
            </button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-8">
      <div className="text-center">
        <div className="w-20 h-20 bg-gradient-to-br from-green-500 to-green-600 rounded-2xl flex items-center justify-center mx-auto mb-6 shadow-glow-success">
          ✅
        </div>
        
        <h2 className="text-3xl font-bold text-gray-900 mb-4">
          Analysis Results
        </h2>
        <p className="text-lg text-gray-600 max-w-2xl mx-auto">
          AI-enhanced analysis completed successfully. Review your statistical estimates and generate professional reports.
        </p>
      </div>

      {analysisResults && (
        <div className="max-w-6xl mx-auto">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            <div className="card p-6 text-center">
              <div className="w-12 h-12 bg-blue-100 rounded-xl flex items-center justify-center mx-auto mb-4">
                👥
              </div>
              <p className="text-2xl font-bold text-blue-800">
                {analysisResults.cleanedShape ? analysisResults.cleanedShape[0].toLocaleString() : 'N/A'}
              </p>
              <p className="text-sm text-gray-600">Records Analyzed</p>
            </div>

            <div className="card p-6 text-center">
              <div className="w-12 h-12 bg-green-100 rounded-xl flex items-center justify-center mx-auto mb-4">
                🎯
              </div>
              <p className="text-2xl font-bold text-green-800">
                {Object.keys(analysisResults.estimates || {}).length}
              </p>
              <p className="text-sm text-gray-600">Variables Estimated</p>
            </div>

            <div className="card p-6 text-center">
              <div className="w-12 h-12 bg-yellow-100 rounded-xl flex items-center justify-center mx-auto mb-4">
                ⚠️
              </div>
              <p className="text-2xl font-bold text-yellow-800">
                {analysisResults.missingInfo?.totalMissing || 0}
              </p>
              <p className="text-sm text-gray-600">Missing Values</p>
            </div>

            <div className="card p-6 text-center">
              <div className="w-12 h-12 bg-red-100 rounded-xl flex items-center justify-center mx-auto mb-4">
                ⏱️
              </div>
              <p className="text-2xl font-bold text-red-800">
                {analysisResults.processingTime ? `${Math.round(analysisResults.processingTime)}ms` : 'N/A'}
              </p>
              <p className="text-sm text-gray-600">Processing Time</p>
            </div>
          </div>

          {analysisResults.estimates && (
            <div className="card p-8 mt-8">
              <div className="flex items-center space-x-3 mb-6">
                <div className="w-12 h-12 bg-blue-100 rounded-xl flex items-center justify-center">
                  📈
                </div>
                <div>
                  <h3 className="text-xl font-bold text-gray-900">Statistical Estimates</h3>
                  <p className="text-gray-600">Population estimates with margins of error</p>
                </div>
              </div>

              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="bg-gray-50">
                      <th className="px-4 py-3 text-left font-semibold text-gray-700">Variable</th>
                      <th className="px-4 py-3 text-center font-semibold text-gray-700">Unweighted Mean</th>
                      <th className="px-4 py-3 text-center font-semibold text-gray-700">Weighted Mean</th>
                      <th className="px-4 py-3 text-center font-semibold text-gray-700">Margin of Error</th>
                      <th className="px-4 py-3 text-center font-semibold text-gray-700">Standard Error</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(analysisResults.estimates).map(([variable, est]) => (
                      <tr key={variable} className="border-b border-gray-100 hover:bg-gray-50">
                        <td className="px-4 py-3 font-medium text-gray-900">{variable}</td>
                        <td className="px-4 py-3 text-center text-gray-700">
                          {formatNumber(est.unweighted.mean)}
                        </td>
                        <td className="px-4 py-3 text-center text-blue-700 font-semibold">
                          {formatNumber(est.weighted.mean)}
                        </td>
                        <td className="px-4 py-3 text-center text-yellow-700">
                          ±{formatNumber(est.weighted.moe)}
                        </td>
                        <td className="px-4 py-3 text-center text-gray-600">
                          {formatNumber(est.weighted.se)}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      )}

      <div className="max-w-4xl mx-auto">
        <div className="card p-8 text-center">
          <div className="w-16 h-16 bg-green-100 rounded-xl flex items-center justify-center mx-auto mb-4">
            📄
          </div>
          <h3 className="text-xl font-bold text-gray-900 mb-2">Generate Professional Report</h3>
          <p className="text-gray-600 mb-6">
            Create a comprehensive PDF report with all analysis results, methodology, and visualizations.
          </p>
          <button
            onClick={onGenerateReport}
            className="btn-success"
          >
            <div className="flex items-center space-x-2">
              📥
              <span>Generate PDF Report</span>
            </div>
          </button>
        </div>
      </div>

      <div className="flex justify-between items-center max-w-4xl mx-auto pt-8">
        <button onClick={onPrev} className="btn-secondary">
          ← Previous Step
        </button>
        <div className="flex items-center space-x-2">
          <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
          <span className="text-sm text-green-600 font-medium">Analysis Complete</span>
        </div>
      </div>
    </div>
  );
};

export default AnalysisResults; 