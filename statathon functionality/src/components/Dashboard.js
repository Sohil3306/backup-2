import React from 'react';

const Dashboard = ({ analytics, dataInfo, analysisResults, config }) => {
  const formatNumber = (num) => {
    if (num >= 1000000) return (num / 1000000).toFixed(1) + 'M';
    if (num >= 1000) return (num / 1000).toFixed(1) + 'K';
    return num.toString();
  };

  const metrics = [
    {
      title: 'Files Processed',
      value: formatNumber(analytics.totalFilesProcessed),
      change: '+12%',
      icon: '📁',
      color: 'blue'
    },
    {
      title: 'Records Analyzed',
      value: formatNumber(analytics.totalRecordsAnalyzed),
      change: '+8%',
      icon: '👥',
      color: 'green'
    },
    {
      title: 'Avg. Processing Time',
      value: analytics.averageProcessingTime > 0 ? `${Math.round(analytics.averageProcessingTime)}ms` : 'Ready',
      change: '-15%',
      icon: '⏱️',
      color: 'yellow'
    },
    {
      title: 'Success Rate',
      value: `${analytics.successRate}%`,
      change: '+2%',
      icon: '✅',
      color: 'green'
    }
  ];

  return (
    <div className="space-y-8">
      <div className="text-center">
        <div className="w-20 h-20 bg-gradient-to-br from-blue-500 to-blue-600 rounded-2xl flex items-center justify-center mx-auto mb-6 shadow-glow">
          📊
        </div>
        
        <h2 className="text-3xl font-bold text-gray-900 mb-4">
          Analytics Dashboard
        </h2>
        <p className="text-lg text-gray-600 max-w-2xl mx-auto">
          Monitor your AI-enhanced survey analysis performance and track key metrics.
        </p>
      </div>

      <div className="max-w-7xl mx-auto">
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          {metrics.map((metric, index) => (
            <div key={metric.title} className="card p-6">
              <div className="flex items-center justify-between mb-4">
                <div className={`w-12 h-12 bg-${metric.color}-100 rounded-xl flex items-center justify-center`}>
                  {metric.icon}
                </div>
                <span className={`text-sm font-medium text-${metric.color}-600`}>
                  {metric.change}
                </span>
              </div>
              <p className="text-2xl font-bold text-gray-900 mb-1">{metric.value}</p>
              <p className="text-sm text-gray-600">{metric.title}</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default Dashboard; 