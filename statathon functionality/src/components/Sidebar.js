import React from 'react';

const Sidebar = ({ sidebarOpen, setSidebarOpen, currentStep, goToStep, currentView, setCurrentView, analytics }) => {
  const steps = [
    {
      id: 1,
      title: 'Data Upload',
      description: 'Upload your survey data',
      status: currentStep >= 1 ? 'completed' : 'pending'
    },
    {
      id: 2,
      title: 'Data Cleaning',
      description: 'Configure cleaning methods',
      status: currentStep >= 2 ? 'completed' : 'pending'
    },
    {
      id: 3,
      title: 'Weight Application',
      description: 'Apply survey weights',
      status: currentStep >= 3 ? 'completed' : 'pending'
    },
    {
      id: 4,
      title: 'Analysis & Results',
      description: 'Generate estimates & reports',
      status: currentStep >= 4 ? 'completed' : 'pending'
    }
  ];

  const formatNumber = (num) => {
    if (num >= 1000000) return (num / 1000000).toFixed(1) + 'M';
    if (num >= 1000) return (num / 1000).toFixed(1) + 'K';
    return num.toString();
  };

  if (!sidebarOpen) return null;

  return (
    <div className="fixed inset-y-0 left-0 z-40 w-80 bg-white border-r border-gray-200 lg:static lg:translate-x-0">
      <div className="flex flex-col h-full">
        <div className="p-6 border-b border-gray-200">
          <div className="flex items-center space-x-3">
            <div className="w-10 h-10 bg-gradient-to-br from-blue-500 to-blue-600 rounded-xl flex items-center justify-center">
              🧠
            </div>
            <div>
              <h2 className="text-lg font-bold text-gray-900">Workflow</h2>
              <p className="text-sm text-gray-500">Step {currentStep} of 4</p>
            </div>
          </div>
        </div>

        <div className="flex-1 overflow-y-auto p-6 space-y-8">
          <div>
            <h3 className="text-sm font-semibold text-gray-900 mb-4 flex items-center">
              🎯 Progress
            </h3>
            <div className="space-y-3">
              {steps.map((step, index) => (
                <button
                  key={step.id}
                  onClick={() => goToStep(step.id)}
                  className={`w-full flex items-center space-x-3 p-3 rounded-xl transition-all duration-200 ${
                    step.status === 'completed'
                      ? 'bg-green-50 border border-green-200 hover:bg-green-100'
                      : currentStep === step.id
                      ? 'bg-blue-50 border border-blue-200 shadow-lg'
                      : 'bg-gray-50 border border-gray-200 hover:bg-gray-100'
                  }`}
                >
                  <div className={`w-8 h-8 rounded-lg flex items-center justify-center ${
                    step.status === 'completed'
                      ? 'bg-green-500'
                      : currentStep === step.id
                      ? 'bg-blue-500'
                      : 'bg-gray-300'
                  }`}>
                    {step.status === 'completed' ? '✅' : '📋'}
                  </div>
                  <div className="flex-1 text-left">
                    <p className={`text-sm font-medium ${
                      step.status === 'completed'
                        ? 'text-green-700'
                        : currentStep === step.id
                        ? 'text-blue-700'
                        : 'text-gray-700'
                    }`}>
                      {step.title}
                    </p>
                    <p className="text-xs text-gray-500">{step.description}</p>
                  </div>
                </button>
              ))}
            </div>
          </div>

          <div>
            <h3 className="text-sm font-semibold text-gray-900 mb-4 flex items-center">
              📊 Analytics
            </h3>
            <div className="space-y-3">
              <div className="p-3 bg-green-50 rounded-xl border border-green-200">
                <div className="flex items-center justify-between">
                  <span className="text-sm font-medium text-green-700">Files Processed</span>
                  <span className="text-lg font-bold text-green-800">
                    {formatNumber(analytics.totalFilesProcessed)}
                  </span>
                </div>
              </div>
              
              <div className="p-3 bg-blue-50 rounded-xl border border-blue-200">
                <div className="flex items-center justify-between">
                  <span className="text-sm font-medium text-blue-700">Records Analyzed</span>
                  <span className="text-lg font-bold text-blue-800">
                    {formatNumber(analytics.totalRecordsAnalyzed)}
                  </span>
                </div>
              </div>
            </div>
          </div>
        </div>

        <div className="p-6 border-t border-gray-200">
          <div className="text-center">
            <p className="text-xs text-gray-500 mb-2">MoSPI Hackathon 2024</p>
            <div className="flex items-center justify-center space-x-1">
              <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
              <span className="text-xs text-green-600 font-medium">AI-Enhanced</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Sidebar; 