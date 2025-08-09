import React from 'react';

const ProgressTracker = ({ currentStep, goToStep, dataInfo, analysisResults }) => {
  const steps = [
    {
      id: 1,
      title: 'Data Upload',
      description: 'Upload survey data',
      status: dataInfo ? 'completed' : currentStep >= 1 ? 'active' : 'pending'
    },
    {
      id: 2,
      title: 'Data Cleaning',
      description: 'Configure cleaning methods',
      status: currentStep >= 2 ? 'completed' : currentStep === 2 ? 'active' : 'pending'
    },
    {
      id: 3,
      title: 'Weight Application',
      description: 'Apply survey weights',
      status: currentStep >= 3 ? 'completed' : currentStep === 3 ? 'active' : 'pending'
    },
    {
      id: 4,
      title: 'Analysis & Results',
      description: 'Generate estimates & reports',
      status: analysisResults ? 'completed' : currentStep === 4 ? 'active' : 'pending'
    }
  ];

  return (
    <div className="max-w-4xl mx-auto">
      <div className="card p-6">
        <div className="flex items-center justify-between mb-6">
          <h3 className="text-lg font-bold text-gray-900">Workflow Progress</h3>
          <span className="text-sm text-gray-500">Step {currentStep} of 4</span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          {steps.map((step, index) => (
            <button
              key={step.id}
              onClick={() => goToStep(step.id)}
              className={`relative p-4 rounded-xl border transition-all duration-200 ${
                step.status === 'completed'
                  ? 'text-green-600 bg-green-50 border-green-200 hover:bg-green-100'
                  : currentStep === step.id
                  ? 'text-blue-600 bg-blue-50 border-blue-200 shadow-lg'
                  : 'text-gray-500 bg-gray-50 border-gray-200 hover:bg-gray-100'
              } ${step.status !== 'pending' ? 'hover:shadow-lg cursor-pointer' : 'cursor-not-allowed'}`}
            >
              <div className="flex items-center space-x-3">
                <div className={`w-8 h-8 rounded-lg flex items-center justify-center ${
                  step.status === 'completed' ? 'bg-green-500' :
                  step.status === 'active' ? 'bg-blue-500' : 'bg-gray-300'
                }`}>
                  {step.status === 'completed' ? '✅' : '📋'}
                </div>
                <div className="flex-1 text-left">
                  <p className={`text-sm font-medium ${
                    step.status === 'completed' ? 'text-green-700' :
                    step.status === 'active' ? 'text-blue-700' : 'text-gray-500'
                  }`}>
                    {step.title}
                  </p>
                  <p className="text-xs text-gray-500">{step.description}</p>
                </div>
              </div>
            </button>
          ))}
        </div>

        <div className="mt-6">
          <div className="flex justify-between text-xs text-gray-500 mb-2">
            <span>Progress</span>
            <span>{Math.round((currentStep / 4) * 100)}%</span>
          </div>
          <div className="w-full bg-gray-200 rounded-full h-2">
            <div 
              className="bg-gradient-to-r from-blue-500 to-blue-600 h-2 rounded-full transition-all duration-500"
              style={{ width: `${(currentStep / 4) * 100}%` }}
            />
          </div>
        </div>
      </div>
    </div>
  );
};

export default ProgressTracker; 