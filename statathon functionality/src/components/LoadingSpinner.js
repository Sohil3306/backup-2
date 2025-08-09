import React from 'react';

const LoadingSpinner = () => {
  return (
    <div className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-50">
      <div className="bg-white rounded-2xl p-8 shadow-xl max-w-sm mx-4">
        <div className="text-center">
          <div className="w-16 h-16 bg-gradient-to-br from-blue-500 to-blue-600 rounded-2xl flex items-center justify-center mx-auto mb-4 shadow-glow animate-spin">
            🧠
          </div>
          
          <h3 className="text-lg font-bold text-gray-900 mb-2">
            AI Processing...
          </h3>
          <p className="text-gray-600 mb-4">
            Our intelligent system is analyzing your data
          </p>
          
          <div className="flex items-center justify-center space-x-1">
            <div className="w-2 h-2 bg-blue-500 rounded-full animate-bounce"></div>
            <div className="w-2 h-2 bg-blue-500 rounded-full animate-bounce" style={{animationDelay: '0.2s'}}></div>
            <div className="w-2 h-2 bg-blue-500 rounded-full animate-bounce" style={{animationDelay: '0.4s'}}></div>
          </div>
          
          <div className="flex items-center justify-center space-x-1 mt-4">
            ✨
            <span className="text-xs text-blue-600 font-medium">AI-Enhanced Analysis</span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default LoadingSpinner; 