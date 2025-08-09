import React from 'react';

const Header = ({ sidebarOpen, setSidebarOpen, currentView, setCurrentView, analytics }) => {
  const formatNumber = (num) => {
    if (num >= 1000000) return (num / 1000000).toFixed(1) + 'M';
    if (num >= 1000) return (num / 1000).toFixed(1) + 'K';
    return num.toString();
  };

  return (
    <header className="bg-white/80 backdrop-blur-md border-b border-gray-200/50 sticky top-0 z-50">
      <div className="px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          <div className="flex items-center space-x-4">
            <button
              onClick={() => setSidebarOpen(!sidebarOpen)}
              className="p-2 rounded-xl hover:bg-gray-100 transition-colors duration-200 lg:hidden"
            >
              {sidebarOpen ? '✕' : '☰'}
            </button>

            <div className="flex items-center space-x-3">
              <div className="relative">
                <div className="w-10 h-10 bg-gradient-to-br from-blue-500 to-blue-600 rounded-xl flex items-center justify-center shadow-glow">
                  🧠
                </div>
                <div className="absolute -top-1 -right-1 w-4 h-4 bg-green-500 rounded-full border-2 border-white animate-pulse"></div>
              </div>
              
              <div className="hidden sm:block">
                <h1 className="text-xl font-bold gradient-text">
                  AI-Enhanced Survey Analysis
                </h1>
                <p className="text-xs text-gray-500 -mt-1">
                  MoSPI Hackathon Solution
                </p>
              </div>
            </div>
          </div>

          <div className="hidden lg:flex items-center space-x-1">
            <button
              onClick={() => setCurrentView('workflow')}
              className={`px-4 py-2 rounded-xl font-medium transition-all duration-200 ${
                currentView === 'workflow'
                  ? 'bg-blue-100 text-blue-700 shadow-lg'
                  : 'text-gray-600 hover:text-blue-600 hover:bg-gray-50'
              }`}
            >
              📊 Workflow
            </button>
            
            <button
              onClick={() => setCurrentView('dashboard')}
              className={`px-4 py-2 rounded-xl font-medium transition-all duration-200 ${
                currentView === 'dashboard'
                  ? 'bg-blue-100 text-blue-700 shadow-lg'
                  : 'text-gray-600 hover:text-blue-600 hover:bg-gray-50'
              }`}
            >
              📈 Dashboard
            </button>
            
            <button
              onClick={() => setCurrentView('results')}
              className={`px-4 py-2 rounded-xl font-medium transition-all duration-200 ${
                currentView === 'results'
                  ? 'bg-blue-100 text-blue-700 shadow-lg'
                  : 'text-gray-600 hover:text-blue-600 hover:bg-gray-50'
              }`}
            >
              📄 Results
            </button>
          </div>

          <div className="flex items-center space-x-3">
            <div className="hidden md:flex items-center space-x-4">
              <div className="flex items-center space-x-2 px-3 py-1.5 bg-green-50 rounded-lg border border-green-200">
                <span className="text-sm font-medium text-green-700">
                  {formatNumber(analytics.totalFilesProcessed)} files
                </span>
              </div>
              
              <div className="flex items-center space-x-2 px-3 py-1.5 bg-blue-50 rounded-lg border border-blue-200">
                <span className="text-sm font-medium text-blue-700">
                  {formatNumber(analytics.totalRecordsAnalyzed)} records
                </span>
              </div>
            </div>

            <button className="relative p-2 rounded-xl hover:bg-gray-100 transition-colors duration-200">
              🔔
              <div className="absolute -top-1 -right-1 w-3 h-3 bg-red-500 rounded-full"></div>
            </button>

            <button className="flex items-center space-x-2 p-2 rounded-xl hover:bg-gray-100 transition-colors duration-200">
              <div className="w-8 h-8 bg-gradient-to-br from-blue-500 to-purple-500 rounded-lg flex items-center justify-center">
                👤
              </div>
              <span className="hidden sm:block text-sm font-medium text-gray-700">
                Analyst
              </span>
            </button>
          </div>
        </div>
      </div>
    </header>
  );
};

export default Header; 