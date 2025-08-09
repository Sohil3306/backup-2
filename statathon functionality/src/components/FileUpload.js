import React, { useState } from 'react';

const FileUpload = ({ onFileUpload, dataInfo, loading }) => {
  const [dragActive, setDragActive] = useState(false);

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      onFileUpload(e.dataTransfer.files[0]);
    }
  };

  const handleFileInput = (e) => {
    if (e.target.files && e.target.files[0]) {
      onFileUpload(e.target.files[0]);
    }
  };

  return (
    <div className="space-y-8">
      <div className="text-center">
        <div className="w-20 h-20 bg-gradient-to-br from-blue-500 to-blue-600 rounded-2xl flex items-center justify-center mx-auto mb-6 shadow-glow">
          📤
        </div>
        
        <h2 className="text-3xl font-bold text-gray-900 mb-4">
          Upload Your Survey Data
        </h2>
        <p className="text-lg text-gray-600 max-w-2xl mx-auto">
          Start your AI-enhanced analysis by uploading your CSV or Excel survey data. 
          Our intelligent system will automatically detect and validate your data structure.
        </p>
      </div>

      <div className="max-w-4xl mx-auto">
        <div
          onDragEnter={handleDrag}
          onDragLeave={handleDrag}
          onDragOver={handleDrag}
          onDrop={handleDrop}
          className={`relative border-2 border-dashed rounded-2xl p-12 text-center transition-all duration-300 cursor-pointer ${
            dragActive
              ? 'border-blue-500 bg-blue-50 shadow-glow'
              : 'border-gray-300 hover:border-blue-400 hover:bg-gray-50'
          }`}
          onClick={() => document.getElementById('fileInput').click()}
        >
          <input 
            id="fileInput"
            type="file" 
            accept=".csv,.xlsx,.xls"
            onChange={handleFileInput}
            className="hidden"
          />
          
          <div className="space-y-4">
            <div className="flex justify-center">
              <div className={`w-16 h-16 rounded-xl flex items-center justify-center ${
                dragActive ? 'bg-blue-100' : 'bg-gray-100'
              }`}>
                {dragActive ? '📤' : '📄'}
              </div>
            </div>
            
            <div>
              <h3 className="text-xl font-semibold text-gray-900 mb-2">
                {dragActive ? 'Drop your file here' : 'Drag & drop your file here'}
              </h3>
              <p className="text-gray-600 mb-4">
                or click to browse your files
              </p>
              <p className="text-sm text-gray-500">
                Supports CSV, Excel (.xlsx, .xls) files up to 50MB
              </p>
            </div>
          </div>
        </div>
      </div>

      {dataInfo && (
        <div className="max-w-4xl mx-auto">
          <div className="card p-8">
            <div className="flex items-center space-x-3 mb-6">
              <div className="w-10 h-10 bg-green-100 rounded-xl flex items-center justify-center">
                ✅
              </div>
              <div>
                <h3 className="text-xl font-bold text-gray-900">Data Successfully Loaded</h3>
                <p className="text-gray-600">Your survey data is ready for analysis</p>
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-6">
              <div className="p-4 bg-blue-50 rounded-xl">
                <div className="flex items-center space-x-2 mb-2">
                  <span className="font-semibold text-blue-700">Dataset Size</span>
                </div>
                <p className="text-2xl font-bold text-blue-800">
                  {dataInfo.shape[0].toLocaleString()} × {dataInfo.shape[1]}
                </p>
                <p className="text-sm text-blue-600">rows × columns</p>
              </div>

              <div className="p-4 bg-green-50 rounded-xl">
                <div className="flex items-center space-x-2 mb-2">
                  <span className="font-semibold text-green-700">Total Columns</span>
                </div>
                <p className="text-2xl font-bold text-green-800">
                  {dataInfo.columns.length}
                </p>
                <p className="text-sm text-green-600">variables</p>
              </div>

              <div className="p-4 bg-yellow-50 rounded-xl">
                <div className="flex items-center space-x-2 mb-2">
                  <span className="font-semibold text-yellow-700">Numeric Columns</span>
                </div>
                <p className="text-2xl font-bold text-yellow-800">
                  {dataInfo.numericColumns.length}
                </p>
                <p className="text-sm text-yellow-600">for analysis</p>
              </div>
            </div>
          </div>
        </div>
      )}

      {loading && (
        <div className="text-center py-12">
          <div className="inline-flex items-center space-x-3">
            <div className="w-6 h-6 border-2 border-blue-500 border-t-transparent rounded-full animate-spin"></div>
            <span className="text-lg font-medium text-gray-700">Processing your data...</span>
          </div>
        </div>
      )}
    </div>
  );
};

export default FileUpload; 