import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import Papa from 'papaparse';
import * as XLSX from 'xlsx';
import { jsPDF } from 'jspdf';
import 'jspdf-autotable';
import * as math from 'mathjs';
import _ from 'lodash';
import toast, { Toaster } from 'react-hot-toast';

// Components
import Header from './components/Header';
import Sidebar from './components/Sidebar';
import FileUpload from './components/FileUpload';
import DataCleaning from './components/DataCleaning';
import WeightApplication from './components/WeightApplication';
import AnalysisResults from './components/AnalysisResults';
import Dashboard from './components/Dashboard';
import LoadingSpinner from './components/LoadingSpinner';
import ProgressTracker from './components/ProgressTracker';

function App() {
  // State management
  const [currentView, setCurrentView] = useState('workflow');
  const [currentStep, setCurrentStep] = useState(1);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [loading, setLoading] = useState(false);
  const [dataInfo, setDataInfo] = useState(null);
  const [analysisResults, setAnalysisResults] = useState(null);
  
  // Configuration state
  const [config, setConfig] = useState({
    imputationMethod: '',
    imputationColumns: [],
    outlierMethod: '',
    outlierHandling: 'winsorize',
    outlierColumns: [],
    weightColumn: '',
    estimateColumns: []
  });

  // Analytics state
  const [analytics, setAnalytics] = useState({
    totalFilesProcessed: 0,
    totalRecordsAnalyzed: 0,
    averageProcessingTime: 0,
    successRate: 98
  });

  // Update config helper
  const updateConfig = (key, value) => {
    setConfig(prev => ({ ...prev, [key]: value }));
  };

  // Navigation helpers
  const goToStep = (step) => {
    if (step <= currentStep || (step === 1 && dataInfo) || 
        (step === 2 && dataInfo) || (step === 3 && dataInfo) || 
        (step === 4 && dataInfo)) {
      setCurrentStep(step);
    }
  };

  const nextStep = () => {
    if (currentStep < 4) {
      setCurrentStep(currentStep + 1);
    }
  };

  const prevStep = () => {
    if (currentStep > 1) {
      setCurrentStep(currentStep - 1);
    }
  };

  // File upload handler
  const handleFileUpload = async (file) => {
    setLoading(true);
    
    try {
      const data = await parseFile(file);
      const info = analyzeData(data);
      
      setDataInfo(info);
      setAnalytics(prev => ({
        ...prev,
        totalFilesProcessed: prev.totalFilesProcessed + 1,
        totalRecordsAnalyzed: prev.totalRecordsAnalyzed + info.shape[0]
      }));
      
      toast.success('File uploaded successfully!');
      nextStep();
    } catch (error) {
      toast.error('Error uploading file: ' + error.message);
    } finally {
      setLoading(false);
    }
  };

  // File parsing
  const parseFile = (file) => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      
      reader.onload = (e) => {
        try {
          const content = e.target.result;
          
          if (file.name.endsWith('.csv')) {
            Papa.parse(content, {
              header: true,
              complete: (results) => {
                if (results.errors.length > 0) {
                  reject(new Error('CSV parsing errors'));
                } else {
                  resolve(results.data);
                }
              },
              error: (error) => reject(error)
            });
          } else if (file.name.match(/\.xlsx?$/)) {
            const workbook = XLSX.read(content, { type: 'binary' });
            const sheetName = workbook.SheetNames[0];
            const worksheet = workbook.Sheets[sheetName];
            const data = XLSX.utils.sheet_to_json(worksheet);
            resolve(data);
          } else {
            reject(new Error('Unsupported file format'));
          }
        } catch (error) {
          reject(error);
        }
      };
      
      reader.onerror = () => reject(new Error('File reading failed'));
      
      if (file.name.endsWith('.csv')) {
        reader.readAsText(file);
      } else {
        reader.readAsBinaryString(file);
      }
    });
  };

  // Data analysis
  const analyzeData = (data) => {
    if (!data || data.length === 0) {
      throw new Error('No data found in file');
    }

    const columns = Object.keys(data[0]);
    const numericColumns = columns.filter(col => {
      const sampleValues = data.slice(0, 100).map(row => row[col]).filter(val => val !== null && val !== undefined && val !== '');
      return sampleValues.length > 0 && sampleValues.every(val => !isNaN(Number(val)));
    });

    return {
      shape: [data.length, columns.length],
      columns,
      numericColumns,
      data
    };
  };

  // Run analysis
  const runAnalysis = async () => {
    setLoading(true);
    
    try {
      // Simulate processing time
      await new Promise(resolve => setTimeout(resolve, 2000));
      
      const results = performAnalysis(dataInfo.data, config);
      setAnalysisResults(results);
      
      setAnalytics(prev => ({
        ...prev,
        averageProcessingTime: (prev.averageProcessingTime + results.processingTime) / 2
      }));
      
      toast.success('Analysis completed successfully!');
    } catch (error) {
      toast.error('Analysis failed: ' + error.message);
    } finally {
      setLoading(false);
    }
  };

  // Perform statistical analysis
  const performAnalysis = (data, config) => {
    const startTime = Date.now();
    
    // Clean data based on config
    let cleanedData = [...data];
    
    // Apply imputation
    if (config.imputationMethod && config.imputationColumns.length > 0) {
      cleanedData = applyImputation(cleanedData, config);
    }
    
    // Apply outlier handling
    if (config.outlierMethod && config.outlierColumns.length > 0) {
      cleanedData = handleOutliers(cleanedData, config);
    }
    
    // Calculate estimates
    const estimates = {};
    const columnsToEstimate = config.estimateColumns.length > 0 
      ? config.estimateColumns 
      : dataInfo.numericColumns;
    
    columnsToEstimate.forEach(col => {
      const values = cleanedData.map(row => Number(row[col])).filter(val => !isNaN(val));
      if (values.length > 0) {
        estimates[col] = calculateEstimates(values, config.weightColumn ? cleanedData.map(row => Number(row[config.weightColumn])) : null);
      }
    });
    
    return {
      estimates,
      cleanedShape: [cleanedData.length, Object.keys(cleanedData[0]).length],
      missingInfo: {
        totalMissing: data.length * dataInfo.columns.length - cleanedData.length * Object.keys(cleanedData[0]).length
      },
      processingTime: Date.now() - startTime,
      timestamp: new Date().toISOString()
    };
  };

  // Apply imputation
  const applyImputation = (data, config) => {
    const result = [...data];
    
    config.imputationColumns.forEach(col => {
      const values = data.map(row => Number(row[col])).filter(val => !isNaN(val));
      
      if (values.length === 0) return;
      
      let imputedValue;
      switch (config.imputationMethod) {
        case 'mean':
          imputedValue = math.mean(values);
          break;
        case 'median':
          imputedValue = math.median(values);
          break;
        case 'knn':
          imputedValue = math.mean(values); // Simplified KNN
          break;
        default:
          imputedValue = math.mean(values);
      }
      
      result.forEach(row => {
        if (row[col] === null || row[col] === undefined || row[col] === '' || isNaN(Number(row[col]))) {
          row[col] = imputedValue;
        }
      });
    });
    
    return result;
  };

  // Handle outliers
  const handleOutliers = (data, config) => {
    const result = [...data];
    
    config.outlierColumns.forEach(col => {
      const values = data.map(row => Number(row[col])).filter(val => !isNaN(val));
      
      if (values.length === 0) return;
      
      let outliers = [];
      
      if (config.outlierMethod === 'iqr') {
        const q1 = math.quantileSeq(values, 0.25);
        const q3 = math.quantileSeq(values, 0.75);
        const iqr = q3 - q1;
        const lowerBound = q1 - 1.5 * iqr;
        const upperBound = q3 + 1.5 * iqr;
        
        outliers = values.filter(val => val < lowerBound || val > upperBound);
      } else if (config.outlierMethod === 'zscore') {
        const mean = math.mean(values);
        const std = math.std(values);
        outliers = values.filter(val => Math.abs((val - mean) / std) > 3);
      }
      
      if (config.outlierHandling === 'winsorize') {
        const sortedValues = [...values].sort((a, b) => a - b);
        const lowerPercentile = math.quantileSeq(sortedValues, 0.05);
        const upperPercentile = math.quantileSeq(sortedValues, 0.95);
        
        result.forEach(row => {
          const val = Number(row[col]);
          if (!isNaN(val)) {
            if (val < lowerPercentile) row[col] = lowerPercentile;
            if (val > upperPercentile) row[col] = upperPercentile;
          }
        });
      } else if (config.outlierHandling === 'remove') {
        // Remove rows with outliers
        const outlierIndices = [];
        result.forEach((row, index) => {
          const val = Number(row[col]);
          if (!isNaN(val) && outliers.includes(val)) {
            outlierIndices.push(index);
          }
        });
        
        // Remove in reverse order to maintain indices
        outlierIndices.reverse().forEach(index => {
          result.splice(index, 1);
        });
      }
    });
    
    return result;
  };

  // Calculate statistical estimates
  const calculateEstimates = (values, weights = null) => {
    const n = values.length;
    
    // Unweighted estimates
    const unweighted = {
      mean: math.mean(values),
      std: math.std(values),
      se: math.std(values) / Math.sqrt(n)
    };
    
    // Weighted estimates
    let weighted = { ...unweighted };
    
    if (weights && weights.length === n) {
      const validWeights = weights.filter(w => !isNaN(w) && w > 0);
      const validValues = values.filter((_, i) => !isNaN(weights[i]) && weights[i] > 0);
      
      if (validWeights.length > 0) {
        const totalWeight = math.sum(validWeights);
        const weightedSum = validValues.reduce((sum, val, i) => sum + val * validWeights[i], 0);
        const weightedMean = weightedSum / totalWeight;
        
        // Simplified weighted standard error
        const weightedVariance = validValues.reduce((sum, val, i) => {
          return sum + validWeights[i] * Math.pow(val - weightedMean, 2);
        }, 0) / totalWeight;
        
        weighted = {
          mean: weightedMean,
          std: Math.sqrt(weightedVariance),
          se: Math.sqrt(weightedVariance / validWeights.length),
          moe: Math.sqrt(weightedVariance / validWeights.length) * 1.96 // 95% confidence interval
        };
      }
    }
    
    return { unweighted, weighted };
  };

  // Generate PDF report
  const generateReport = () => {
    const doc = new jsPDF();
    
    // Title
    doc.setFontSize(20);
    doc.text('AI-Enhanced Survey Analysis Report', 20, 20);
    
    // Subtitle
    doc.setFontSize(12);
    doc.text('MoSPI Hackathon Solution', 20, 30);
    doc.text(`Generated on: ${new Date().toLocaleDateString()}`, 20, 40);
    
    // Data summary
    doc.setFontSize(14);
    doc.text('Data Summary', 20, 60);
    doc.setFontSize(10);
    doc.text(`Records: ${dataInfo.shape[0]}`, 20, 70);
    doc.text(`Variables: ${dataInfo.shape[1]}`, 20, 80);
    doc.text(`Numeric Variables: ${dataInfo.numericColumns.length}`, 20, 90);
    
    // Analysis results
    if (analysisResults && analysisResults.estimates) {
      doc.setFontSize(14);
      doc.text('Statistical Estimates', 20, 110);
      
      const tableData = Object.entries(analysisResults.estimates).map(([variable, est]) => [
        variable,
        est.weighted.mean.toFixed(2),
        `±${est.weighted.moe.toFixed(2)}`,
        est.weighted.se.toFixed(4)
      ]);
      
      doc.autoTable({
        startY: 120,
        head: [['Variable', 'Weighted Mean', 'Margin of Error', 'Standard Error']],
        body: tableData,
        theme: 'grid'
      });
    }
    
    // Save the PDF
    doc.save('ai-enhanced-survey-analysis-report.pdf');
    toast.success('PDF report generated successfully!');
  };

  // Render current step content
  const renderStepContent = () => {
    switch (currentStep) {
      case 1:
        return <FileUpload onFileUpload={handleFileUpload} dataInfo={dataInfo} loading={loading} />;
      case 2:
        return <DataCleaning config={config} updateConfig={updateConfig} dataInfo={dataInfo} onNext={nextStep} onPrev={prevStep} />;
      case 3:
        return <WeightApplication config={config} updateConfig={updateConfig} dataInfo={dataInfo} onNext={nextStep} onPrev={prevStep} />;
      case 4:
        return <AnalysisResults 
          analysisResults={analysisResults} 
          dataInfo={dataInfo} 
          config={config} 
          onRunAnalysis={runAnalysis} 
          onGenerateReport={generateReport} 
          loading={loading} 
          onPrev={prevStep} 
        />;
      default:
        return <FileUpload onFileUpload={handleFileUpload} dataInfo={dataInfo} loading={loading} />;
    }
  };

  // Render current view
  const renderView = () => {
    switch (currentView) {
      case 'workflow':
        return (
          <div className="space-y-8">
            <ProgressTracker 
              currentStep={currentStep} 
              goToStep={goToStep} 
              dataInfo={dataInfo} 
              analysisResults={analysisResults} 
            />
            {renderStepContent()}
          </div>
        );
      case 'dashboard':
        return <Dashboard analytics={analytics} dataInfo={dataInfo} analysisResults={analysisResults} config={config} />;
      case 'results':
        return <AnalysisResults 
          analysisResults={analysisResults} 
          dataInfo={dataInfo} 
          config={config} 
          onRunAnalysis={runAnalysis} 
          onGenerateReport={generateReport} 
          loading={loading} 
          detailed={true} 
        />;
      default:
        return renderStepContent();
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-blue-50">
      <Toaster position="top-right" />
      
      <Header 
        sidebarOpen={sidebarOpen} 
        setSidebarOpen={setSidebarOpen} 
        currentView={currentView} 
        setCurrentView={setCurrentView} 
        analytics={analytics} 
      />
      
      <div className="flex">
        <Sidebar 
          sidebarOpen={sidebarOpen} 
          setSidebarOpen={setSidebarOpen} 
          currentStep={currentStep} 
          goToStep={goToStep} 
          currentView={currentView} 
          setCurrentView={setCurrentView} 
          analytics={analytics} 
        />
        
        <main className="flex-1 lg:ml-0">
          <div className="p-6 lg:p-8">
            <AnimatePresence mode="wait">
              <motion.div
                key={currentView + currentStep}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
                transition={{ duration: 0.3 }}
              >
                {renderView()}
              </motion.div>
            </AnimatePresence>
          </div>
        </main>
      </div>
      
      {loading && <LoadingSpinner />}
    </div>
  );
}

export default App; 