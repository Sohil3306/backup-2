import * as math from 'mathjs';
import _ from 'lodash';

export class DataProcessor {
  constructor() {
    this.data = null;
    this.cleanedData = null;
    this.weights = null;
  }

  // Load data from CSV or Excel
  loadData(data, fileType = 'csv') {
    this.data = data;
    this.cleanedData = null;
    this.weights = null;
    return this.getDataInfo();
  }

  // Get basic data information
  getDataInfo() {
    if (!this.data || this.data.length === 0) {
      return null;
    }

    const columns = Object.keys(this.data[0]);
    const numericColumns = columns.filter(col => 
      this.data.some(row => !isNaN(parseFloat(row[col])) && row[col] !== null && row[col] !== '')
    );

    return {
      shape: [this.data.length, columns.length],
      columns: columns,
      numericColumns: numericColumns,
      dtypes: this.getDataTypes(),
      sampleData: this.data.slice(0, 5)
    };
  }

  // Get data types for each column
  getDataTypes() {
    if (!this.data || this.data.length === 0) return {};

    const types = {};
    const columns = Object.keys(this.data[0]);

    columns.forEach(col => {
      const values = this.data.map(row => row[col]).filter(val => val !== null && val !== '');
      if (values.length === 0) {
        types[col] = 'string';
        return;
      }

      const isNumeric = values.every(val => !isNaN(parseFloat(val)));
      types[col] = isNumeric ? 'number' : 'string';
    });

    return types;
  }

  // Detect missing values
  detectMissingValues() {
    if (!this.data) return null;

    const columns = Object.keys(this.data[0]);
    const missingPerColumn = {};
    let totalMissing = 0;

    columns.forEach(col => {
      const missing = this.data.filter(row => 
        row[col] === null || row[col] === undefined || row[col] === '' || isNaN(row[col])
      ).length;
      missingPerColumn[col] = missing;
      totalMissing += missing;
    });

    const missingPercentage = {};
    columns.forEach(col => {
      missingPercentage[col] = (missingPerColumn[col] / this.data.length) * 100;
    });

    return {
      totalMissing,
      missingPerColumn,
      missingPercentage
    };
  }

  // Impute missing values
  imputeMissingValues(method = 'mean', columns = null) {
    if (!this.data) return null;

    this.cleanedData = JSON.parse(JSON.stringify(this.data)); // Deep copy

    if (!columns) {
      const info = this.getDataInfo();
      columns = info.numericColumns;
    }

    columns.forEach(col => {
      const values = this.data
        .map(row => parseFloat(row[col]))
        .filter(val => !isNaN(val) && val !== null);

      if (values.length === 0) return;

      let imputeValue;
      switch (method) {
        case 'mean':
          imputeValue = math.mean(values);
          break;
        case 'median':
          imputeValue = math.median(values);
          break;
        case 'knn':
          // Simple KNN implementation
          imputeValue = this.knnImpute(col, values);
          break;
        default:
          imputeValue = math.mean(values);
      }

      this.cleanedData.forEach(row => {
        if (row[col] === null || row[col] === undefined || row[col] === '' || isNaN(row[col])) {
          row[col] = imputeValue;
        }
      });
    });

    return this.cleanedData;
  }

  // Simple KNN imputation
  knnImpute(column, validValues, k = 5) {
    const mean = math.mean(validValues);
    const std = math.std(validValues);
    
    // For simplicity, return mean if no valid neighbors
    if (validValues.length < k) {
      return mean;
    }

    // Find k nearest neighbors based on other numeric columns
    const info = this.getDataInfo();
    const otherNumericCols = info.numericColumns.filter(col => col !== column);
    
    if (otherNumericCols.length === 0) {
      return mean;
    }

    // Calculate distances and find nearest neighbors
    const distances = [];
    this.data.forEach((row, index) => {
      if (row[column] !== null && row[column] !== undefined && row[column] !== '' && !isNaN(row[column])) {
        let distance = 0;
        otherNumericCols.forEach(otherCol => {
          const val1 = parseFloat(row[otherCol]) || 0;
          const val2 = parseFloat(this.data[0][otherCol]) || 0;
          distance += Math.pow(val1 - val2, 2);
        });
        distances.push({ index, distance: Math.sqrt(distance) });
      }
    });

    distances.sort((a, b) => a.distance - b.distance);
    const nearestNeighbors = distances.slice(0, k);
    const neighborValues = nearestNeighbors.map(n => parseFloat(this.data[n.index][column]));
    
    return math.mean(neighborValues);
  }

  // Detect outliers
  detectOutliers(method = 'iqr', columns = null) {
    if (!this.data) return null;

    if (!columns) {
      const info = this.getDataInfo();
      columns = info.numericColumns;
    }

    const outliersInfo = {};

    columns.forEach(col => {
      const values = this.data
        .map(row => parseFloat(row[col]))
        .filter(val => !isNaN(val) && val !== null);

      if (values.length === 0) return;

      let outliers = [];
      
      if (method === 'iqr') {
        const sorted = values.sort((a, b) => a - b);
        const q1 = math.quantileSeq(sorted, 0.25);
        const q3 = math.quantileSeq(sorted, 0.75);
        const iqr = q3 - q1;
        const lowerBound = q1 - 1.5 * iqr;
        const upperBound = q3 + 1.5 * iqr;
        
        outliers = values.filter(val => val < lowerBound || val > upperBound);
      } else if (method === 'zscore') {
        const mean = math.mean(values);
        const std = math.std(values);
        const zScores = values.map(val => Math.abs((val - mean) / std));
        outliers = values.filter((val, index) => zScores[index] > 3);
      }

      outliersInfo[col] = {
        count: outliers.length,
        percentage: (outliers.length / values.length) * 100,
        values: outliers
      };
    });

    return outliersInfo;
  }

  // Handle outliers
  handleOutliers(method = 'winsorize', columns = null) {
    if (!this.cleanedData) {
      this.cleanedData = JSON.parse(JSON.stringify(this.data));
    }

    if (!columns) {
      const info = this.getDataInfo();
      columns = info.numericColumns;
    }

    columns.forEach(col => {
      const values = this.cleanedData
        .map(row => parseFloat(row[col]))
        .filter(val => !isNaN(val) && val !== null);

      if (values.length === 0) return;

      if (method === 'winsorize') {
        const sorted = values.sort((a, b) => a - b);
        const lowerPercentile = math.quantileSeq(sorted, 0.05);
        const upperPercentile = math.quantileSeq(sorted, 0.95);
        
        this.cleanedData.forEach(row => {
          const val = parseFloat(row[col]);
          if (!isNaN(val)) {
            row[col] = Math.max(lowerPercentile, Math.min(upperPercentile, val));
          }
        });
      } else if (method === 'remove') {
        const sorted = values.sort((a, b) => a - b);
        const q1 = math.quantileSeq(sorted, 0.25);
        const q3 = math.quantileSeq(sorted, 0.75);
        const iqr = q3 - q1;
        const lowerBound = q1 - 1.5 * iqr;
        const upperBound = q3 + 1.5 * iqr;
        
        this.cleanedData = this.cleanedData.filter(row => {
          const val = parseFloat(row[col]);
          return isNaN(val) || (val >= lowerBound && val <= upperBound);
        });
      }
    });

    return this.cleanedData;
  }

  // Apply survey weights
  applyWeights(weightColumn = null) {
    if (!this.cleanedData) {
      this.cleanedData = JSON.parse(JSON.stringify(this.data));
    }

    if (weightColumn && this.cleanedData[0][weightColumn] !== undefined) {
      this.weights = this.cleanedData.map(row => parseFloat(row[weightColumn]) || 1);
    } else {
      this.weights = new Array(this.cleanedData.length).fill(1);
    }

    return this.weights;
  }

  // Calculate statistical estimates
  calculateEstimates(columns = null) {
    if (!this.cleanedData) {
      this.cleanedData = JSON.parse(JSON.stringify(this.data));
    }

    if (!columns) {
      const info = this.getDataInfo();
      columns = info.numericColumns;
    }

    const estimates = {};

    columns.forEach(col => {
      const values = this.cleanedData
        .map(row => parseFloat(row[col]))
        .filter(val => !isNaN(val) && val !== null);

      if (values.length === 0) return;

      const weights = this.weights ? 
        this.cleanedData
          .map((row, index) => ({ val: parseFloat(row[col]), weight: this.weights[index] }))
          .filter(item => !isNaN(item.val) && item.val !== null)
          .map(item => item.weight) : 
        new Array(values.length).fill(1);

      // Unweighted estimates
      const unweightedMean = math.mean(values);
      const unweightedStd = math.std(values);
      const unweightedSe = unweightedStd / Math.sqrt(values.length);
      const unweightedMoe = 1.96 * unweightedSe;

      // Weighted estimates
      const weightedValues = this.cleanedData
        .map((row, index) => ({ val: parseFloat(row[col]), weight: this.weights[index] }))
        .filter(item => !isNaN(item.val) && item.val !== null);

      const totalWeight = weightedValues.reduce((sum, item) => sum + item.weight, 0);
      const weightedMean = weightedValues.reduce((sum, item) => sum + (item.val * item.weight), 0) / totalWeight;
      
      const weightedVariance = weightedValues.reduce((sum, item) => 
        sum + (Math.pow(item.val - weightedMean, 2) * item.weight), 0) / totalWeight;
      const weightedStd = Math.sqrt(weightedVariance);
      const weightedSe = weightedStd / Math.sqrt(weightedValues.length);
      const weightedMoe = 1.96 * weightedSe;

      estimates[col] = {
        unweighted: {
          mean: unweightedMean,
          std: unweightedStd,
          se: unweightedSe,
          moe: unweightedMoe
        },
        weighted: {
          mean: weightedMean,
          std: weightedStd,
          se: weightedSe,
          moe: weightedMoe
        }
      };
    });

    return estimates;
  }

  // Get correlation matrix
  getCorrelationMatrix(columns = null) {
    if (!this.cleanedData) {
      this.cleanedData = JSON.parse(JSON.stringify(this.data));
    }

    if (!columns) {
      const info = this.getDataInfo();
      columns = info.numericColumns;
    }

    const correlationMatrix = {};

    columns.forEach(col1 => {
      correlationMatrix[col1] = {};
      columns.forEach(col2 => {
        const values1 = this.cleanedData
          .map(row => parseFloat(row[col1]))
          .filter(val => !isNaN(val) && val !== null);
        const values2 = this.cleanedData
          .map(row => parseFloat(row[col2]))
          .filter(val => !isNaN(val) && val !== null);

        if (values1.length === values2.length && values1.length > 1) {
          const correlation = this.calculateCorrelation(values1, values2);
          correlationMatrix[col1][col2] = correlation;
        } else {
          correlationMatrix[col1][col2] = 0;
        }
      });
    });

    return correlationMatrix;
  }

  // Calculate correlation coefficient
  calculateCorrelation(x, y) {
    const n = x.length;
    const sumX = x.reduce((a, b) => a + b, 0);
    const sumY = y.reduce((a, b) => a + b, 0);
    const sumXY = x.reduce((sum, xi, i) => sum + xi * y[i], 0);
    const sumX2 = x.reduce((sum, xi) => sum + xi * xi, 0);
    const sumY2 = y.reduce((sum, yi) => sum + yi * yi, 0);

    const numerator = n * sumXY - sumX * sumY;
    const denominator = Math.sqrt((n * sumX2 - sumX * sumX) * (n * sumY2 - sumY * sumY));

    return denominator === 0 ? 0 : numerator / denominator;
  }

  // Get summary statistics
  getSummaryStatistics(columns = null) {
    if (!this.cleanedData) {
      this.cleanedData = JSON.parse(JSON.stringify(this.data));
    }

    if (!columns) {
      const info = this.getDataInfo();
      columns = info.numericColumns;
    }

    const summary = {};

    columns.forEach(col => {
      const values = this.cleanedData
        .map(row => parseFloat(row[col]))
        .filter(val => !isNaN(val) && val !== null);

      if (values.length === 0) return;

      const sorted = values.sort((a, b) => a - b);
      
      summary[col] = {
        count: values.length,
        mean: math.mean(values),
        median: math.median(values),
        std: math.std(values),
        min: math.min(values),
        max: math.max(values),
        q1: math.quantileSeq(sorted, 0.25),
        q3: math.quantileSeq(sorted, 0.75),
        iqr: math.quantileSeq(sorted, 0.75) - math.quantileSeq(sorted, 0.25)
      };
    });

    return summary;
  }
} 