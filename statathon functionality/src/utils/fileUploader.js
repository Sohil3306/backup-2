import Papa from 'papaparse';
import * as XLSX from 'xlsx';

export class FileUploader {
  constructor() {
    this.supportedTypes = {
      'text/csv': 'csv',
      'application/vnd.ms-excel': 'xls',
      'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': 'xlsx'
    };
  }

  // Check if file type is supported
  isSupportedFile(file) {
    return file.type in this.supportedTypes || 
           file.name.endsWith('.csv') || 
           file.name.endsWith('.xlsx') || 
           file.name.endsWith('.xls');
  }

  // Get file type from file
  getFileType(file) {
    if (file.type in this.supportedTypes) {
      return this.supportedTypes[file.type];
    }
    
    if (file.name.endsWith('.csv')) return 'csv';
    if (file.name.endsWith('.xlsx')) return 'xlsx';
    if (file.name.endsWith('.xls')) return 'xls';
    
    return null;
  }

  // Parse CSV file
  parseCSV(file) {
    return new Promise((resolve, reject) => {
      Papa.parse(file, {
        header: true,
        skipEmptyLines: true,
        complete: (results) => {
          if (results.errors.length > 0) {
            reject(new Error(`CSV parsing errors: ${results.errors.map(e => e.message).join(', ')}`));
          } else {
            resolve(results.data);
          }
        },
        error: (error) => {
          reject(new Error(`CSV parsing failed: ${error.message}`));
        }
      });
    });
  }

  // Parse Excel file
  parseExcel(file) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      
      reader.onload = (e) => {
        try {
          const data = new Uint8Array(e.target.result);
          const workbook = XLSX.read(data, { type: 'array' });
          
          // Get the first sheet
          const sheetName = workbook.SheetNames[0];
          const worksheet = workbook.Sheets[sheetName];
          
          // Convert to JSON
          const jsonData = XLSX.utils.sheet_to_json(worksheet, { header: 1 });
          
          if (jsonData.length === 0) {
            reject(new Error('Excel file is empty'));
            return;
          }
          
          // Convert to array of objects
          const headers = jsonData[0];
          const rows = jsonData.slice(1);
          
          const result = rows.map(row => {
            const obj = {};
            headers.forEach((header, index) => {
              obj[header] = row[index] !== undefined ? row[index] : null;
            });
            return obj;
          });
          
          resolve(result);
        } catch (error) {
          reject(new Error(`Excel parsing failed: ${error.message}`));
        }
      };
      
      reader.onerror = () => {
        reject(new Error('Failed to read Excel file'));
      };
      
      reader.readAsArrayBuffer(file);
    });
  }

  // Upload and parse file
  async uploadFile(file) {
    if (!this.isSupportedFile(file)) {
      throw new Error('Unsupported file type. Please upload a CSV or Excel file.');
    }

    const fileType = this.getFileType(file);
    
    try {
      let data;
      
      if (fileType === 'csv') {
        data = await this.parseCSV(file);
      } else if (fileType === 'xlsx' || fileType === 'xls') {
        data = await this.parseExcel(file);
      } else {
        throw new Error('Unsupported file type');
      }

      // Clean the data
      const cleanedData = this.cleanData(data);
      
      return {
        success: true,
        data: cleanedData,
        fileType: fileType,
        fileName: file.name,
        fileSize: file.size
      };
    } catch (error) {
      throw new Error(`File upload failed: ${error.message}`);
    }
  }

  // Clean the parsed data
  cleanData(data) {
    if (!Array.isArray(data) || data.length === 0) {
      return [];
    }

    return data.map(row => {
      const cleanedRow = {};
      Object.keys(row).forEach(key => {
        let value = row[key];
        
        // Handle empty values
        if (value === '' || value === undefined || value === null) {
          cleanedRow[key] = null;
        } else {
          // Try to convert to number if possible
          const numValue = parseFloat(value);
          if (!isNaN(numValue) && value.toString().trim() !== '') {
            cleanedRow[key] = numValue;
          } else {
            cleanedRow[key] = value.toString().trim();
          }
        }
      });
      return cleanedRow;
    }).filter(row => {
      // Remove completely empty rows
      return Object.values(row).some(value => value !== null && value !== '');
    });
  }

  // Validate data structure
  validateData(data) {
    if (!Array.isArray(data) || data.length === 0) {
      return { valid: false, error: 'Data is empty or invalid' };
    }

    const firstRow = data[0];
    if (!firstRow || typeof firstRow !== 'object') {
      return { valid: false, error: 'Invalid data structure' };
    }

    const columns = Object.keys(firstRow);
    if (columns.length === 0) {
      return { valid: false, error: 'No columns found in data' };
    }

    // Check if all rows have the same columns
    for (let i = 1; i < data.length; i++) {
      const row = data[i];
      if (!row || typeof row !== 'object') {
        return { valid: false, error: `Invalid row at index ${i}` };
      }
      
      const rowColumns = Object.keys(row);
      if (rowColumns.length !== columns.length) {
        return { valid: false, error: `Row ${i} has different number of columns` };
      }
    }

    return { valid: true, columns };
  }

  // Get sample data for preview
  getSampleData(data, maxRows = 5) {
    if (!Array.isArray(data) || data.length === 0) {
      return [];
    }

    return data.slice(0, Math.min(maxRows, data.length));
  }

  // Export data to CSV
  exportToCSV(data, filename = 'export.csv') {
    if (!Array.isArray(data) || data.length === 0) {
      throw new Error('No data to export');
    }

    const csv = Papa.unparse(data);
    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
    
    const link = document.createElement('a');
    if (link.download !== undefined) {
      const url = URL.createObjectURL(blob);
      link.setAttribute('href', url);
      link.setAttribute('download', filename);
      link.style.visibility = 'hidden';
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    }
  }

  // Export data to Excel
  exportToExcel(data, filename = 'export.xlsx') {
    if (!Array.isArray(data) || data.length === 0) {
      throw new Error('No data to export');
    }

    const worksheet = XLSX.utils.json_to_sheet(data);
    const workbook = XLSX.utils.book_new();
    XLSX.utils.book_append_sheet(workbook, worksheet, 'Sheet1');
    
    XLSX.writeFile(workbook, filename);
  }
} 