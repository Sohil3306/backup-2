# AI Enhanced Survey Analysis Application - API Documentation

## Overview

This application provides a comprehensive suite of tools for automated survey data processing, cleaning, weighting, and statistical analysis. It's designed specifically for official statistical agencies and research organizations that need to process large-scale survey datasets efficiently and reproducibly.

## Key Features

### 🔄 Data Processing & Analysis
- **Multi-format Support**: CSV, Excel (.xlsx, .xls)
- **Automated Data Profiling**: Comprehensive data quality assessment
- **Advanced Missing Value Treatment**: Multiple imputation methods (mean, median, KNN, interpolation)
- **Sophisticated Outlier Detection**: IQR, Z-score, Isolation Forest, Local Outlier Factor
- **Custom Validation Rules**: Range checks, categorical validation, regex patterns
- **Survey Weight Application**: Design weights, normalization, trimming

### 📊 Statistical Estimation
- **Weighted & Unweighted Estimates**: Means, standard deviations, confidence intervals
- **Margin of Error Calculations**: 95% confidence intervals by default
- **Design Effect Analysis**: Impact of survey weights on variance
- **Effective Sample Size**: Adjusted sample sizes accounting for weighting

### 📈 Visualization & Reporting
- **Interactive Dashboards**: Data quality, distribution, correlation analysis
- **Professional Reports**: PDF and HTML formats with templates
- **Audit Trails**: Complete processing history and validation logs
- **Export Capabilities**: Clean data export in multiple formats

## API Endpoints

### Core Endpoints

#### 1. Health Check
```
GET /health
```
**Description**: Check API status and current state.

**Response**:
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00",
  "version": "2.0",
  "data_loaded": true,
  "data_cleaned": true,
  "estimates_available": true
}
```

#### 2. File Upload
```
POST /upload
```
**Description**: Upload CSV or Excel files for analysis.

**Request**: Multipart form data with file
- `file`: CSV or Excel file
- `sheet_name` (optional): Excel sheet name

**Response**:
```json
{
  "success": true,
  "message": "File uploaded and profiled successfully",
  "filename": "survey_data.csv",
  "data_profile": {
    "basic_info": {
      "shape": [1000, 8],
      "columns": ["age", "income", "education", ...],
      "dtypes": {...},
      "memory_usage": 64000
    },
    "missing_analysis": {...},
    "data_quality": {...},
    "statistical_summary": {...}
  }
}
```

#### 3. Data Profiling
```
GET /profile
```
**Description**: Get comprehensive data profiling information.

**Response**: Detailed data profile including:
- Basic information (shape, columns, data types)
- Missing value analysis with patterns
- Data quality metrics (completeness, uniqueness, consistency)
- Statistical summaries for numeric variables
- Categorical variable analysis
- Correlation analysis

#### 4. Data Cleaning
```
POST /clean
```
**Description**: Perform advanced data cleaning operations.

**Request Body**:
```json
{
  "imputation": {
    "method": "auto",
    "columns": ["income", "age"],
    "kwargs": {
      "n_neighbors": 5
    }
  },
  "outliers": {
    "detection_methods": ["iqr", "zscore", "isolation_forest"],
    "handling_method": "winsorize",
    "columns": ["satisfaction_score"],
    "kwargs": {
      "lower_percentile": 5,
      "upper_percentile": 95
    }
  }
}
```

**Imputation Methods**:
- `auto`: Automatically choose best method
- `mean`: Mean imputation
- `median`: Median imputation
- `knn`: K-Nearest Neighbors imputation
- `forward_fill`: Forward fill
- `backward_fill`: Backward fill
- `interpolate`: Linear interpolation

**Outlier Detection Methods**:
- `iqr`: Interquartile Range method
- `zscore`: Z-score method (threshold: 3)
- `isolation_forest`: Isolation Forest algorithm
- `lof`: Local Outlier Factor

**Outlier Handling Methods**:
- `winsorize`: Cap values at percentiles
- `remove`: Remove outlier observations
- `transform`: Log or Box-Cox transformation

#### 5. Validation Rules
```
POST /validate
```
**Description**: Add custom validation rules for data quality checks.

**Request Body**:
```json
{
  "rules": [
    {
      "type": "range",
      "column": "age",
      "condition": "between",
      "kwargs": {"min_val": 18, "max_val": 100}
    },
    {
      "type": "categorical",
      "column": "education_level",
      "condition": "in_list",
      "kwargs": {"allowed_values": ["High School", "Bachelor", "Master", "PhD"]}
    },
    {
      "type": "regex",
      "column": "email",
      "condition": "matches",
      "kwargs": {"pattern": "^[\\w\\.-]+@[\\w\\.-]+\\.\\w+$"}
    }
  ]
}
```

#### 6. Statistical Analysis
```
POST /analyze
```
**Description**: Perform comprehensive weighted statistical analysis.

**Request Body**:
```json
{
  "weights": {
    "column": "survey_weight",
    "method": "design",
    "kwargs": {
      "normalize_weights": true,
      "trim_weights": true,
      "trim_percentile": 95
    }
  },
  "estimation": {
    "variables": ["age", "income", "satisfaction_score"],
    "confidence_level": 0.95
  }
}
```

**Weight Methods**:
- `design`: Use provided design weights
- `mean_imputation`: Impute missing weights with mean
- `unit_weight`: Replace missing weights with 1.0

**Response**:
```json
{
  "success": true,
  "estimates": {
    "age": {
      "unweighted": {
        "mean": 35.2,
        "std": 12.1,
        "se": 0.38,
        "moe": 0.75,
        "ci_lower": 34.45,
        "ci_upper": 35.95,
        "n": 1000
      },
      "weighted": {
        "mean": 36.1,
        "std": 11.8,
        "se": 0.42,
        "moe": 0.82,
        "ci_lower": 35.28,
        "ci_upper": 36.92,
        "design_effect": 1.23,
        "effective_n": 813
      }
    }
  },
  "visualizations": {...},
  "audit_trail": {...}
}
```

#### 7. Report Generation
```
POST /report
```
**Description**: Generate comprehensive PDF and HTML reports.

**Request Body**:
```json
{
  "format": "both",
  "title": "Survey Analysis Report",
  "author": "Statistical Agency"
}
```

**Formats**:
- `pdf`: PDF report only
- `html`: HTML report only
- `both`: Both PDF and HTML

#### 8. Data Export
```
POST /export
```
**Description**: Export cleaned data and analysis results.

**Request Body**:
```json
{
  "format": "excel"
}
```

**Export Formats**:
- `csv`: Cleaned data as CSV
- `excel`: Multi-sheet Excel workbook
- `json`: Complete analysis results as JSON

#### 9. Column Information
```
GET /columns
```
**Description**: Get information about available columns.

**Response**:
```json
{
  "success": true,
  "columns": ["age", "income", "education", ...],
  "numeric_columns": ["age", "income", "satisfaction_score"],
  "categorical_columns": ["education", "region", "employment_status"],
  "datetime_columns": []
}
```

## Data Quality Metrics

The application automatically calculates comprehensive data quality metrics:

### Completeness
- Percentage of non-missing values per column
- Missing value patterns and correlations

### Uniqueness
- Percentage of unique values per column
- Duplicate detection and reporting

### Consistency
- Case variation detection for text data
- Format consistency checks

### Validity
- Range validation for numeric data
- Format validation for categorical data
- Custom rule validation

## Statistical Methods

### Survey Weighting
The application supports various survey weighting methodologies:

1. **Design Weights**: Use pre-calculated survey weights
2. **Post-stratification**: Adjust weights based on known population totals
3. **Raking**: Iterative proportional fitting
4. **Weight Trimming**: Cap extreme weights to reduce variance

### Estimation Methods
- **Horvitz-Thompson Estimator**: Unbiased estimator using inclusion probabilities
- **Ratio Estimator**: Improved efficiency using auxiliary information
- **Regression Estimator**: Linear regression-based estimation

### Variance Estimation
- **Taylor Series Linearization**: Standard approach for complex surveys
- **Bootstrap Methods**: Resampling-based variance estimation
- **Jackknife Methods**: Delete-one-group variance estimation

## Usage Examples

### Complete Analysis Workflow

1. **Upload Data**:
```python
import requests

# Upload file
with open('survey_data.csv', 'rb') as f:
    response = requests.post('http://localhost:5000/upload', 
                           files={'file': f})
```

2. **Clean Data**:
```python
cleaning_config = {
    "imputation": {"method": "auto"},
    "outliers": {"detection_methods": ["iqr"], "handling_method": "winsorize"}
}
response = requests.post('http://localhost:5000/clean', json=cleaning_config)
```

3. **Analyze with Weights**:
```python
analysis_config = {
    "weights": {"column": "survey_weight", "method": "design"},
    "estimation": {"variables": ["income", "satisfaction"], "confidence_level": 0.95}
}
response = requests.post('http://localhost:5000/analyze', json=analysis_config)
```

4. **Generate Report**:
```python
report_config = {"format": "pdf", "title": "Monthly Survey Report"}
response = requests.post('http://localhost:5000/report', json=report_config)
```

## Error Handling

The API provides comprehensive error handling with detailed error messages:

```json
{
  "error": "Missing required field: weight_column",
  "details": "Survey weight column 'weights' not found in dataset",
  "suggestions": ["Check column names", "Upload file with weight column"]
}
```

## Performance Considerations

### Large Datasets
- Streaming data processing for files > 100MB
- Chunked processing for memory efficiency
- Progress indicators for long-running operations

### Optimization
- Automatic data type optimization
- Parallel processing for independent operations
- Caching of intermediate results

## Security Features

- File type validation
- Input sanitization
- Size limits on uploads
- Audit logging of all operations

## Compliance & Standards

The application follows international standards for official statistics:

- **UN Fundamental Principles**: Independence, impartiality, reliability
- **ISO 20252**: Market research standards
- **SDMX**: Statistical Data and Metadata eXchange
- **FAIR Principles**: Findable, Accessible, Interoperable, Reusable

## Testing

Run the comprehensive test suite:

```bash
python test_enhanced_functionality.py
```

The test suite validates:
- All API endpoints
- Data processing functionality
- Statistical calculations
- Report generation
- Error handling

## Support & Troubleshooting

### Common Issues

1. **Memory Errors**: Reduce file size or increase system memory
2. **Encoding Issues**: Ensure UTF-8 encoding for CSV files
3. **Missing Dependencies**: Run `pip install -r requirements.txt`

### Logs
Application logs are stored in the `logs/` directory with detailed operation history.

### Performance Monitoring
Built-in performance monitoring tracks:
- Processing times
- Memory usage
- API response times
- Error rates

---

For additional support or feature requests, please refer to the project documentation or contact the development team.
