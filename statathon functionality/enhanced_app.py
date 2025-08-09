import os
import json
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify, send_file, session
from flask_cors import CORS
from werkzeug.utils import secure_filename
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import base64
from datetime import datetime
import zipfile
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
import tempfile
import statsmodels.api as sm
from statsmodels.stats.weightstats import DescrStatsW
import missingno as msno
import xlsxwriter
from reportlab.graphics.shapes import Drawing
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics.charts.linecharts import HorizontalLineChart
import logging
import traceback
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend
app.secret_key = 'your-secret-key-here-change-in-production'

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size

# Create necessary folders
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs('reports', exist_ok=True)
os.makedirs('temp', exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

class AdvancedDataProcessor:
    def __init__(self):
        self.data = None
        self.original_data = None
        self.cleaned_data = None
        self.weights = None
        self.config = {}
        self.processing_log = []
        self.validation_rules = []
        self.estimates = {}
        self.diagnostics = {}
        
    def log_operation(self, operation, details):
        """Log processing operations for audit trail"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'operation': operation,
            'details': details
        }
        self.processing_log.append(log_entry)
        logger.info(f"{operation}: {details}")
        
    def load_data(self, file_path, sheet_name=None):
        """Enhanced data loading with better error handling and schema detection"""
        try:
            if file_path.endswith('.csv'):
                # Try different encodings
                encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
                for encoding in encodings:
                    try:
                        self.data = pd.read_csv(file_path, encoding=encoding)
                        self.log_operation('data_load', f'CSV loaded with {encoding} encoding')
                        break
                    except UnicodeDecodeError:
                        continue
                else:
                    raise ValueError("Could not decode CSV file with any standard encoding")
            else:
                if sheet_name:
                    self.data = pd.read_excel(file_path, sheet_name=sheet_name)
                else:
                    self.data = pd.read_excel(file_path)
                self.log_operation('data_load', f'Excel file loaded, sheet: {sheet_name or "default"}')
            
            # Store original data
            self.original_data = self.data.copy()
            
            # Auto-detect data types and suggest improvements
            self.data = self._optimize_dtypes(self.data)
            
            self.log_operation('data_load', f'Data loaded successfully: {self.data.shape}')
            return self.data
            
        except Exception as e:
            self.log_operation('data_load_error', str(e))
            raise e
    
    def _optimize_dtypes(self, df):
        """Optimize data types for better performance"""
        optimized_df = df.copy()
        
        for col in optimized_df.columns:
            col_type = optimized_df[col].dtype
            
            if col_type == 'object':
                # Try to convert to numeric
                numeric_series = pd.to_numeric(optimized_df[col], errors='coerce')
                if not numeric_series.isnull().all():
                    optimized_df[col] = numeric_series
                else:
                    # Check if it's a date
                    try:
                        date_series = pd.to_datetime(optimized_df[col], errors='coerce')
                        if not date_series.isnull().all():
                            optimized_df[col] = date_series
                    except:
                        pass
            
            elif col_type in ['int64', 'float64']:
                # Downcast numeric types if possible
                if col_type == 'int64':
                    optimized_df[col] = pd.to_numeric(optimized_df[col], downcast='integer')
                else:
                    optimized_df[col] = pd.to_numeric(optimized_df[col], downcast='float')
        
        return optimized_df
    
    def get_data_profile(self):
        """Generate comprehensive data profiling information"""
        if self.data is None:
            return None
        
        profile = {
            'basic_info': {
                'shape': self.data.shape,
                'columns': self.data.columns.tolist(),
                'dtypes': self.data.dtypes.astype(str).to_dict(),
                'memory_usage': self.data.memory_usage(deep=True).sum(),
            },
            'missing_analysis': self._analyze_missing_data(),
            'data_quality': self._assess_data_quality(),
            'statistical_summary': self._get_statistical_summary(),
            'categorical_analysis': self._analyze_categorical_data(),
            'correlation_analysis': self._analyze_correlations()
        }
        
        return profile
    
    def _analyze_missing_data(self):
        """Comprehensive missing data analysis"""
        missing_info = {
            'total_missing': self.data.isnull().sum().sum(),
            'missing_per_column': self.data.isnull().sum().to_dict(),
            'missing_percentage': (self.data.isnull().sum() / len(self.data) * 100).to_dict(),
            'missing_patterns': self._get_missing_patterns()
        }
        
        # Generate missing data visualization
        plt.figure(figsize=(12, 8))
        msno.matrix(self.data)
        plt.title('Missing Data Pattern')
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
        plt.close()
        img_buffer.seek(0)
        missing_info['visualization'] = base64.b64encode(img_buffer.getvalue()).decode()
        
        return missing_info
    
    def _get_missing_patterns(self):
        """Identify missing data patterns"""
        missing_patterns = []
        missing_df = self.data.isnull()
        
        # Find columns that are always missing together
        for col1 in missing_df.columns:
            for col2 in missing_df.columns:
                if col1 != col2:
                    correlation = missing_df[col1].corr(missing_df[col2])
                    if correlation > 0.8:  # High correlation in missingness
                        missing_patterns.append({
                            'columns': [col1, col2],
                            'correlation': correlation,
                            'pattern_type': 'correlated_missing'
                        })
        
        return missing_patterns
    
    def _assess_data_quality(self):
        """Assess overall data quality"""
        quality_metrics = {}
        
        for col in self.data.columns:
            col_data = self.data[col]
            
            quality_metrics[col] = {
                'completeness': (1 - col_data.isnull().sum() / len(col_data)) * 100,
                'uniqueness': len(col_data.unique()) / len(col_data) * 100 if len(col_data) > 0 else 0,
                'consistency': self._check_consistency(col_data),
                'validity': self._check_validity(col_data)
            }
        
        return quality_metrics
    
    def _check_consistency(self, series):
        """Check data consistency within a column"""
        if series.dtype == 'object':
            # Check for case inconsistencies
            if series.dropna().empty:
                return 100
            
            unique_values = series.dropna().unique()
            case_variations = defaultdict(list)
            
            for val in unique_values:
                if isinstance(val, str):
                    normalized = val.lower().strip()
                    case_variations[normalized].append(val)
            
            inconsistent_count = sum(len(variations) - 1 for variations in case_variations.values())
            consistency_score = (1 - inconsistent_count / len(unique_values)) * 100
            return max(0, consistency_score)
        
        return 100  # Numeric data is considered consistent by default
    
    def _check_validity(self, series):
        """Check data validity based on data type expectations"""
        if series.dtype in ['int64', 'float64']:
            # Check for infinite values
            invalid_count = np.isinf(series).sum()
            validity_score = (1 - invalid_count / len(series)) * 100
            return validity_score
        
        elif series.dtype == 'object':
            # Check for empty strings or whitespace-only strings
            if series.dropna().empty:
                return 100
            
            invalid_count = series.dropna().apply(lambda x: isinstance(x, str) and x.strip() == '').sum()
            validity_score = (1 - invalid_count / series.dropna().count()) * 100
            return validity_score
        
        return 100
    
    def _get_statistical_summary(self):
        """Enhanced statistical summary"""
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            return {}
        
        summary = {}
        for col in numeric_cols:
            data = self.data[col].dropna()
            if len(data) > 0:
                summary[col] = {
                    'count': len(data),
                    'mean': data.mean(),
                    'median': data.median(),
                    'std': data.std(),
                    'min': data.min(),
                    'max': data.max(),
                    'q25': data.quantile(0.25),
                    'q75': data.quantile(0.75),
                    'skewness': stats.skew(data),
                    'kurtosis': stats.kurtosis(data),
                    'cv': data.std() / data.mean() if data.mean() != 0 else 0  # Coefficient of variation
                }
        
        return summary
    
    def _analyze_categorical_data(self):
        """Analyze categorical variables"""
        categorical_cols = self.data.select_dtypes(include=['object']).columns
        
        analysis = {}
        for col in categorical_cols:
            data = self.data[col].dropna()
            if len(data) > 0:
                value_counts = data.value_counts()
                analysis[col] = {
                    'unique_count': len(value_counts),
                    'most_frequent': value_counts.index[0] if len(value_counts) > 0 else None,
                    'most_frequent_count': value_counts.iloc[0] if len(value_counts) > 0 else 0,
                    'frequency_distribution': value_counts.head(10).to_dict(),
                    'cardinality_ratio': len(value_counts) / len(data)
                }
        
        return analysis
    
    def _analyze_correlations(self):
        """Analyze correlations between numeric variables"""
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) < 2:
            return {}
        
        corr_matrix = self.data[numeric_cols].corr()
        
        # Find strong correlations
        strong_correlations = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_value = corr_matrix.iloc[i, j]
                if abs(corr_value) > 0.7:  # Strong correlation threshold
                    strong_correlations.append({
                        'var1': corr_matrix.columns[i],
                        'var2': corr_matrix.columns[j],
                        'correlation': corr_value
                    })
        
        return {
            'correlation_matrix': corr_matrix.to_dict(),
            'strong_correlations': strong_correlations
        }
    
    def advanced_missing_imputation(self, method='auto', columns=None, **kwargs):
        """Advanced missing value imputation with multiple methods"""
        if columns is None:
            columns = self.data.select_dtypes(include=[np.number]).columns.tolist()
        
        if self.cleaned_data is None:
            self.cleaned_data = self.data.copy()
        
        imputation_report = {}
        
        for col in columns:
            if col not in self.cleaned_data.columns or not self.cleaned_data[col].isnull().any():
                continue
            
            original_missing = self.cleaned_data[col].isnull().sum()
            
            if method == 'auto':
                # Choose best method based on data characteristics
                method_to_use = self._choose_best_imputation_method(col)
            else:
                method_to_use = method
            
            if method_to_use == 'mean':
                self.cleaned_data[col].fillna(self.cleaned_data[col].mean(), inplace=True)
            
            elif method_to_use == 'median':
                self.cleaned_data[col].fillna(self.cleaned_data[col].median(), inplace=True)
            
            elif method_to_use == 'mode':
                mode_value = self.cleaned_data[col].mode().iloc[0] if not self.cleaned_data[col].mode().empty else 0
                self.cleaned_data[col].fillna(mode_value, inplace=True)
            
            elif method_to_use == 'knn':
                n_neighbors = kwargs.get('n_neighbors', 5)
                numeric_cols = self.cleaned_data.select_dtypes(include=[np.number]).columns
                imputer = KNNImputer(n_neighbors=n_neighbors)
                self.cleaned_data[numeric_cols] = imputer.fit_transform(self.cleaned_data[numeric_cols])
            
            elif method_to_use == 'forward_fill':
                self.cleaned_data[col].fillna(method='ffill', inplace=True)
            
            elif method_to_use == 'backward_fill':
                self.cleaned_data[col].fillna(method='bfill', inplace=True)
            
            elif method_to_use == 'interpolate':
                self.cleaned_data[col] = self.cleaned_data[col].interpolate(method='linear')
            
            imputation_report[col] = {
                'method_used': method_to_use,
                'original_missing': original_missing,
                'remaining_missing': self.cleaned_data[col].isnull().sum()
            }
        
        self.log_operation('imputation', f'Imputed missing values using {method} method')
        return imputation_report
    
    def _choose_best_imputation_method(self, column):
        """Automatically choose the best imputation method for a column"""
        col_data = self.data[column].dropna()
        
        if len(col_data) == 0:
            return 'mean'
        
        # For numeric data
        if self.data[column].dtype in ['int64', 'float64']:
            # Check distribution
            skewness = abs(stats.skew(col_data))
            
            if skewness > 1:  # Highly skewed
                return 'median'
            else:
                return 'mean'
        
        # For categorical data
        else:
            return 'mode'
    
    def advanced_outlier_detection(self, methods=['iqr', 'zscore', 'isolation_forest'], columns=None):
        """Advanced outlier detection using multiple methods"""
        if columns is None:
            columns = self.data.select_dtypes(include=[np.number]).columns.tolist()
        
        outlier_results = {}
        
        for col in columns:
            if col not in self.data.columns:
                continue
            
            col_data = self.data[col].dropna()
            if len(col_data) == 0:
                continue
            
            outlier_results[col] = {}
            
            # IQR Method
            if 'iqr' in methods:
                Q1 = col_data.quantile(0.25)
                Q3 = col_data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                iqr_outliers = col_data[(col_data < lower_bound) | (col_data > upper_bound)]
                
                outlier_results[col]['iqr'] = {
                    'count': len(iqr_outliers),
                    'percentage': len(iqr_outliers) / len(col_data) * 100,
                    'outliers': iqr_outliers.tolist(),
                    'bounds': {'lower': lower_bound, 'upper': upper_bound}
                }
            
            # Z-Score Method
            if 'zscore' in methods:
                z_scores = np.abs(stats.zscore(col_data))
                zscore_outliers = col_data[z_scores > 3]
                
                outlier_results[col]['zscore'] = {
                    'count': len(zscore_outliers),
                    'percentage': len(zscore_outliers) / len(col_data) * 100,
                    'outliers': zscore_outliers.tolist(),
                    'threshold': 3
                }
            
            # Isolation Forest Method
            if 'isolation_forest' in methods and len(col_data) > 10:
                iso_forest = IsolationForest(contamination=0.1, random_state=42)
                outlier_labels = iso_forest.fit_predict(col_data.values.reshape(-1, 1))
                iso_outliers = col_data[outlier_labels == -1]
                
                outlier_results[col]['isolation_forest'] = {
                    'count': len(iso_outliers),
                    'percentage': len(iso_outliers) / len(col_data) * 100,
                    'outliers': iso_outliers.tolist()
                }
            
            # Local Outlier Factor
            if 'lof' in methods and len(col_data) > 20:
                lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
                outlier_labels = lof.fit_predict(col_data.values.reshape(-1, 1))
                lof_outliers = col_data[outlier_labels == -1]
                
                outlier_results[col]['lof'] = {
                    'count': len(lof_outliers),
                    'percentage': len(lof_outliers) / len(col_data) * 100,
                    'outliers': lof_outliers.tolist()
                }
        
        self.log_operation('outlier_detection', f'Detected outliers using methods: {methods}')
        return outlier_results
    
    def handle_outliers_advanced(self, method='winsorize', columns=None, **kwargs):
        """Advanced outlier handling with multiple methods"""
        if columns is None:
            columns = self.data.select_dtypes(include=[np.number]).columns.tolist()
        
        if self.cleaned_data is None:
            self.cleaned_data = self.data.copy()
        
        handling_report = {}
        
        for col in columns:
            if col not in self.cleaned_data.columns:
                continue
            
            original_data = self.cleaned_data[col].copy()
            col_data = original_data.dropna()
            
            if len(col_data) == 0:
                continue
            
            if method == 'winsorize':
                lower_percentile = kwargs.get('lower_percentile', 5)
                upper_percentile = kwargs.get('upper_percentile', 95)
                
                lower_bound = np.percentile(col_data, lower_percentile)
                upper_bound = np.percentile(col_data, upper_percentile)
                
                self.cleaned_data[col] = self.cleaned_data[col].clip(lower=lower_bound, upper=upper_bound)
                
                handling_report[col] = {
                    'method': 'winsorize',
                    'lower_bound': lower_bound,
                    'upper_bound': upper_bound,
                    'values_modified': ((original_data < lower_bound) | (original_data > upper_bound)).sum()
                }
            
            elif method == 'remove':
                Q1 = col_data.quantile(0.25)
                Q3 = col_data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                original_length = len(self.cleaned_data)
                self.cleaned_data = self.cleaned_data[
                    (self.cleaned_data[col] >= lower_bound) & 
                    (self.cleaned_data[col] <= upper_bound)
                ]
                
                handling_report[col] = {
                    'method': 'remove',
                    'rows_removed': original_length - len(self.cleaned_data),
                    'lower_bound': lower_bound,
                    'upper_bound': upper_bound
                }
            
            elif method == 'transform':
                # Log transformation for positive skewed data
                if col_data.min() > 0:
                    self.cleaned_data[col] = np.log1p(self.cleaned_data[col])
                    handling_report[col] = {'method': 'log_transform'}
                else:
                    # Box-Cox transformation
                    try:
                        transformed_data, lambda_param = stats.boxcox(col_data + 1)  # Add 1 to handle zeros
                        self.cleaned_data[col] = transformed_data
                        handling_report[col] = {'method': 'boxcox_transform', 'lambda': lambda_param}
                    except:
                        handling_report[col] = {'method': 'transform_failed'}
        
        self.log_operation('outlier_handling', f'Handled outliers using {method} method')
        return handling_report
    
    def add_validation_rule(self, rule_type, column, condition, **kwargs):
        """Add custom validation rules"""
        rule = {
            'type': rule_type,
            'column': column,
            'condition': condition,
            'kwargs': kwargs,
            'id': len(self.validation_rules)
        }
        self.validation_rules.append(rule)
        return rule['id']
    
    def validate_data(self):
        """Apply validation rules and return violations"""
        violations = []
        
        for rule in self.validation_rules:
            rule_violations = self._apply_validation_rule(rule)
            violations.extend(rule_violations)
        
        # Built-in validation rules
        violations.extend(self._check_data_consistency())
        violations.extend(self._check_skip_patterns())
        
        self.log_operation('validation', f'Found {len(violations)} validation violations')
        return violations
    
    def _apply_validation_rule(self, rule):
        """Apply a single validation rule"""
        violations = []
        
        if rule['column'] not in self.data.columns:
            return violations
        
        col_data = self.data[rule['column']]
        
        if rule['type'] == 'range':
            min_val = rule['kwargs'].get('min_val')
            max_val = rule['kwargs'].get('max_val')
            
            if min_val is not None:
                violations.extend(self.data[col_data < min_val].index.tolist())
            if max_val is not None:
                violations.extend(self.data[col_data > max_val].index.tolist())
        
        elif rule['type'] == 'categorical':
            allowed_values = rule['kwargs'].get('allowed_values', [])
            invalid_mask = ~col_data.isin(allowed_values)
            violations.extend(self.data[invalid_mask].index.tolist())
        
        elif rule['type'] == 'regex':
            pattern = rule['kwargs'].get('pattern')
            if pattern:
                invalid_mask = ~col_data.astype(str).str.match(pattern, na=False)
                violations.extend(self.data[invalid_mask].index.tolist())
        
        return [{'rule_id': rule['id'], 'row_index': idx, 'rule_type': rule['type']} for idx in violations]
    
    def _check_data_consistency(self):
        """Check for data consistency issues"""
        violations = []
        
        # Check for duplicate rows
        duplicates = self.data.duplicated()
        if duplicates.any():
            violations.extend([
                {'rule_type': 'duplicate_row', 'row_index': idx}
                for idx in self.data[duplicates].index.tolist()
            ])
        
        return violations
    
    def _check_skip_patterns(self):
        """Check for skip pattern violations in survey data"""
        violations = []
        
        # Example: If age < 18, employment status should be 'student' or 'unemployed'
        # This would be customizable based on survey design
        
        return violations
    
    def apply_survey_weights(self, weight_column=None, weight_method='design', **kwargs):
        """Apply survey weights with different methodologies"""
        if self.cleaned_data is None:
            self.cleaned_data = self.data.copy()
        
        if weight_column and weight_column in self.cleaned_data.columns:
            self.weights = self.cleaned_data[weight_column].copy()
            
            # Handle missing weights
            if self.weights.isnull().any():
                if weight_method == 'mean_imputation':
                    self.weights.fillna(self.weights.mean(), inplace=True)
                elif weight_method == 'unit_weight':
                    self.weights.fillna(1.0, inplace=True)
                else:
                    # Remove rows with missing weights
                    valid_weights_mask = self.weights.notna()
                    self.cleaned_data = self.cleaned_data[valid_weights_mask]
                    self.weights = self.weights[valid_weights_mask]
        
        else:
            # Create equal weights
            self.weights = pd.Series(1.0, index=self.cleaned_data.index)
        
        # Normalize weights if requested
        if kwargs.get('normalize_weights', False):
            self.weights = self.weights / self.weights.sum() * len(self.weights)
        
        # Trim extreme weights if requested
        if kwargs.get('trim_weights', False):
            trim_percentile = kwargs.get('trim_percentile', 95)
            upper_bound = np.percentile(self.weights, trim_percentile)
            self.weights = self.weights.clip(upper=upper_bound)
        
        self.log_operation('weight_application', f'Applied {weight_method} weights')
        return self.weights
    
    def calculate_weighted_estimates(self, variables=None, confidence_level=0.95):
        """Calculate comprehensive weighted estimates with margins of error"""
        if variables is None:
            variables = self.cleaned_data.select_dtypes(include=[np.number]).columns.tolist()
        
        if self.weights is None:
            self.apply_survey_weights()
        
        estimates = {}
        alpha = 1 - confidence_level
        z_score = stats.norm.ppf(1 - alpha/2)
        
        for var in variables:
            if var not in self.cleaned_data.columns:
                continue
            
            # Get valid data and corresponding weights
            valid_mask = self.cleaned_data[var].notna()
            data = self.cleaned_data.loc[valid_mask, var]
            weights = self.weights[valid_mask]
            
            if len(data) == 0:
                continue
            
            # Calculate weighted statistics using statsmodels
            weighted_stats = DescrStatsW(data, weights=weights, ddof=1)
            
            # Unweighted estimates
            unweighted_mean = data.mean()
            unweighted_std = data.std()
            unweighted_se = unweighted_std / np.sqrt(len(data))
            unweighted_moe = z_score * unweighted_se
            
            # Weighted estimates
            weighted_mean = weighted_stats.mean
            weighted_std = weighted_stats.std
            weighted_se = weighted_stats.std_mean
            weighted_moe = z_score * weighted_se
            
            # Additional statistics
            weighted_var = weighted_stats.var
            design_effect = weighted_var / (unweighted_std**2) if unweighted_std > 0 else 1
            effective_sample_size = len(data) / design_effect if design_effect > 0 else len(data)
            
            estimates[var] = {
                'unweighted': {
                    'mean': float(unweighted_mean),
                    'std': float(unweighted_std),
                    'se': float(unweighted_se),
                    'moe': float(unweighted_moe),
                    'ci_lower': float(unweighted_mean - unweighted_moe),
                    'ci_upper': float(unweighted_mean + unweighted_moe),
                    'n': int(len(data))
                },
                'weighted': {
                    'mean': float(weighted_mean),
                    'std': float(weighted_std),
                    'se': float(weighted_se),
                    'moe': float(weighted_moe),
                    'ci_lower': float(weighted_mean - weighted_moe),
                    'ci_upper': float(weighted_mean + weighted_moe),
                    'design_effect': float(design_effect),
                    'effective_n': float(effective_sample_size)
                }
            }
        
        self.estimates = estimates
        self.log_operation('estimation', f'Calculated estimates for {len(estimates)} variables')
        return estimates
    
    def generate_comprehensive_visualizations(self):
        """Generate comprehensive data visualizations"""
        if self.cleaned_data is None:
            return None
        
        visualizations = {}
        
        # 1. Data Quality Dashboard
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('Data Quality and Analysis Dashboard', fontsize=16, fontweight='bold')
        
        # Missing values heatmap
        if self.data.isnull().any().any():
            sns.heatmap(self.data.isnull(), cbar=True, ax=axes[0,0], cmap='viridis')
            axes[0,0].set_title('Missing Values Pattern')
        else:
            axes[0,0].text(0.5, 0.5, 'No Missing Values', ha='center', va='center', transform=axes[0,0].transAxes)
            axes[0,0].set_title('Missing Values Pattern')
        
        # Data quality scores
        if hasattr(self, 'data_profile') and 'data_quality' in self.data_profile:
            quality_data = self.data_profile['data_quality']
            columns = list(quality_data.keys())[:10]  # Limit to 10 columns
            completeness = [quality_data[col]['completeness'] for col in columns]
            
            axes[0,1].bar(range(len(columns)), completeness, color='skyblue')
            axes[0,1].set_xticks(range(len(columns)))
            axes[0,1].set_xticklabels(columns, rotation=45, ha='right')
            axes[0,1].set_ylabel('Completeness %')
            axes[0,1].set_title('Data Completeness by Column')
        
        # Distribution of first numeric variable
        numeric_cols = self.cleaned_data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            col = numeric_cols[0]
            data = self.cleaned_data[col].dropna()
            if len(data) > 0:
                axes[0,2].hist(data, bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
                axes[0,2].axvline(data.mean(), color='red', linestyle='--', label=f'Mean: {data.mean():.2f}')
                axes[0,2].axvline(data.median(), color='blue', linestyle='--', label=f'Median: {data.median():.2f}')
                axes[0,2].set_xlabel(col)
                axes[0,2].set_ylabel('Frequency')
                axes[0,2].set_title(f'Distribution of {col}')
                axes[0,2].legend()
        
        # Box plots for outlier detection
        if len(numeric_cols) > 0:
            plot_cols = numeric_cols[:5]  # Limit to 5 columns for readability
            box_data = [self.cleaned_data[col].dropna() for col in plot_cols]
            axes[1,0].boxplot(box_data, labels=plot_cols)
            axes[1,0].set_title('Box Plots - Outlier Detection')
            axes[1,0].tick_params(axis='x', rotation=45)
        
        # Correlation heatmap
        if len(numeric_cols) > 1:
            corr_matrix = self.cleaned_data[numeric_cols].corr()
            im = axes[1,1].imshow(corr_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
            axes[1,1].set_xticks(range(len(corr_matrix.columns)))
            axes[1,1].set_yticks(range(len(corr_matrix.columns)))
            axes[1,1].set_xticklabels(corr_matrix.columns, rotation=45, ha='right')
            axes[1,1].set_yticklabels(corr_matrix.columns)
            axes[1,1].set_title('Correlation Matrix')
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=axes[1,1], shrink=0.8)
            cbar.set_label('Correlation Coefficient')
        
        # Estimates comparison (if available)
        if self.estimates:
            vars_to_plot = list(self.estimates.keys())[:5]
            unweighted_means = [self.estimates[var]['unweighted']['mean'] for var in vars_to_plot]
            weighted_means = [self.estimates[var]['weighted']['mean'] for var in vars_to_plot]
            
            x = np.arange(len(vars_to_plot))
            width = 0.35
            
            axes[1,2].bar(x - width/2, unweighted_means, width, label='Unweighted', alpha=0.7)
            axes[1,2].bar(x + width/2, weighted_means, width, label='Weighted', alpha=0.7)
            axes[1,2].set_xlabel('Variables')
            axes[1,2].set_ylabel('Mean Values')
            axes[1,2].set_title('Weighted vs Unweighted Estimates')
            axes[1,2].set_xticks(x)
            axes[1,2].set_xticklabels(vars_to_plot, rotation=45, ha='right')
            axes[1,2].legend()
        
        plt.tight_layout()
        
        # Save main dashboard
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
        img_buffer.seek(0)
        visualizations['dashboard'] = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close()
        
        # 2. Generate individual plots
        if self.estimates:
            visualizations.update(self._generate_estimates_plots())
        
        return visualizations
    
    def _generate_estimates_plots(self):
        """Generate specific plots for estimates"""
        plots = {}
        
        # Confidence intervals plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        variables = list(self.estimates.keys())
        y_pos = np.arange(len(variables))
        
        # Extract confidence intervals
        weighted_means = [self.estimates[var]['weighted']['mean'] for var in variables]
        ci_lowers = [self.estimates[var]['weighted']['ci_lower'] for var in variables]
        ci_uppers = [self.estimates[var]['weighted']['ci_upper'] for var in variables]
        
        # Create error bars
        errors = [[mean - lower for mean, lower in zip(weighted_means, ci_lowers)],
                  [upper - mean for mean, upper in zip(weighted_means, ci_uppers)]]
        
        ax.errorbar(weighted_means, y_pos, xerr=errors, fmt='o', capsize=5, capthick=2)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(variables)
        ax.set_xlabel('Estimated Mean')
        ax.set_title('Weighted Estimates with Confidence Intervals')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
        img_buffer.seek(0)
        plots['confidence_intervals'] = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close()
        
        return plots
    
    def generate_audit_trail(self):
        """Generate comprehensive audit trail"""
        audit_trail = {
            'processing_log': self.processing_log,
            'data_transformations': {
                'original_shape': self.original_data.shape if self.original_data is not None else None,
                'final_shape': self.cleaned_data.shape if self.cleaned_data is not None else None,
                'columns_added': [],
                'columns_removed': [],
                'rows_removed': 0
            },
            'validation_results': self.validate_data() if hasattr(self, 'validation_rules') else [],
            'quality_metrics': self.get_data_profile() if self.data is not None else {},
            'processing_summary': {
                'total_operations': len(self.processing_log),
                'start_time': self.processing_log[0]['timestamp'] if self.processing_log else None,
                'end_time': self.processing_log[-1]['timestamp'] if self.processing_log else None
            }
        }
        
        return audit_trail

# Global processor instance
processor = AdvancedDataProcessor()

# API Routes
@app.route('/')
def index():
    return jsonify({
        'message': 'AI Enhanced Survey Analysis API',
        'version': '2.0',
        'endpoints': {
            'upload': '/upload - POST - Upload CSV/Excel files',
            'profile': '/profile - GET - Get data profiling information',
            'clean': '/clean - POST - Clean data with advanced methods',
            'validate': '/validate - POST - Validate data with custom rules',
            'analyze': '/analyze - POST - Perform comprehensive analysis',
            'report': '/report - POST - Generate detailed reports',
            'export': '/export - POST - Export cleaned data and results'
        }
    })

@app.route('/upload', methods=['POST'])
def upload_file():
    """Enhanced file upload with better validation and error handling"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Please upload CSV or Excel files.'}), 400
        
        # Save file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Load and profile data
        sheet_name = request.form.get('sheet_name')
        data = processor.load_data(filepath, sheet_name)
        
        # Generate comprehensive data profile
        data_profile = processor.get_data_profile()
        
        session['current_file'] = filepath
        session['upload_time'] = datetime.now().isoformat()
        
        return jsonify({
            'success': True,
            'message': 'File uploaded and profiled successfully',
            'filename': filename,
            'data_profile': data_profile,
            'processing_log': processor.processing_log
        })
    
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        return jsonify({'error': f'Error processing file: {str(e)}'}), 500

@app.route('/profile', methods=['GET'])
def get_data_profile():
    """Get comprehensive data profiling information"""
    try:
        if processor.data is None:
            return jsonify({'error': 'No data loaded. Please upload a file first.'}), 400
        
        profile = processor.get_data_profile()
        return jsonify({
            'success': True,
            'profile': profile
        })
    
    except Exception as e:
        logger.error(f"Profiling error: {str(e)}")
        return jsonify({'error': f'Error generating profile: {str(e)}'}), 500

@app.route('/clean', methods=['POST'])
def clean_data():
    """Advanced data cleaning with multiple methods"""
    try:
        if processor.data is None:
            return jsonify({'error': 'No data loaded. Please upload a file first.'}), 400
        
        config = request.json or {}
        
        cleaning_results = {}
        
        # Missing value imputation
        if config.get('imputation'):
            imputation_config = config['imputation']
            imputation_results = processor.advanced_missing_imputation(
                method=imputation_config.get('method', 'auto'),
                columns=imputation_config.get('columns'),
                **imputation_config.get('kwargs', {})
            )
            cleaning_results['imputation'] = imputation_results
        
        # Outlier detection and handling
        if config.get('outliers'):
            outlier_config = config['outliers']
            
            # Detect outliers
            outlier_detection_results = processor.advanced_outlier_detection(
                methods=outlier_config.get('detection_methods', ['iqr', 'zscore']),
                columns=outlier_config.get('columns')
            )
            cleaning_results['outlier_detection'] = outlier_detection_results
            
            # Handle outliers
            if outlier_config.get('handling_method'):
                outlier_handling_results = processor.handle_outliers_advanced(
                    method=outlier_config['handling_method'],
                    columns=outlier_config.get('columns'),
                    **outlier_config.get('kwargs', {})
                )
                cleaning_results['outlier_handling'] = outlier_handling_results
        
        # Data validation
        validation_results = processor.validate_data()
        cleaning_results['validation'] = validation_results
        
        return jsonify({
            'success': True,
            'cleaning_results': cleaning_results,
            'cleaned_data_shape': processor.cleaned_data.shape if processor.cleaned_data is not None else None,
            'processing_log': processor.processing_log[-10:]  # Last 10 operations
        })
    
    except Exception as e:
        logger.error(f"Cleaning error: {str(e)}")
        return jsonify({'error': f'Error cleaning data: {str(e)}'}), 500

@app.route('/validate', methods=['POST'])
def add_validation_rules():
    """Add custom validation rules"""
    try:
        rules = request.json.get('rules', [])
        
        added_rules = []
        for rule in rules:
            rule_id = processor.add_validation_rule(
                rule_type=rule['type'],
                column=rule['column'],
                condition=rule['condition'],
                **rule.get('kwargs', {})
            )
            added_rules.append({'rule_id': rule_id, 'rule': rule})
        
        return jsonify({
            'success': True,
            'added_rules': added_rules,
            'total_rules': len(processor.validation_rules)
        })
    
    except Exception as e:
        logger.error(f"Validation error: {str(e)}")
        return jsonify({'error': f'Error adding validation rules: {str(e)}'}), 500

@app.route('/analyze', methods=['POST'])
def analyze_data():
    """Comprehensive data analysis with weighting and estimation"""
    try:
        if processor.data is None:
            return jsonify({'error': 'No data loaded. Please upload a file first.'}), 400
        
        config = request.json or {}
        
        # Apply survey weights
        weight_config = config.get('weights', {})
        weights = processor.apply_survey_weights(
            weight_column=weight_config.get('column'),
            weight_method=weight_config.get('method', 'design'),
            **weight_config.get('kwargs', {})
        )
        
        # Calculate weighted estimates
        estimation_config = config.get('estimation', {})
        estimates = processor.calculate_weighted_estimates(
            variables=estimation_config.get('variables'),
            confidence_level=estimation_config.get('confidence_level', 0.95)
        )
        
        # Generate visualizations
        visualizations = processor.generate_comprehensive_visualizations()
        
        # Generate audit trail
        audit_trail = processor.generate_audit_trail()
        
        return jsonify({
            'success': True,
            'estimates': estimates,
            'visualizations': visualizations,
            'audit_trail': audit_trail,
            'summary': {
                'total_observations': len(processor.cleaned_data) if processor.cleaned_data is not None else 0,
                'variables_analyzed': len(estimates),
                'weight_column': weight_config.get('column', 'equal_weights'),
                'confidence_level': estimation_config.get('confidence_level', 0.95)
            }
        })
    
    except Exception as e:
        logger.error(f"Analysis error: {str(e)}")
        return jsonify({'error': f'Error in analysis: {str(e)}'}), 500

@app.route('/report', methods=['POST'])
def generate_comprehensive_report():
    """Generate comprehensive PDF and HTML reports"""
    try:
        if processor.data is None:
            return jsonify({'error': 'No data loaded. Please upload a file first.'}), 400
        
        report_config = request.json or {}
        report_format = report_config.get('format', 'pdf')  # 'pdf', 'html', or 'both'
        
        # Generate report content
        report_data = {
            'title': report_config.get('title', 'Survey Data Analysis Report'),
            'author': report_config.get('author', 'AI Enhanced Survey Analysis System'),
            'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'data_profile': processor.get_data_profile(),
            'estimates': processor.estimates,
            'audit_trail': processor.generate_audit_trail(),
            'visualizations': processor.generate_comprehensive_visualizations()
        }
        
        reports_generated = []
        
        if report_format in ['pdf', 'both']:
            pdf_path = generate_pdf_report(report_data)
            reports_generated.append({'type': 'pdf', 'path': pdf_path})
        
        if report_format in ['html', 'both']:
            html_path = generate_html_report(report_data)
            reports_generated.append({'type': 'html', 'path': html_path})
        
        return jsonify({
            'success': True,
            'reports': reports_generated,
            'download_urls': [f'/download/{os.path.basename(report["path"])}' for report in reports_generated]
        })
    
    except Exception as e:
        logger.error(f"Report generation error: {str(e)}")
        return jsonify({'error': f'Error generating report: {str(e)}'}), 500

def generate_pdf_report(report_data):
    """Generate comprehensive PDF report"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    pdf_path = f"reports/survey_analysis_report_{timestamp}.pdf"
    
    doc = SimpleDocTemplate(pdf_path, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30,
        alignment=1,
        textColor=colors.darkblue
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        spaceAfter=12,
        textColor=colors.darkblue
    )
    
    # Title page
    story.append(Paragraph(report_data['title'], title_style))
    story.append(Spacer(1, 20))
    story.append(Paragraph(f"Generated by: {report_data['author']}", styles['Normal']))
    story.append(Paragraph(f"Date: {report_data['date']}", styles['Normal']))
    story.append(Spacer(1, 40))
    
    # Executive Summary
    story.append(Paragraph("Executive Summary", heading_style))
    data_profile = report_data['data_profile']
    if data_profile and 'basic_info' in data_profile:
        basic_info = data_profile['basic_info']
        summary_text = f"""
        This report presents the results of an automated survey data analysis performed on a dataset 
        containing {basic_info['shape'][0]:,} observations and {basic_info['shape'][1]} variables. 
        The analysis included comprehensive data cleaning, validation, weighting, and statistical estimation 
        with confidence intervals.
        """
        story.append(Paragraph(summary_text, styles['Normal']))
    story.append(Spacer(1, 20))
    
    # Data Overview
    story.append(Paragraph("Data Overview", heading_style))
    if data_profile and 'basic_info' in data_profile:
        basic_info = data_profile['basic_info']
        
        # Create data overview table
        overview_data = [
            ['Metric', 'Value'],
            ['Total Observations', f"{basic_info['shape'][0]:,}"],
            ['Total Variables', f"{basic_info['shape'][1]}"],
            ['Memory Usage', f"{basic_info['memory_usage'] / 1024 / 1024:.2f} MB"],
            ['Missing Values', f"{data_profile.get('missing_analysis', {}).get('total_missing', 0):,}"]
        ]
        
        overview_table = Table(overview_data)
        overview_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(overview_table)
        story.append(Spacer(1, 20))
    
    # Statistical Estimates
    if report_data['estimates']:
        story.append(Paragraph("Statistical Estimates", heading_style))
        story.append(Paragraph(
            "The following table presents weighted and unweighted estimates with 95% confidence intervals:",
            styles['Normal']
        ))
        story.append(Spacer(1, 12))
        
        # Create estimates table
        estimates_data = [['Variable', 'Unweighted Mean', 'Weighted Mean', 'Margin of Error', 'Design Effect']]
        for var, est in report_data['estimates'].items():
            estimates_data.append([
                var,
                f"{est['unweighted']['mean']:.4f}",
                f"{est['weighted']['mean']:.4f}",
                f"±{est['weighted']['moe']:.4f}",
                f"{est['weighted'].get('design_effect', 1):.2f}"
            ])
        
        estimates_table = Table(estimates_data)
        estimates_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightgreen),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(estimates_table)
        story.append(Spacer(1, 20))
    
    # Data Quality Assessment
    if data_profile and 'data_quality' in data_profile:
        story.append(Paragraph("Data Quality Assessment", heading_style))
        
        quality_data = [['Variable', 'Completeness %', 'Uniqueness %', 'Consistency %', 'Validity %']]
        for var, metrics in list(data_profile['data_quality'].items())[:10]:  # Limit to 10 variables
            quality_data.append([
                var,
                f"{metrics['completeness']:.1f}%",
                f"{metrics['uniqueness']:.1f}%",
                f"{metrics['consistency']:.1f}%",
                f"{metrics['validity']:.1f}%"
            ])
        
        quality_table = Table(quality_data)
        quality_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.orange),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(quality_table)
        story.append(Spacer(1, 20))
    
    # Processing Log
    if report_data['audit_trail'] and 'processing_log' in report_data['audit_trail']:
        story.append(Paragraph("Processing Audit Trail", heading_style))
        
        log_data = [['Timestamp', 'Operation', 'Details']]
        for log_entry in report_data['audit_trail']['processing_log'][-10:]:  # Last 10 operations
            log_data.append([
                log_entry['timestamp'][:19],  # Remove microseconds
                log_entry['operation'],
                log_entry['details'][:50] + '...' if len(log_entry['details']) > 50 else log_entry['details']
            ])
        
        log_table = Table(log_data)
        log_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 9),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(log_table)
    
    # Build PDF
    doc.build(story)
    
    return pdf_path

def generate_html_report(report_data):
    """Generate comprehensive HTML report"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    html_path = f"reports/survey_analysis_report_{timestamp}.html"
    
    # HTML template
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>{report_data['title']}</title>
        <style>
            body {{
                font-family: 'Arial', sans-serif;
                line-height: 1.6;
                margin: 0;
                padding: 20px;
                background-color: #f5f5f5;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
                background-color: white;
                padding: 40px;
                border-radius: 10px;
                box-shadow: 0 0 20px rgba(0,0,0,0.1);
            }}
            h1 {{
                color: #2c3e50;
                text-align: center;
                border-bottom: 3px solid #3498db;
                padding-bottom: 10px;
            }}
            h2 {{
                color: #34495e;
                border-left: 4px solid #3498db;
                padding-left: 15px;
                margin-top: 30px;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
            }}
            th, td {{
                border: 1px solid #ddd;
                padding: 12px;
                text-align: left;
            }}
            th {{
                background-color: #3498db;
                color: white;
            }}
            tr:nth-child(even) {{
                background-color: #f9f9f9;
            }}
            .summary {{
                background-color: #ecf0f1;
                padding: 20px;
                border-radius: 5px;
                margin: 20px 0;
            }}
            .visualization {{
                text-align: center;
                margin: 20px 0;
            }}
            .visualization img {{
                max-width: 100%;
                height: auto;
                border: 1px solid #ddd;
                border-radius: 5px;
            }}
            .metric {{
                display: inline-block;
                background-color: #3498db;
                color: white;
                padding: 10px 20px;
                margin: 5px;
                border-radius: 25px;
                font-weight: bold;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>{report_data['title']}</h1>
            
            <div class="summary">
                <p><strong>Generated by:</strong> {report_data['author']}</p>
                <p><strong>Date:</strong> {report_data['date']}</p>
            </div>
    """
    
    # Add data overview
    if report_data['data_profile'] and 'basic_info' in report_data['data_profile']:
        basic_info = report_data['data_profile']['basic_info']
        html_content += f"""
            <h2>Data Overview</h2>
            <div class="summary">
                <span class="metric">Observations: {basic_info['shape'][0]:,}</span>
                <span class="metric">Variables: {basic_info['shape'][1]}</span>
                <span class="metric">Memory: {basic_info['memory_usage'] / 1024 / 1024:.2f} MB</span>
            </div>
        """
    
    # Add estimates table
    if report_data['estimates']:
        html_content += """
            <h2>Statistical Estimates</h2>
            <table>
                <thead>
                    <tr>
                        <th>Variable</th>
                        <th>Unweighted Mean</th>
                        <th>Weighted Mean</th>
                        <th>Margin of Error</th>
                        <th>95% CI Lower</th>
                        <th>95% CI Upper</th>
                        <th>Design Effect</th>
                    </tr>
                </thead>
                <tbody>
        """
        
        for var, est in report_data['estimates'].items():
            html_content += f"""
                <tr>
                    <td>{var}</td>
                    <td>{est['unweighted']['mean']:.4f}</td>
                    <td>{est['weighted']['mean']:.4f}</td>
                    <td>±{est['weighted']['moe']:.4f}</td>
                    <td>{est['weighted']['ci_lower']:.4f}</td>
                    <td>{est['weighted']['ci_upper']:.4f}</td>
                    <td>{est['weighted'].get('design_effect', 1):.2f}</td>
                </tr>
            """
        
        html_content += """
                </tbody>
            </table>
        """
    
    # Add visualizations
    if report_data['visualizations']:
        html_content += "<h2>Data Visualizations</h2>"
        for viz_name, viz_data in report_data['visualizations'].items():
            html_content += f"""
                <div class="visualization">
                    <h3>{viz_name.replace('_', ' ').title()}</h3>
                    <img src="data:image/png;base64,{viz_data}" alt="{viz_name}">
                </div>
            """
    
    # Add audit trail
    if report_data['audit_trail'] and 'processing_log' in report_data['audit_trail']:
        html_content += """
            <h2>Processing Audit Trail</h2>
            <table>
                <thead>
                    <tr>
                        <th>Timestamp</th>
                        <th>Operation</th>
                        <th>Details</th>
                    </tr>
                </thead>
                <tbody>
        """
        
        for log_entry in report_data['audit_trail']['processing_log'][-15:]:  # Last 15 operations
            html_content += f"""
                <tr>
                    <td>{log_entry['timestamp'][:19]}</td>
                    <td>{log_entry['operation']}</td>
                    <td>{log_entry['details']}</td>
                </tr>
            """
        
        html_content += """
                </tbody>
            </table>
        """
    
    # Close HTML
    html_content += """
        </div>
    </body>
    </html>
    """
    
    # Write HTML file
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    return html_path

@app.route('/export', methods=['POST'])
def export_data():
    """Export cleaned data and results in various formats"""
    try:
        if processor.cleaned_data is None:
            return jsonify({'error': 'No cleaned data available. Please clean the data first.'}), 400
        
        export_config = request.json or {}
        export_format = export_config.get('format', 'csv')  # 'csv', 'excel', 'json'
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if export_format == 'csv':
            export_path = f"reports/cleaned_data_{timestamp}.csv"
            processor.cleaned_data.to_csv(export_path, index=False)
        
        elif export_format == 'excel':
            export_path = f"reports/cleaned_data_{timestamp}.xlsx"
            with pd.ExcelWriter(export_path, engine='xlsxwriter') as writer:
                processor.cleaned_data.to_excel(writer, sheet_name='Cleaned_Data', index=False)
                
                if processor.estimates:
                    # Create estimates summary sheet
                    estimates_df = pd.DataFrame.from_dict(
                        {var: {**est['weighted'], **est['unweighted']} 
                         for var, est in processor.estimates.items()}, 
                        orient='index'
                    )
                    estimates_df.to_excel(writer, sheet_name='Estimates')
                
                # Add audit trail sheet
                if processor.processing_log:
                    audit_df = pd.DataFrame(processor.processing_log)
                    audit_df.to_excel(writer, sheet_name='Audit_Trail', index=False)
        
        elif export_format == 'json':
            export_path = f"reports/analysis_results_{timestamp}.json"
            export_data = {
                'cleaned_data': processor.cleaned_data.to_dict('records'),
                'estimates': processor.estimates,
                'audit_trail': processor.generate_audit_trail(),
                'data_profile': processor.get_data_profile()
            }
            
            with open(export_path, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
        
        return jsonify({
            'success': True,
            'export_path': export_path,
            'download_url': f'/download/{os.path.basename(export_path)}'
        })
    
    except Exception as e:
        logger.error(f"Export error: {str(e)}")
        return jsonify({'error': f'Error exporting data: {str(e)}'}), 500

@app.route('/download/<filename>')
def download_file(filename):
    """Download generated reports and exports"""
    try:
        file_path = os.path.join('reports', filename)
        if os.path.exists(file_path):
            return send_file(file_path, as_attachment=True)
        else:
            return jsonify({'error': 'File not found'}), 404
    
    except Exception as e:
        logger.error(f"Download error: {str(e)}")
        return jsonify({'error': f'Error downloading file: {str(e)}'}), 500

@app.route('/columns')
def get_columns():
    """Get available columns for configuration"""
    try:
        if processor.data is None:
            return jsonify({'error': 'No data loaded'}), 400
        
        return jsonify({
            'success': True,
            'columns': processor.data.columns.tolist(),
            'numeric_columns': processor.data.select_dtypes(include=[np.number]).columns.tolist(),
            'categorical_columns': processor.data.select_dtypes(include=['object']).columns.tolist(),
            'datetime_columns': processor.data.select_dtypes(include=['datetime64']).columns.tolist()
        })
    
    except Exception as e:
        logger.error(f"Columns error: {str(e)}")
        return jsonify({'error': f'Error getting columns: {str(e)}'}), 500

@app.route('/health')
def health_check():
    """API health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '2.0',
        'data_loaded': processor.data is not None,
        'data_cleaned': processor.cleaned_data is not None,
        'estimates_available': bool(processor.estimates)
    })

# Error handlers
@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large. Maximum size is 100MB.'}), 413

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {str(error)}")
    return jsonify({'error': 'Internal server error. Please try again.'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
