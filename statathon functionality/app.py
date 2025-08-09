# Import enhanced functionality
from enhanced_app import *

# For backward compatibility, keep the original imports
import os
import json
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify, send_file, session
from werkzeug.utils import secure_filename
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
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

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

class DataProcessor:
    def __init__(self):
        self.data = None
        self.cleaned_data = None
        self.weights = None
        self.config = {}
        
    def load_data(self, file_path):
        """Load data from CSV or Excel file"""
        if file_path.endswith('.csv'):
            self.data = pd.read_csv(file_path)
        else:
            self.data = pd.read_excel(file_path)
        return self.data
    
    def detect_missing_values(self):
        """Detect missing values in the dataset"""
        missing_info = {
            'total_missing': self.data.isnull().sum().sum(),
            'missing_per_column': self.data.isnull().sum().to_dict(),
            'missing_percentage': (self.data.isnull().sum() / len(self.data) * 100).to_dict()
        }
        return missing_info
    
    def impute_missing_values(self, method='mean', columns=None):
        """Impute missing values using specified method"""
        if columns is None:
            columns = self.data.select_dtypes(include=[np.number]).columns
        
        self.cleaned_data = self.data.copy()
        
        for col in columns:
            if col in self.data.columns and self.data[col].isnull().any():
                if method == 'mean':
                    self.cleaned_data[col].fillna(self.data[col].mean(), inplace=True)
                elif method == 'median':
                    self.cleaned_data[col].fillna(self.data[col].median(), inplace=True)
                elif method == 'knn':
                    imputer = KNNImputer(n_neighbors=5)
                    self.cleaned_data[col] = imputer.fit_transform(self.data[[col]])
        
        return self.cleaned_data
    
    def detect_outliers(self, method='iqr', columns=None):
        """Detect outliers using specified method"""
        if columns is None:
            columns = self.data.select_dtypes(include=[np.number]).columns
        
        outliers_info = {}
        
        for col in columns:
            if col in self.data.columns:
                data = self.data[col].dropna()
                
                if method == 'iqr':
                    Q1 = data.quantile(0.25)
                    Q3 = data.quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    outliers = data[(data < lower_bound) | (data > upper_bound)]
                
                elif method == 'zscore':
                    z_scores = np.abs(stats.zscore(data))
                    outliers = data[z_scores > 3]
                
                outliers_info[col] = {
                    'count': len(outliers),
                    'percentage': len(outliers) / len(data) * 100,
                    'values': outliers.tolist()
                }
        
        return outliers_info
    
    def handle_outliers(self, method='winsorize', columns=None):
        """Handle outliers using specified method"""
        if columns is None:
            columns = self.data.select_dtypes(include=[np.number]).columns
        
        if self.cleaned_data is None:
            self.cleaned_data = self.data.copy()
        
        for col in columns:
            if col in self.cleaned_data.columns:
                data = self.cleaned_data[col].dropna()
                
                if method == 'winsorize':
                    # Winsorize at 5th and 95th percentiles
                    lower_percentile = np.percentile(data, 5)
                    upper_percentile = np.percentile(data, 95)
                    self.cleaned_data[col] = self.cleaned_data[col].clip(lower=lower_percentile, upper=upper_percentile)
                
                elif method == 'remove':
                    Q1 = data.quantile(0.25)
                    Q3 = data.quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    self.cleaned_data = self.cleaned_data[
                        (self.cleaned_data[col] >= lower_bound) & 
                        (self.cleaned_data[col] <= upper_bound)
                    ]
        
        return self.cleaned_data
    
    def apply_weights(self, weight_column=None):
        """Apply survey weights for estimation"""
        if weight_column and weight_column in self.cleaned_data.columns:
            self.weights = self.cleaned_data[weight_column]
        else:
            # Create equal weights if no weight column specified
            self.weights = pd.Series(1, index=self.cleaned_data.index)
        
        return self.weights
    
    def calculate_estimates(self, columns=None):
        """Calculate weighted and unweighted estimates with margins of error"""
        if columns is None:
            columns = self.cleaned_data.select_dtypes(include=[np.number]).columns
        
        estimates = {}
        
        for col in columns:
            if col in self.cleaned_data.columns:
                data = self.cleaned_data[col].dropna()
                weights = self.weights[data.index]
                
                # Unweighted estimates
                unweighted_mean = data.mean()
                unweighted_std = data.std()
                unweighted_se = data.std() / np.sqrt(len(data))
                unweighted_moe = 1.96 * unweighted_se
                
                # Weighted estimates
                weighted_mean = np.average(data, weights=weights)
                weighted_variance = np.average((data - weighted_mean)**2, weights=weights)
                weighted_std = np.sqrt(weighted_variance)
                weighted_se = weighted_std / np.sqrt(len(data))
                weighted_moe = 1.96 * weighted_se
                
                estimates[col] = {
                    'unweighted': {
                        'mean': unweighted_mean,
                        'std': unweighted_std,
                        'se': unweighted_se,
                        'moe': unweighted_moe
                    },
                    'weighted': {
                        'mean': weighted_mean,
                        'std': weighted_std,
                        'se': weighted_se,
                        'moe': weighted_moe
                    }
                }
        
        return estimates
    
    def generate_visualizations(self):
        """Generate visualizations for the data"""
        if self.cleaned_data is None:
            return None
        
        # Set style
        plt.style.use('seaborn-v0_8')
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Data Analysis Dashboard', fontsize=16, fontweight='bold')
        
        # 1. Missing values heatmap
        missing_data = self.cleaned_data.isnull()
        sns.heatmap(missing_data, cbar=True, ax=axes[0,0])
        axes[0,0].set_title('Missing Values Heatmap')
        
        # 2. Distribution plot for first numeric column
        numeric_cols = self.cleaned_data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            col = numeric_cols[0]
            sns.histplot(self.cleaned_data[col].dropna(), kde=True, ax=axes[0,1])
            axes[0,1].set_title(f'Distribution of {col}')
        
        # 3. Box plot for outliers
        if len(numeric_cols) > 0:
            self.cleaned_data[numeric_cols[:5]].boxplot(ax=axes[1,0])
            axes[1,0].set_title('Box Plot - Outlier Detection')
            axes[1,0].tick_params(axis='x', rotation=45)
        
        # 4. Correlation heatmap
        if len(numeric_cols) > 1:
            corr_matrix = self.cleaned_data[numeric_cols].corr()
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=axes[1,1])
            axes[1,1].set_title('Correlation Matrix')
        
        plt.tight_layout()
        
        # Save plot to base64 string
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
        img_buffer.seek(0)
        img_str = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close()
        
        return img_str

# Global data processor instance
processor = DataProcessor()

@app.route('/')
def index():
    return render_template('simple_index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    print("Upload endpoint called")  # Debug log
    
    if 'file' not in request.files:
        print("No file in request.files")  # Debug log
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    print(f"File received: {file.filename}, size: {file.content_length if hasattr(file, 'content_length') else 'unknown'}")  # Debug log
    
    if file.filename == '':
        print("Empty filename")  # Debug log
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        try:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            print(f"Saving file to: {filepath}")  # Debug log
            
            file.save(filepath)
            print(f"File saved successfully")  # Debug log
            
            # Load data
            data = processor.load_data(filepath)
            print(f"Data loaded: {data.shape}")  # Debug log
            
            # Get basic info
            data_info = {
                'shape': data.shape,
                'columns': data.columns.tolist(),
                'dtypes': data.dtypes.astype(str).to_dict(),
                'sample_data': data.head(5).to_dict('records')
            }
            
            session['current_file'] = filepath
            
            print(f"Upload successful: {data_info['shape']}")  # Debug log
            
            return jsonify({
                'success': True,
                'message': 'File uploaded successfully',
                'data_info': data_info
            })
        
        except Exception as e:
            print(f"Error processing file: {str(e)}")  # Debug log
            return jsonify({'error': f'Error processing file: {str(e)}'}), 500
    else:
        print(f"Invalid file type: {file.filename}")  # Debug log
        return jsonify({'error': 'Invalid file type'}), 400

@app.route('/analyze', methods=['POST'])
def analyze_data():
    try:
        config = request.json
        
        # Apply cleaning based on configuration
        if config.get('imputation_method'):
            processor.impute_missing_values(
                method=config['imputation_method'],
                columns=config.get('imputation_columns')
            )
        
        if config.get('outlier_method'):
            processor.handle_outliers(
                method=config['outlier_method'],
                columns=config.get('outlier_columns')
            )
        
        # Apply weights
        processor.apply_weights(config.get('weight_column'))
        
        # Calculate estimates
        estimates = processor.calculate_estimates(config.get('estimate_columns'))
        
        # Generate visualizations
        viz_img = processor.generate_visualizations()
        
        # Get missing values info
        missing_info = processor.detect_missing_values()
        
        # Get outliers info
        outliers_info = processor.detect_outliers()
        
        return jsonify({
            'success': True,
            'estimates': estimates,
            'missing_info': missing_info,
            'outliers_info': outliers_info,
            'visualization': viz_img,
            'cleaned_shape': processor.cleaned_data.shape if processor.cleaned_data is not None else None
        })
    
    except Exception as e:
        return jsonify({'error': f'Error in analysis: {str(e)}'}), 500

@app.route('/generate_report', methods=['POST'])
def generate_report():
    try:
        report_config = request.json
        
        # Create PDF report
        doc = SimpleDocTemplate("report.pdf", pagesize=A4)
        styles = getSampleStyleSheet()
        story = []
        
        # Title
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            alignment=1
        )
        story.append(Paragraph("Survey Data Analysis Report", title_style))
        story.append(Spacer(1, 20))
        
        # Report metadata
        story.append(Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
        story.append(Spacer(1, 20))
        
        # Data summary
        if processor.data is not None:
            story.append(Paragraph("Data Summary", styles['Heading2']))
            story.append(Paragraph(f"Dataset shape: {processor.data.shape[0]} rows × {processor.data.shape[1]} columns", styles['Normal']))
            story.append(Spacer(1, 12))
        
        # Estimates table
        if hasattr(processor, 'estimates') and processor.estimates:
            story.append(Paragraph("Statistical Estimates", styles['Heading2']))
            
            # Create estimates table
            table_data = [['Variable', 'Unweighted Mean', 'Weighted Mean', 'Margin of Error']]
            for var, est in processor.estimates.items():
                table_data.append([
                    var,
                    f"{est['unweighted']['mean']:.4f}",
                    f"{est['weighted']['mean']:.4f}",
                    f"±{est['weighted']['moe']:.4f}"
                ])
            
            table = Table(table_data)
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 14),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(table)
            story.append(Spacer(1, 20))
        
        # Build PDF
        doc.build(story)
        
        return send_file("report.pdf", as_attachment=True, download_name="survey_analysis_report.pdf")
    
    except Exception as e:
        return jsonify({'error': f'Error generating report: {str(e)}'}), 500

@app.route('/get_columns')
def get_columns():
    if processor.data is not None:
        return jsonify({
            'columns': processor.data.columns.tolist(),
            'numeric_columns': processor.data.select_dtypes(include=[np.number]).columns.tolist()
        })
    return jsonify({'columns': [], 'numeric_columns': []})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) 