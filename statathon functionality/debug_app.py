from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import numpy as np
import os

app = Flask(__name__)

# Simple data processor for debugging
class SimpleDataProcessor:
    def __init__(self):
        self.data = None
    
    def load_data(self, file_path):
        if file_path.endswith('.csv'):
            self.data = pd.read_csv(file_path)
        else:
            self.data = pd.read_excel(file_path)
        return self.data
    
    def get_basic_stats(self):
        if self.data is None:
            return None
        
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        stats = {}
        
        for col in numeric_cols:
            stats[col] = {
                'mean': float(self.data[col].mean()),
                'std': float(self.data[col].std()),
                'count': int(self.data[col].count())
            }
        
        return stats

processor = SimpleDataProcessor()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/test')
def test():
    return render_template('simple_test.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    print("Upload endpoint called")
    
    if 'file' not in request.files:
        print("No file in request")
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        print("No filename")
        return jsonify({'error': 'No file selected'}), 400
    
    print(f"Processing file: {file.filename}")
    
    try:
        # Save file temporarily
        filepath = f"temp_{file.filename}"
        file.save(filepath)
        
        # Load data
        data = processor.load_data(filepath)
        
        # Get basic info
        data_info = {
            'shape': data.shape,
            'columns': data.columns.tolist(),
            'dtypes': data.dtypes.astype(str).to_dict(),
            'sample_data': data.head(3).to_dict('records')
        }
        
        # Clean up
        os.remove(filepath)
        
        print(f"File processed successfully: {data_info['shape']}")
        
        return jsonify({
            'success': True,
            'message': 'File uploaded successfully',
            'data_info': data_info
        })
    
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        return jsonify({'error': f'Error processing file: {str(e)}'}), 500

@app.route('/analyze', methods=['POST'])
def analyze_data():
    print("Analyze endpoint called")
    
    try:
        config = request.json
        print(f"Config received: {config}")
        
        # Get basic stats
        stats = processor.get_basic_stats()
        
        if stats:
            return jsonify({
                'success': True,
                'estimates': stats,
                'missing_info': {'total_missing': 0},
                'outliers_info': {},
                'cleaned_shape': processor.data.shape
            })
        else:
            return jsonify({'error': 'No data available for analysis'}), 400
    
    except Exception as e:
        print(f"Error in analysis: {str(e)}")
        return jsonify({'error': f'Error in analysis: {str(e)}'}), 500

@app.route('/get_columns')
def get_columns():
    print("Get columns endpoint called")
    
    if processor.data is not None:
        return jsonify({
            'columns': processor.data.columns.tolist(),
            'numeric_columns': processor.data.select_dtypes(include=[np.number]).columns.tolist()
        })
    return jsonify({'columns': [], 'numeric_columns': []})

if __name__ == '__main__':
    print("Starting debug Flask app...")
    app.run(debug=True, host='0.0.0.0', port=5001) 