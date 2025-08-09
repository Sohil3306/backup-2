#!/usr/bin/env python3
"""
Enhanced AI Survey Analysis Application Startup Script
This script starts the full-featured backend with all advanced functionality.
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def check_dependencies():
    """Check if all required dependencies are installed"""
    print("🔍 Checking dependencies...")
    
    required_packages = [
        'flask', 'pandas', 'numpy', 'scikit-learn', 'openpyxl', 
        'reportlab', 'matplotlib', 'seaborn', 'plotly', 'scipy',
        'statsmodels', 'missingno', 'xlsxwriter', 'flask-cors'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"  ✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"  ❌ {package}")
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("Installing missing packages...")
        
        try:
            subprocess.check_call([
                sys.executable, '-m', 'pip', 'install', '--upgrade'
            ] + missing_packages)
            print("✅ All packages installed successfully!")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install packages: {e}")
            return False
    
    return True

def setup_directories():
    """Create necessary directories"""
    print("📁 Setting up directories...")
    
    directories = ['uploads', 'reports', 'temp', 'logs']
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"  ✅ {directory}/")
    
    return True

def check_sample_data():
    """Check if sample data exists, create if not"""
    print("📊 Checking sample data...")
    
    sample_data_path = "sample_data.csv"
    
    if not os.path.exists(sample_data_path):
        print("  Creating sample dataset...")
        
        # Create sample survey data
        import pandas as pd
        import numpy as np
        
        np.random.seed(42)
        n_samples = 1000
        
        data = {
            'respondent_id': range(1, n_samples + 1),
            'age': np.random.normal(35, 12, n_samples).clip(18, 80),
            'income': np.random.lognormal(10, 0.5, n_samples),
            'education_level': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_samples, p=[0.4, 0.35, 0.2, 0.05]),
            'satisfaction_score': np.random.normal(7, 2, n_samples).clip(1, 10),
            'region': np.random.choice(['North', 'South', 'East', 'West'], n_samples),
            'employment_status': np.random.choice(['Employed', 'Unemployed', 'Student', 'Retired'], n_samples, p=[0.6, 0.1, 0.15, 0.15]),
            'survey_weight': np.random.uniform(0.5, 2.0, n_samples),
            'household_size': np.random.poisson(2.5, n_samples).clip(1, 8),
            'urban_rural': np.random.choice(['Urban', 'Rural'], n_samples, p=[0.7, 0.3])
        }
        
        # Introduce some missing values and outliers for testing
        missing_indices = np.random.choice(n_samples, size=int(n_samples * 0.05), replace=False)
        data['income'][missing_indices] = np.nan
        
        outlier_indices = np.random.choice(n_samples, size=int(n_samples * 0.02), replace=False)
        data['satisfaction_score'][outlier_indices] = np.random.choice([-2, 15], size=len(outlier_indices))
        
        df = pd.DataFrame(data)
        df.to_csv(sample_data_path, index=False)
        
        print(f"  ✅ Created sample dataset: {sample_data_path}")
    else:
        print(f"  ✅ Sample data exists: {sample_data_path}")
    
    return True

def start_application():
    """Start the Flask application"""
    print("🚀 Starting AI Enhanced Survey Analysis Application...")
    print("=" * 60)
    
    # Import and run the enhanced app
    try:
        from enhanced_app import app
        
        print("📡 Server starting on http://localhost:5000")
        print("📊 API Documentation available at http://localhost:5000")
        print("🔧 Use the React frontend to interact with the API")
        print("=" * 60)
        print("Press Ctrl+C to stop the server")
        
        app.run(
            debug=True,
            host='0.0.0.0',
            port=5000,
            use_reloader=False  # Avoid double startup in debug mode
        )
        
    except ImportError as e:
        print(f"❌ Failed to import enhanced app: {e}")
        print("Falling back to basic app...")
        
        try:
            from app import app
            app.run(debug=True, host='0.0.0.0', port=5000)
        except ImportError as e:
            print(f"❌ Failed to start any application: {e}")
            return False
    
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
        return True
    
    except Exception as e:
        print(f"❌ Application failed to start: {e}")
        return False

def main():
    """Main startup sequence"""
    print("🎯 AI Enhanced Survey Analysis Application")
    print("   Advanced Data Processing & Statistical Analysis")
    print("=" * 60)
    
    # Step 1: Check dependencies
    if not check_dependencies():
        print("❌ Dependency check failed. Please install required packages.")
        sys.exit(1)
    
    # Step 2: Setup directories
    if not setup_directories():
        print("❌ Directory setup failed.")
        sys.exit(1)
    
    # Step 3: Check sample data
    if not check_sample_data():
        print("❌ Sample data setup failed.")
        sys.exit(1)
    
    print("✅ All checks passed. Starting application...\n")
    
    # Step 4: Start application
    if not start_application():
        print("❌ Application startup failed.")
        sys.exit(1)

if __name__ == "__main__":
    main()
