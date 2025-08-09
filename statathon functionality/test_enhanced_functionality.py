#!/usr/bin/env python3
"""
Comprehensive test script for the AI Enhanced Survey Analysis Application
Tests all major functionality including data processing, cleaning, weighting, and reporting.
"""

import requests
import json
import pandas as pd
import numpy as np
import os
import time
from datetime import datetime
import sys

# Configuration
API_BASE_URL = "http://localhost:5000"
TEST_DATA_PATH = "sample_data.csv"

class TestSurveyAnalysis:
    def __init__(self, base_url=API_BASE_URL):
        self.base_url = base_url
        self.session = requests.Session()
        self.test_results = []
        
    def log_test(self, test_name, success, message="", response_data=None):
        """Log test results"""
        result = {
            'test_name': test_name,
            'success': success,
            'message': message,
            'timestamp': datetime.now().isoformat(),
            'response_data': response_data
        }
        self.test_results.append(result)
        
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}: {message}")
        
        if not success and response_data:
            print(f"   Response: {response_data}")
    
    def create_test_data(self):
        """Create comprehensive test dataset"""
        np.random.seed(42)
        n_samples = 1000
        
        # Create realistic survey data
        data = {
            'respondent_id': range(1, n_samples + 1),
            'age': np.random.normal(35, 12, n_samples).clip(18, 80),
            'income': np.random.lognormal(10, 0.5, n_samples),
            'education_level': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_samples, p=[0.4, 0.35, 0.2, 0.05]),
            'satisfaction_score': np.random.normal(7, 2, n_samples).clip(1, 10),
            'region': np.random.choice(['North', 'South', 'East', 'West'], n_samples),
            'employment_status': np.random.choice(['Employed', 'Unemployed', 'Student', 'Retired'], n_samples, p=[0.6, 0.1, 0.15, 0.15]),
            'survey_weight': np.random.uniform(0.5, 2.0, n_samples)
        }
        
        # Introduce missing values intentionally
        missing_indices = np.random.choice(n_samples, size=int(n_samples * 0.1), replace=False)
        data['income'][missing_indices] = np.nan
        
        # Introduce outliers
        outlier_indices = np.random.choice(n_samples, size=int(n_samples * 0.05), replace=False)
        data['satisfaction_score'][outlier_indices] = np.random.choice([-5, 15], size=len(outlier_indices))
        
        # Create DataFrame and save
        df = pd.DataFrame(data)
        df.to_csv(TEST_DATA_PATH, index=False)
        
        print(f"✅ Created test dataset with {n_samples} samples: {TEST_DATA_PATH}")
        return df
    
    def test_health_check(self):
        """Test API health check"""
        try:
            response = self.session.get(f"{self.base_url}/health")
            success = response.status_code == 200
            
            if success:
                data = response.json()
                message = f"API is healthy. Status: {data.get('status')}"
            else:
                message = f"Health check failed with status {response.status_code}"
            
            self.log_test("Health Check", success, message, response.json() if success else None)
            return success
            
        except Exception as e:
            self.log_test("Health Check", False, f"Exception: {str(e)}")
            return False
    
    def test_file_upload(self):
        """Test file upload functionality"""
        try:
            if not os.path.exists(TEST_DATA_PATH):
                self.create_test_data()
            
            with open(TEST_DATA_PATH, 'rb') as f:
                files = {'file': (TEST_DATA_PATH, f, 'text/csv')}
                response = self.session.post(f"{self.base_url}/upload", files=files)
            
            success = response.status_code == 200
            
            if success:
                data = response.json()
                profile = data.get('data_profile', {})
                basic_info = profile.get('basic_info', {})
                shape = basic_info.get('shape', [0, 0])
                message = f"Uploaded {shape[0]} rows × {shape[1]} columns"
            else:
                message = f"Upload failed with status {response.status_code}"
            
            self.log_test("File Upload", success, message, response.json() if success else None)
            return success
            
        except Exception as e:
            self.log_test("File Upload", False, f"Exception: {str(e)}")
            return False
    
    def test_data_profiling(self):
        """Test data profiling functionality"""
        try:
            response = self.session.get(f"{self.base_url}/profile")
            success = response.status_code == 200
            
            if success:
                data = response.json()
                profile = data.get('profile', {})
                
                # Check if all expected profile sections are present
                expected_sections = ['basic_info', 'missing_analysis', 'data_quality', 'statistical_summary']
                missing_sections = [section for section in expected_sections if section not in profile]
                
                if missing_sections:
                    success = False
                    message = f"Missing profile sections: {missing_sections}"
                else:
                    basic_info = profile['basic_info']
                    missing_info = profile['missing_analysis']
                    message = f"Profiled data: {basic_info['shape'][0]} rows, {missing_info['total_missing']} missing values"
            else:
                message = f"Profiling failed with status {response.status_code}"
            
            self.log_test("Data Profiling", success, message, response.json() if success else None)
            return success
            
        except Exception as e:
            self.log_test("Data Profiling", False, f"Exception: {str(e)}")
            return False
    
    def test_data_cleaning(self):
        """Test comprehensive data cleaning functionality"""
        try:
            # Test missing value imputation
            cleaning_config = {
                'imputation': {
                    'method': 'auto',
                    'columns': ['income', 'age']
                },
                'outliers': {
                    'detection_methods': ['iqr', 'zscore'],
                    'handling_method': 'winsorize',
                    'columns': ['satisfaction_score'],
                    'kwargs': {'lower_percentile': 5, 'upper_percentile': 95}
                }
            }
            
            response = self.session.post(
                f"{self.base_url}/clean",
                json=cleaning_config,
                headers={'Content-Type': 'application/json'}
            )
            
            success = response.status_code == 200
            
            if success:
                data = response.json()
                cleaning_results = data.get('cleaning_results', {})
                
                # Check if imputation was performed
                imputation_results = cleaning_results.get('imputation', {})
                outlier_results = cleaning_results.get('outlier_detection', {})
                
                message = f"Cleaned data - Imputation: {len(imputation_results)} vars, Outliers detected: {len(outlier_results)} vars"
            else:
                message = f"Cleaning failed with status {response.status_code}"
            
            self.log_test("Data Cleaning", success, message, response.json() if success else None)
            return success
            
        except Exception as e:
            self.log_test("Data Cleaning", False, f"Exception: {str(e)}")
            return False
    
    def test_validation_rules(self):
        """Test custom validation rules"""
        try:
            # Add validation rules
            validation_config = {
                'rules': [
                    {
                        'type': 'range',
                        'column': 'age',
                        'condition': 'between',
                        'kwargs': {'min_val': 18, 'max_val': 100}
                    },
                    {
                        'type': 'categorical',
                        'column': 'education_level',
                        'condition': 'in_list',
                        'kwargs': {'allowed_values': ['High School', 'Bachelor', 'Master', 'PhD']}
                    }
                ]
            }
            
            response = self.session.post(
                f"{self.base_url}/validate",
                json=validation_config,
                headers={'Content-Type': 'application/json'}
            )
            
            success = response.status_code == 200
            
            if success:
                data = response.json()
                added_rules = data.get('added_rules', [])
                message = f"Added {len(added_rules)} validation rules"
            else:
                message = f"Validation setup failed with status {response.status_code}"
            
            self.log_test("Validation Rules", success, message, response.json() if success else None)
            return success
            
        except Exception as e:
            self.log_test("Validation Rules", False, f"Exception: {str(e)}")
            return False
    
    def test_weighted_analysis(self):
        """Test weighted statistical analysis"""
        try:
            analysis_config = {
                'weights': {
                    'column': 'survey_weight',
                    'method': 'design',
                    'kwargs': {'normalize_weights': True}
                },
                'estimation': {
                    'variables': ['age', 'income', 'satisfaction_score'],
                    'confidence_level': 0.95
                }
            }
            
            response = self.session.post(
                f"{self.base_url}/analyze",
                json=analysis_config,
                headers={'Content-Type': 'application/json'}
            )
            
            success = response.status_code == 200
            
            if success:
                data = response.json()
                estimates = data.get('estimates', {})
                summary = data.get('summary', {})
                
                # Check if estimates contain required fields
                required_fields = ['unweighted', 'weighted']
                valid_estimates = all(
                    all(field in est for field in required_fields)
                    for est in estimates.values()
                )
                
                if not valid_estimates:
                    success = False
                    message = "Estimates missing required fields"
                else:
                    message = f"Analyzed {summary.get('variables_analyzed', 0)} variables with {summary.get('total_observations', 0)} observations"
            else:
                message = f"Analysis failed with status {response.status_code}"
            
            self.log_test("Weighted Analysis", success, message, response.json() if success else None)
            return success
            
        except Exception as e:
            self.log_test("Weighted Analysis", False, f"Exception: {str(e)}")
            return False
    
    def test_report_generation(self):
        """Test PDF and HTML report generation"""
        try:
            report_config = {
                'format': 'both',
                'title': 'Test Survey Analysis Report',
                'author': 'Automated Test System'
            }
            
            response = self.session.post(
                f"{self.base_url}/report",
                json=report_config,
                headers={'Content-Type': 'application/json'}
            )
            
            success = response.status_code == 200
            
            if success:
                data = response.json()
                reports = data.get('reports', [])
                download_urls = data.get('download_urls', [])
                
                message = f"Generated {len(reports)} reports: {[r['type'] for r in reports]}"
            else:
                message = f"Report generation failed with status {response.status_code}"
            
            self.log_test("Report Generation", success, message, response.json() if success else None)
            return success
            
        except Exception as e:
            self.log_test("Report Generation", False, f"Exception: {str(e)}")
            return False
    
    def test_data_export(self):
        """Test data export functionality"""
        try:
            export_config = {
                'format': 'excel'
            }
            
            response = self.session.post(
                f"{self.base_url}/export",
                json=export_config,
                headers={'Content-Type': 'application/json'}
            )
            
            success = response.status_code == 200
            
            if success:
                data = response.json()
                export_path = data.get('export_path')
                download_url = data.get('download_url')
                
                message = f"Exported data to {os.path.basename(export_path)}"
            else:
                message = f"Export failed with status {response.status_code}"
            
            self.log_test("Data Export", success, message, response.json() if success else None)
            return success
            
        except Exception as e:
            self.log_test("Data Export", False, f"Exception: {str(e)}")
            return False
    
    def test_columns_endpoint(self):
        """Test columns information endpoint"""
        try:
            response = self.session.get(f"{self.base_url}/columns")
            success = response.status_code == 200
            
            if success:
                data = response.json()
                columns = data.get('columns', [])
                numeric_columns = data.get('numeric_columns', [])
                categorical_columns = data.get('categorical_columns', [])
                
                message = f"Retrieved {len(columns)} columns ({len(numeric_columns)} numeric, {len(categorical_columns)} categorical)"
            else:
                message = f"Columns endpoint failed with status {response.status_code}"
            
            self.log_test("Columns Endpoint", success, message, response.json() if success else None)
            return success
            
        except Exception as e:
            self.log_test("Columns Endpoint", False, f"Exception: {str(e)}")
            return False
    
    def run_full_test_suite(self):
        """Run complete test suite"""
        print("🚀 Starting AI Enhanced Survey Analysis - Full Test Suite")
        print("=" * 70)
        
        # Create test data
        self.create_test_data()
        
        # Test sequence (order matters for stateful operations)
        tests = [
            self.test_health_check,
            self.test_file_upload,
            self.test_data_profiling,
            self.test_columns_endpoint,
            self.test_validation_rules,
            self.test_data_cleaning,
            self.test_weighted_analysis,
            self.test_report_generation,
            self.test_data_export
        ]
        
        start_time = time.time()
        passed_tests = 0
        
        for test in tests:
            if test():
                passed_tests += 1
            time.sleep(1)  # Brief pause between tests
        
        # Generate test summary
        end_time = time.time()
        total_time = end_time - start_time
        
        print("\n" + "=" * 70)
        print("📊 TEST SUMMARY")
        print("=" * 70)
        print(f"Total Tests: {len(tests)}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {len(tests) - passed_tests}")
        print(f"Success Rate: {(passed_tests / len(tests)) * 100:.1f}%")
        print(f"Total Time: {total_time:.2f} seconds")
        
        if passed_tests == len(tests):
            print("🎉 ALL TESTS PASSED! The system is fully functional.")
        else:
            print("⚠️  Some tests failed. Check the logs above for details.")
        
        return passed_tests == len(tests)
    
    def generate_test_report(self):
        """Generate detailed test report"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = f"test_report_{timestamp}.json"
        
        test_summary = {
            'timestamp': datetime.now().isoformat(),
            'total_tests': len(self.test_results),
            'passed_tests': sum(1 for r in self.test_results if r['success']),
            'failed_tests': sum(1 for r in self.test_results if not r['success']),
            'test_results': self.test_results
        }
        
        with open(report_path, 'w') as f:
            json.dump(test_summary, f, indent=2)
        
        print(f"📄 Test report saved to: {report_path}")
        return report_path

def main():
    """Main test execution"""
    if len(sys.argv) > 1:
        base_url = sys.argv[1]
    else:
        base_url = API_BASE_URL
    
    print(f"Testing API at: {base_url}")
    
    tester = TestSurveyAnalysis(base_url)
    
    try:
        # Run full test suite
        success = tester.run_full_test_suite()
        
        # Generate test report
        tester.generate_test_report()
        
        # Exit with appropriate code
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print("\n⚠️  Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Test suite failed with exception: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
