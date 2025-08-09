# AI-Enhanced Survey Analysis Application

## 🚀 MoSPI Hackathon Solution

A comprehensive, AI-enhanced application for automated data preparation, estimation, and report writing for official statistical agencies.

## ✨ Features

### 🔧 Core Functionality
- **Multi-Format Data Upload**: Support for CSV and Excel files (.xlsx, .xls)
- **AI-Enhanced Data Cleaning**: 
  - Missing value imputation (Mean, Median, K-Nearest Neighbors)
  - Outlier detection (IQR, Z-Score methods)
  - Intelligent data validation
- **Survey Weight Application**: Configurable weight columns for accurate population estimates
- **Statistical Analysis**: 
  - Weighted and unweighted means
  - Margins of error calculation
  - Confidence intervals
  - Standard errors
- **Professional Report Generation**: PDF reports with methodology and visualizations

### 🎨 Advanced UI/UX
- **Modern Design**: Tailwind CSS with custom design system
- **Responsive Layout**: Works on desktop, tablet, and mobile
- **Interactive Workflow**: Step-by-step guided process
- **Real-time Analytics**: Live performance metrics and progress tracking
- **Professional Animations**: Smooth transitions and loading states

### 📊 Analytics Dashboard
- **Performance Metrics**: Files processed, records analyzed, processing time
- **Success Tracking**: Real-time success rates and error handling
- **Progress Visualization**: Visual workflow progress with completion status

## 🛠️ Technology Stack

- **Frontend**: React 18 with modern hooks
- **Styling**: Tailwind CSS with custom components
- **Data Processing**: MathJS for statistical calculations
- **File Handling**: PapaParse (CSV), SheetJS (Excel)
- **PDF Generation**: jsPDF with AutoTable
- **Animations**: CSS animations and transitions
- **Icons**: Emoji-based icons for universal compatibility

## 📦 Installation

### Prerequisites
- Node.js (v16 or higher)
- npm (comes with Node.js)

### Setup Instructions

1. **Clone or download the project**
   ```bash
   # If you have git
   git clone <repository-url>
   cd ai-enhanced-survey-analysis
   
   # Or simply extract the downloaded files
   ```

2. **Install dependencies**
   ```bash
   npm install
   ```

3. **Start the development server**
   ```bash
   npm start
   ```

4. **Open your browser**
   Navigate to `http://localhost:3000`

### Alternative: Using the Batch File
If you're on Windows, you can simply double-click `start_app.bat` to automatically install dependencies and start the application.

## 🎯 Usage Guide

### Step 1: Data Upload
1. Click "Choose File" or drag and drop your CSV/Excel file
2. Supported formats: `.csv`, `.xlsx`, `.xls`
3. Maximum file size: 50MB
4. The system will automatically detect data structure and validate format

### Step 2: Data Cleaning Configuration
1. **Missing Value Imputation**:
   - Select method: Mean, Median, or K-Nearest Neighbors
   - Choose columns to apply imputation (or leave empty for all numeric)
   
2. **Outlier Detection**:
   - Select detection method: IQR or Z-Score
   - Choose handling strategy: Winsorize or Remove
   - Select columns to check for outliers

### Step 3: Weight Application
1. **Weight Column**: Select a column containing survey weights (optional)
2. **Estimation Variables**: Choose variables to estimate (or leave empty for all numeric)
3. The system will use equal weights (1.0) if no weight column is specified

### Step 4: Analysis & Results
1. Click "Run AI-Enhanced Analysis" to start processing
2. Review statistical estimates with margins of error
3. Generate professional PDF report
4. Export results for official use

## 📊 Output Features

### Statistical Estimates
- **Unweighted Statistics**: Raw sample statistics
- **Weighted Statistics**: Population estimates using survey weights
- **Margins of Error**: 95% confidence intervals
- **Standard Errors**: Precision measures

### Professional Reports
- **Executive Summary**: Key findings and methodology
- **Data Quality Assessment**: Missing values and outlier analysis
- **Statistical Tables**: Formatted results with proper notation
- **Methodology Section**: Detailed explanation of methods used

## 🎨 UI Components

### Navigation
- **Header**: App branding, navigation tabs, and analytics summary
- **Sidebar**: Workflow progress, quick actions, and system status
- **Progress Tracker**: Visual step-by-step workflow with completion status

### Views
- **Workflow**: Main step-by-step analysis process
- **Dashboard**: Analytics overview and performance metrics
- **Results**: Detailed analysis results and report generation

## 🔧 Configuration

### Tailwind CSS Customization
The application uses a custom Tailwind configuration with:
- Extended color palette for professional appearance
- Custom animations and transitions
- Responsive design utilities
- Form styling enhancements

### Component Architecture
- **Modular Design**: Each step is a separate component
- **State Management**: React hooks for local state
- **Props Interface**: Clean component communication
- **Error Handling**: Comprehensive error states and user feedback

## 📈 Performance Features

### Optimization
- **Lazy Loading**: Components load as needed
- **Efficient Processing**: Optimized statistical calculations
- **Memory Management**: Proper cleanup and resource handling
- **Responsive Design**: Fast loading on all devices

### Analytics
- **Processing Time Tracking**: Real-time performance monitoring
- **Success Rate Monitoring**: Error tracking and reporting
- **User Activity Logging**: Workflow completion tracking

## 🚀 Deployment

### Development
```bash
npm start
```

### Production Build
```bash
npm run build
```

### Static Hosting
The built application can be deployed to any static hosting service:
- Netlify
- Vercel
- GitHub Pages
- AWS S3
- Any web server

## 📋 Requirements Met

### MoSPI Hackathon Requirements ✅
- ✅ **Data Input & Configuration**: CSV/Excel upload with schema mapping
- ✅ **Cleaning Modules**: Missing-value imputation, outlier detection, rule-based validation
- ✅ **Weight Application**: Survey weights with margins of error
- ✅ **Report Generation**: Professional PDF reports with templates
- ✅ **User Guidance**: Tooltips, explanations, error-checking alerts
- ✅ **Bonus Features**: Dashboard, audit trails, advanced visualizations

### Official Statistical Agency Standards ✅
- ✅ **Methodological Rigor**: Proper statistical methodology
- ✅ **Data Quality**: Comprehensive validation and cleaning
- ✅ **Professional Output**: Official report standards
- ✅ **Reproducibility**: Transparent methodology and processes
- ✅ **Accessibility**: User-friendly interface for non-technical users

## 🤝 Contributing

This application was developed for the MoSPI Hackathon 2024. For questions or contributions, please refer to the hackathon guidelines.

## 📄 License

Developed for MoSPI Hackathon 2024 - Official Statistical Agencies Solution.

---

**Built with ❤️ for improving official statistics through AI-enhanced automation** 