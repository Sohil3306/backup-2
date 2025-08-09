import jsPDF from 'jspdf';
import 'jspdf-autotable';

export class ReportGenerator {
  constructor() {
    this.doc = null;
    this.yPosition = 20;
    this.margin = 20;
    this.pageWidth = 210;
    this.contentWidth = this.pageWidth - (2 * this.margin);
  }

  // Initialize PDF document
  initDocument(title = 'Survey Data Analysis Report') {
    this.doc = new jsPDF();
    this.yPosition = 20;
    
    // Add title
    this.doc.setFontSize(20);
    this.doc.setFont('helvetica', 'bold');
    this.doc.text(title, this.pageWidth / 2, this.yPosition, { align: 'center' });
    this.yPosition += 15;
    
    // Add timestamp
    this.doc.setFontSize(10);
    this.doc.setFont('helvetica', 'normal');
    const timestamp = new Date().toLocaleString();
    this.doc.text(`Generated on: ${timestamp}`, this.pageWidth / 2, this.yPosition, { align: 'center' });
    this.yPosition += 20;
    
    return this.doc;
  }

  // Add section header
  addSectionHeader(text) {
    this.checkPageBreak(15);
    
    this.doc.setFontSize(14);
    this.doc.setFont('helvetica', 'bold');
    this.doc.text(text, this.margin, this.yPosition);
    this.yPosition += 10;
    
    return this;
  }

  // Add paragraph text
  addParagraph(text) {
    this.checkPageBreak(10);
    
    this.doc.setFontSize(10);
    this.doc.setFont('helvetica', 'normal');
    
    const lines = this.doc.splitTextToSize(text, this.contentWidth);
    this.doc.text(lines, this.margin, this.yPosition);
    this.yPosition += (lines.length * 5) + 5;
    
    return this;
  }

  // Add data summary table
  addDataSummary(dataInfo) {
    this.checkPageBreak(30);
    
    this.addSectionHeader('Data Summary');
    
    const summaryData = [
      ['Dataset Shape', `${dataInfo.shape[0]} rows × ${dataInfo.shape[1]} columns`],
      ['Total Variables', dataInfo.columns.length.toString()],
      ['Numeric Variables', dataInfo.numericColumns.length.toString()],
      ['Categorical Variables', (dataInfo.columns.length - dataInfo.numericColumns.length).toString()]
    ];
    
    this.doc.autoTable({
      startY: this.yPosition,
      head: [['Metric', 'Value']],
      body: summaryData,
      theme: 'grid',
      headStyles: { fillColor: [52, 73, 94], textColor: 255 },
      styles: { fontSize: 10 }
    });
    
    this.yPosition = this.doc.lastAutoTable.finalY + 10;
    
    return this;
  }

  // Add missing values table
  addMissingValuesTable(missingInfo) {
    if (!missingInfo || missingInfo.totalMissing === 0) {
      this.addParagraph('No missing values detected in the dataset.');
      return this;
    }
    
    this.checkPageBreak(30);
    
    this.addSectionHeader('Missing Values Analysis');
    
    const missingData = Object.entries(missingInfo.missingPerColumn)
      .filter(([col, count]) => count > 0)
      .map(([col, count]) => [
        col,
        count.toString(),
        `${missingInfo.missingPercentage[col].toFixed(2)}%`
      ]);
    
    if (missingData.length > 0) {
      this.doc.autoTable({
        startY: this.yPosition,
        head: [['Variable', 'Missing Count', 'Missing %']],
        body: missingData,
        theme: 'grid',
        headStyles: { fillColor: [231, 76, 60], textColor: 255 },
        styles: { fontSize: 9 }
      });
      
      this.yPosition = this.doc.lastAutoTable.finalY + 10;
    }
    
    return this;
  }

  // Add outliers table
  addOutliersTable(outliersInfo) {
    if (!outliersInfo) {
      return this;
    }
    
    const outliersData = Object.entries(outliersInfo)
      .filter(([col, info]) => info.count > 0)
      .map(([col, info]) => [
        col,
        info.count.toString(),
        `${info.percentage.toFixed(2)}%`
      ]);
    
    if (outliersData.length > 0) {
      this.checkPageBreak(30);
      
      this.addSectionHeader('Outlier Detection Results');
      
      this.doc.autoTable({
        startY: this.yPosition,
        head: [['Variable', 'Outlier Count', 'Outlier %']],
        body: outliersData,
        theme: 'grid',
        headStyles: { fillColor: [243, 156, 18], textColor: 255 },
        styles: { fontSize: 9 }
      });
      
      this.yPosition = this.doc.lastAutoTable.finalY + 10;
    }
    
    return this;
  }

  // Add statistical estimates table
  addEstimatesTable(estimates) {
    if (!estimates || Object.keys(estimates).length === 0) {
      return this;
    }
    
    this.checkPageBreak(40);
    
    this.addSectionHeader('Statistical Estimates');
    
    const estimatesData = Object.entries(estimates).map(([variable, est]) => [
      variable,
      est.unweighted.mean.toFixed(4),
      est.weighted.mean.toFixed(4),
      `±${est.weighted.moe.toFixed(4)}`,
      est.weighted.se.toFixed(4)
    ]);
    
    this.doc.autoTable({
      startY: this.yPosition,
      head: [['Variable', 'Unweighted Mean', 'Weighted Mean', 'Margin of Error', 'Standard Error']],
      body: estimatesData,
      theme: 'grid',
      headStyles: { fillColor: [46, 204, 113], textColor: 255 },
      styles: { fontSize: 8 }
    });
    
    this.yPosition = this.doc.lastAutoTable.finalY + 10;
    
    return this;
  }

  // Add summary statistics table
  addSummaryStatisticsTable(summaryStats) {
    if (!summaryStats || Object.keys(summaryStats).length === 0) {
      return this;
    }
    
    this.checkPageBreak(40);
    
    this.addSectionHeader('Summary Statistics');
    
    const summaryData = Object.entries(summaryStats).map(([variable, stats]) => [
      variable,
      stats.count.toString(),
      stats.mean.toFixed(2),
      stats.median.toFixed(2),
      stats.std.toFixed(2),
      `${stats.min.toFixed(2)} - ${stats.max.toFixed(2)}`
    ]);
    
    this.doc.autoTable({
      startY: this.yPosition,
      head: [['Variable', 'Count', 'Mean', 'Median', 'Std Dev', 'Range']],
      body: summaryData,
      theme: 'grid',
      headStyles: { fillColor: [52, 152, 219], textColor: 255 },
      styles: { fontSize: 8 }
    });
    
    this.yPosition = this.doc.lastAutoTable.finalY + 10;
    
    return this;
  }

  // Add correlation matrix
  addCorrelationMatrix(correlationMatrix) {
    if (!correlationMatrix || Object.keys(correlationMatrix).length === 0) {
      return this;
    }
    
    this.checkPageBreak(50);
    
    this.addSectionHeader('Correlation Matrix');
    
    const variables = Object.keys(correlationMatrix);
    const correlationData = variables.map(var1 => {
      const row = [var1];
      variables.forEach(var2 => {
        const correlation = correlationMatrix[var1][var2];
        row.push(correlation.toFixed(3));
      });
      return row;
    });
    
    this.doc.autoTable({
      startY: this.yPosition,
      head: ['Variable', ...variables],
      body: correlationData,
      theme: 'grid',
      headStyles: { fillColor: [155, 89, 182], textColor: 255 },
      styles: { fontSize: 7 }
    });
    
    this.yPosition = this.doc.lastAutoTable.finalY + 10;
    
    return this;
  }

  // Add methodology section
  addMethodology(config) {
    this.checkPageBreak(30);
    
    this.addSectionHeader('Methodology');
    
    let methodologyText = 'This analysis was performed using the following methods:\n\n';
    
    if (config.imputationMethod) {
      methodologyText += `• Missing Value Imputation: ${config.imputationMethod.toUpperCase()}\n`;
    }
    
    if (config.outlierMethod) {
      methodologyText += `• Outlier Detection: ${config.outlierMethod.toUpperCase()}\n`;
      methodologyText += `• Outlier Handling: ${config.outlierHandling}\n`;
    }
    
    if (config.weightColumn) {
      methodologyText += `• Survey Weights: Applied using column "${config.weightColumn}"\n`;
    } else {
      methodologyText += `• Survey Weights: Equal weights applied\n`;
    }
    
    methodologyText += '\nStatistical calculations include:\n';
    methodologyText += '• Mean, standard deviation, and standard error\n';
    methodologyText += '• Margin of error (95% confidence interval)\n';
    methodologyText += '• Weighted and unweighted estimates\n';
    methodologyText += '• Correlation analysis for numeric variables';
    
    this.addParagraph(methodologyText);
    
    return this;
  }

  // Add conclusions section
  addConclusions(estimates, missingInfo, outliersInfo) {
    this.checkPageBreak(30);
    
    this.addSectionHeader('Conclusions');
    
    let conclusionsText = 'Based on the analysis:\n\n';
    
    if (missingInfo && missingInfo.totalMissing > 0) {
      conclusionsText += `• ${missingInfo.totalMissing} missing values were identified and handled\n`;
    }
    
    if (outliersInfo) {
      const totalOutliers = Object.values(outliersInfo).reduce((sum, info) => sum + info.count, 0);
      if (totalOutliers > 0) {
        conclusionsText += `• ${totalOutliers} outliers were detected and processed\n`;
      }
    }
    
    if (estimates && Object.keys(estimates).length > 0) {
      conclusionsText += `• Statistical estimates were calculated for ${Object.keys(estimates).length} variables\n`;
    }
    
    conclusionsText += '\nThe analysis provides reliable estimates with appropriate margins of error for survey data interpretation.';
    
    this.addParagraph(conclusionsText);
    
    return this;
  }

  // Check if we need a page break
  checkPageBreak(requiredSpace) {
    if (this.yPosition + requiredSpace > 280) {
      this.doc.addPage();
      this.yPosition = 20;
    }
  }

  // Generate and download the complete report
  generateReport(dataInfo, missingInfo, outliersInfo, estimates, summaryStats, correlationMatrix, config) {
    this.initDocument('Survey Data Analysis Report');
    
    this.addDataSummary(dataInfo);
    this.addMissingValuesTable(missingInfo);
    this.addOutliersTable(outliersInfo);
    this.addEstimatesTable(estimates);
    this.addSummaryStatisticsTable(summaryStats);
    this.addCorrelationMatrix(correlationMatrix);
    this.addMethodology(config);
    this.addConclusions(estimates, missingInfo, outliersInfo);
    
    // Save the PDF
    const filename = `survey_analysis_report_${new Date().toISOString().split('T')[0]}.pdf`;
    this.doc.save(filename);
    
    return filename;
  }

  // Generate a simple report (for testing)
  generateSimpleReport(dataInfo, estimates) {
    this.initDocument('Simple Survey Analysis Report');
    
    this.addDataSummary(dataInfo);
    this.addEstimatesTable(estimates);
    
    const filename = `simple_survey_report_${new Date().toISOString().split('T')[0]}.pdf`;
    this.doc.save(filename);
    
    return filename;
  }
} 