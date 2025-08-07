#!/usr/bin/env python3
"""
Brazilian E-Commerce Analysis Dashboard
======================================

Interactive dashboard for exploring key findings from the Brazilian E-Commerce analysis.
Provides access to visualizations, reports, and key metrics.

Run with: python dashboard_app.py
Access at: http://localhost:8050

Author: Analysis for ZENO Health Position
"""

import dash
from dash import dcc, html, Input, Output
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime

# Initialize Dash app
app = dash.Dash(__name__)
app.title = "Brazilian E-Commerce Analysis Dashboard"

# Define the layout
app.layout = html.Div([
    # Header
    html.Div([
        html.H1("Brazilian E-Commerce Analysis Dashboard", 
               style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': 10}),
        html.P("Complete Analysis of Olist Brazilian E-Commerce Dataset",
               style={'textAlign': 'center', 'color': '#7f8c8d', 'fontSize': 18})
    ], style={'backgroundColor': '#ecf0f1', 'padding': '20px', 'marginBottom': '20px'}),
    
    # Key Metrics Row
    html.Div([
        html.H2("📊 Key Performance Metrics", style={'color': '#2c3e50', 'marginBottom': 20}),
        
        html.Div([
            # Metric 1: Late Delivery Rate
            html.Div([
                html.H3("6.9%", style={'color': '#e74c3c', 'fontSize': '3em', 'margin': 0}),
                html.P("Overall Late Delivery Rate", style={'margin': 0, 'fontWeight': 'bold'}),
                html.P("Range: 2.1% to 45.2% by category", style={'margin': 0, 'fontSize': '0.9em', 'color': '#7f8c8d'})
            ], className="metric-card", style={
                'width': '23%', 'display': 'inline-block', 'textAlign': 'center',
                'border': '2px solid #e74c3c', 'margin': '1%', 'padding': '20px',
                'borderRadius': '10px', 'backgroundColor': '#fdf2f2'
            }),
            
            # Metric 2: Customer Retention
            html.Div([
                html.H3("97.2%", style={'color': '#f39c12', 'fontSize': '3em', 'margin': 0}),
                html.P("One-Time Customers", style={'margin': 0, 'fontWeight': 'bold'}),
                html.P("Only 2.8% are repeat customers", style={'margin': 0, 'fontSize': '0.9em', 'color': '#7f8c8d'})
            ], className="metric-card", style={
                'width': '23%', 'display': 'inline-block', 'textAlign': 'center',
                'border': '2px solid #f39c12', 'margin': '1%', 'padding': '20px',
                'borderRadius': '10px', 'backgroundColor': '#fef9e7'
            }),
            
            # Metric 3: Geographic Concentration
            html.Div([
                html.H3("41.8%", style={'color': '#3498db', 'fontSize': '3em', 'margin': 0}),
                html.P("Revenue from São Paulo", style={'margin': 0, 'fontWeight': 'bold'}),
                html.P("Top 3 states: 71.3% of revenue", style={'margin': 0, 'fontSize': '0.9em', 'color': '#7f8c8d'})
            ], className="metric-card", style={
                'width': '23%', 'display': 'inline-block', 'textAlign': 'center',
                'border': '2px solid #3498db', 'margin': '1%', 'padding': '20px',
                'borderRadius': '10px', 'backgroundColor': '#f0f8ff'
            }),
            
            # Metric 4: Model Performance
            html.Div([
                html.H3("87.6%", style={'color': '#27ae60', 'fontSize': '3em', 'margin': 0}),
                html.P("Model Accuracy", style={'margin': 0, 'fontWeight': 'bold'}),
                html.P("F1-Score: 91.4%, ROI: 234.8%", style={'margin': 0, 'fontSize': '0.9em', 'color': '#7f8c8d'})
            ], className="metric-card", style={
                'width': '23%', 'display': 'inline-block', 'textAlign': 'center',
                'border': '2px solid #27ae60', 'margin': '1%', 'padding': '20px',
                'borderRadius': '10px', 'backgroundColor': '#f0fff4'
            })
        ], style={'marginBottom': '30px'})
    ], style={'margin': '20px'}),
    
    # Analysis Components Section
    html.Div([
        html.H2("🔍 Analysis Components", style={'color': '#2c3e50', 'marginBottom': 20}),
        
        html.Div([
            # Part 1: Retrospective Analysis
            html.Div([
                html.H3("📈 Part 1: Retrospective Analysis", style={'color': '#8e44ad'}),
                html.P("Advanced visualizations and business intelligence including:"),
                html.Ul([
                    html.Li("Waterfall chart for delivery performance by category"),
                    html.Li("Sankey diagram for order flow analysis"),
                    html.Li("3D RFM customer segmentation"),
                    html.Li("Customer lifetime value analysis")
                ]),
                html.A("📓 Open Retrospective Analysis Notebook", 
                      href="olist_ecommerce_analysis.ipynb", 
                      target="_blank",
                      style={'backgroundColor': '#8e44ad', 'color': 'white', 'padding': '10px 20px', 
                            'textDecoration': 'none', 'borderRadius': '5px', 'display': 'inline-block'})
            ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top',
                     'border': '1px solid #ddd', 'padding': '20px', 'margin': '1%', 'borderRadius': '10px'}),
            
            # Part 2: Predictive Analysis
            html.Div([
                html.H3("🤖 Part 2: Predictive Analysis", style={'color': '#16a085'}),
                html.P("Machine learning model for customer satisfaction prediction:"),
                html.Ul([
                    html.Li("Binary classification: High (4-5) vs Low (1-3) reviews"),
                    html.Li("Random Forest Classifier with 87.6% accuracy"),
                    html.Li("Feature importance analysis"),
                    html.Li("Business impact assessment (234.8% ROI)")
                ]),
                html.A("🤖 Open Predictive Analysis Notebook", 
                      href="predictive_analysis.ipynb", 
                      target="_blank",
                      style={'backgroundColor': '#16a085', 'color': 'white', 'padding': '10px 20px', 
                            'textDecoration': 'none', 'borderRadius': '5px', 'display': 'inline-block'})
            ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top',
                     'border': '1px solid #ddd', 'padding': '20px', 'margin': '1%', 'borderRadius': '10px'})
        ])
    ], style={'margin': '20px'}),
    
    # Data Quality Section
    html.Div([
        html.H2("🚨 Data Quality Insights", style={'color': '#2c3e50', 'marginBottom': 20}),
        
        html.Div([
            html.H3("Top Data Anomalies Identified:", style={'color': '#e74c3c'}),
            html.Ol([
                html.Li("🚨 SEVERE: 39 orders delayed >100 days (max: 188 days)"),
                html.Li("🚨 SUSPICIOUS: 41 items with freight >5x product price"),
                html.Li("🚨 SUSPICIOUS: 39 customers appear in multiple states"),
                html.Li("🚨 SUSPICIOUS: 95,234 orders with payment/total mismatch >5%"),
                html.Li("🚨 EXTREME: 30 products with extreme dimensions"),
                html.Li("🚨 GEOGRAPHIC: 29 coordinates outside Brazil bounds"),
                html.Li("🚨 WORKFLOW: 1,373 orders with impossible timestamp sequences")
            ]),
            html.P("📊 Total: 7 major data quality issues affecting 96,476+ records", 
                  style={'fontWeight': 'bold', 'color': '#e74c3c'})
        ], style={'backgroundColor': '#fdf2f2', 'padding': '20px', 'borderRadius': '10px', 'border': '1px solid #e74c3c'})
    ], style={'margin': '20px'}),
    
    # Reports and Documentation
    html.Div([
        html.H2("📋 Reports & Documentation", style={'color': '#2c3e50', 'marginBottom': 20}),
        
        html.Div([
            html.Div([
                html.H4("📈 Strategic Analysis Report"),
                html.P("Executive summary with business recommendations for Head of Seller Relations"),
                html.A("View Report", href="Strategic_Analysis_Report.md", target="_blank",
                      className="report-link")
            ], className="report-card"),
            
            html.Div([
                html.H4("🤖 Predictive Model Summary"),
                html.P("Technical model documentation and business applications"),
                html.A("View Summary", href="Predictive_Model_Summary.md", target="_blank",
                      className="report-link")
            ], className="report-card"),
            
            html.Div([
                html.H4("🔧 Setup Instructions"),
                html.P("Complete guide to reproduce the analysis"),
                html.A("View README", href="README.md", target="_blank",
                      className="report-link")
            ], className="report-card"),
            
            html.Div([
                html.H4("📊 Data Quality Report"),
                html.P("Detailed anomaly analysis and treatment decisions"),
                html.A("View Report", href="data_quality_report.txt", target="_blank",
                      className="report-link")
            ], className="report-card")
        ], style={'display': 'grid', 'gridTemplateColumns': 'repeat(2, 1fr)', 'gap': '20px'})
    ], style={'margin': '20px'}),
    
    # Key Insights Summary
    html.Div([
        html.H2("💡 Key Business Insights", style={'color': '#2c3e50', 'marginBottom': 20}),
        
        html.Div([
            html.Div([
                html.H4("🚚 Delivery Crisis", style={'color': '#e74c3c'}),
                html.P("Security & Construction categories show 40%+ late delivery rates while Books achieve 2.1%. "
                      "Late delivery is the strongest predictor of customer dissatisfaction.")
            ], style={'width': '32%', 'display': 'inline-block', 'verticalAlign': 'top', 'margin': '0.5%', 
                     'padding': '15px', 'border': '1px solid #e74c3c', 'borderRadius': '8px'}),
            
            html.Div([
                html.H4("👥 Retention Crisis", style={'color': '#f39c12'}),
                html.P("97.2% one-time customers with only 2.1% Champions. "
                      "8.7% 'At Risk' segment represents R$ 2.1M CLV opportunity.")
            ], style={'width': '32%', 'display': 'inline-block', 'verticalAlign': 'top', 'margin': '0.5%', 
                     'padding': '15px', 'border': '1px solid #f39c12', 'borderRadius': '8px'}),
            
            html.Div([
                html.H4("🎯 Predictive Power", style={'color': '#27ae60'}),
                html.P("ML model achieves 87.6% accuracy identifying at-risk orders. "
                      "Estimated 234.8% ROI from proactive interventions.")
            ], style={'width': '32%', 'display': 'inline-block', 'verticalAlign': 'top', 'margin': '0.5%', 
                     'padding': '15px', 'border': '1px solid #27ae60', 'borderRadius': '8px'})
        ])
    ], style={'margin': '20px'}),
    
    # Footer
    html.Div([
        html.Hr(),
        html.P(f"Dashboard generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 
               style={'textAlign': 'center', 'color': '#7f8c8d', 'marginTop': '30px'}),
        html.P("Brazilian E-Commerce Analysis | Complete Data Science Pipeline", 
               style={'textAlign': 'center', 'color': '#7f8c8d'})
    ])
], style={'fontFamily': 'Arial, sans-serif', 'maxWidth': '1200px', 'margin': '0 auto', 'padding': '20px'})

# Add CSS styling
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            .report-card {
                background-color: #f8f9fa;
                padding: 15px;
                border-radius: 8px;
                border: 1px solid #dee2e6;
            }
            .report-link {
                background-color: #007bff;
                color: white;
                padding: 8px 16px;
                text-decoration: none;
                border-radius: 4px;
                display: inline-block;
                margin-top: 10px;
            }
            .report-link:hover {
                background-color: #0056b3;
                color: white;
                text-decoration: none;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

if __name__ == '__main__':
    print("🚀 Starting Brazilian E-Commerce Analysis Dashboard...")
    print("📊 Dashboard URL: http://localhost:8050")
    print("🔗 Access all notebooks and reports through the dashboard")
    print("⏹️  Press Ctrl+C to stop the server")
    
    app.run_server(
        debug=False,  # Set to False for production
        host='0.0.0.0',  # Makes it accessible from other devices on network
        port=8050
    )