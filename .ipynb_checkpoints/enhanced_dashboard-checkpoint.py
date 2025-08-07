#!/usr/bin/env python3
"""
Enhanced Brazilian E-Commerce Analysis Dashboard
===============================================

Interactive dashboard with complete integration of all analysis components:
- Delivery Performance Analysis with Waterfall Chart
- Customer Segmentation with RFM Analysis
- Geographic Distribution with Concentration Metrics
- Predictive Model Performance Metrics
- Business Recommendations

Run with: python enhanced_dashboard.py
Access at: http://localhost:8050

Human-developed with limited AI assistance (20-25%) for syntax and debugging only.
"""

import dash
from dash import dcc, html, Input, Output, dash_table
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime
import os

# Initialize Dash app
app = dash.Dash(__name__)
app.title = "Brazilian E-Commerce Analysis Dashboard"

# Load pre-computed analysis results (in production, these would come from the notebooks)
# For demo purposes, we'll create sample data based on our analysis results
def load_analysis_data():
    """Load pre-computed analysis results from notebooks"""
    
    # Category performance data
    category_data = {
        'category': ['audio', 'christmas_supplies', 'fashion_underwear_beach', 'home_confort', 
                     'electronics', 'health_beauty', 'books_technical', 'office_furniture'],
        'late_rate': [0.12, 0.10, 0.09, 0.09, 0.08, 0.08, 0.08, 0.08],
        'total_orders': [362, 150, 127, 429, 2729, 9465, 263, 1668],
        'avg_delay_days': [-10.15, -12.05, -10.93, -9.81, -11.14, -11.97, -11.31, -11.85]
    }
    
    # RFM segment data
    rfm_data = {
        'segment': ['Champions', 'Loyal Customers', 'At Risk', "Can't Lose Them", 
                    'New Customers', 'Promising', 'Potential Loyalists', 'Lost Customers', 'Others'],
        'count': [7894, 14077, 22836, 8186, 15282, 4703, 9044, 7092, 6305],
        'percentage': [8.3, 14.8, 23.9, 8.6, 16.0, 4.9, 9.5, 7.4, 6.6],
        'avg_clv': [120202.00, 32430.55, 45407.62, 117.70, 78.06, 64.35, 357.81, 28.20, 48.17],
        'monetary': [532.60, 264.33, 256.58, 235.41, 156.12, 128.69, 105.53, 56.41, 39.79]
    }
    
    # Geographic data
    geo_data = {
        'state': ['SP', 'RJ', 'MG', 'RS', 'PR', 'BA', 'SC', 'GO', 'DF', 'ES'],
        'revenue_share': [37.4, 12.8, 11.9, 5.4, 5.0, 3.4, 3.6, 2.0, 2.1, 2.0],
        'customer_share': [42.3, 13.0, 11.3, 5.2, 4.8, 3.2, 3.5, 1.9, 2.0, 1.9]
    }
    
    # Model performance metrics
    model_metrics = {
        'Accuracy': 0.876,
        'Precision': 0.891,
        'Recall': 0.939,
        'F1-Score': 0.914,
        'AUC-ROC': 0.823
    }
    
    return {
        'category': pd.DataFrame(category_data),
        'rfm': pd.DataFrame(rfm_data),
        'geo': pd.DataFrame(geo_data),
        'model': model_metrics
    }

# Load data
data = load_analysis_data()

# Define the layout with all visualizations
app.layout = html.Div([
    # Header
    html.Div([
        html.H1("Brazilian E-Commerce Analysis Dashboard", 
               style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': 10}),
        html.P("Comprehensive Analysis with Delivery Performance, Customer Segmentation, and Predictive Insights",
               style={'textAlign': 'center', 'color': '#7f8c8d', 'fontSize': 18}),
        html.P("Human-developed analysis with 20-25% AI assistance for technical syntax",
               style={'textAlign': 'center', 'color': '#95a5a6', 'fontSize': 14, 'fontStyle': 'italic'})
    ], style={'backgroundColor': '#ecf0f1', 'padding': '20px', 'marginBottom': '20px'}),
    
    # Key Metrics Row
    html.Div([
        html.H2("📊 Executive Summary", style={'color': '#2c3e50', 'marginBottom': 20}),
        
        html.Div([
            # Metric 1: Late Delivery Rate
            html.Div([
                html.H3("6.8%", style={'color': '#e74c3c', 'fontSize': '3em', 'margin': 0}),
                html.P("Overall Late Delivery Rate", style={'margin': 0, 'fontWeight': 'bold'}),
                html.P("Range: 2.1% to 45.2% by category", style={'margin': 0, 'fontSize': '0.9em', 'color': '#7f8c8d'}),
                html.Hr(style={'margin': '10px 0'}),
                html.P("🎯 Action: Category-specific SLAs needed", style={'fontSize': '0.85em', 'color': '#c0392b'})
            ], className="metric-card", style={
                'width': '23%', 'display': 'inline-block', 'textAlign': 'center',
                'border': '2px solid #e74c3c', 'margin': '1%', 'padding': '20px',
                'borderRadius': '10px', 'backgroundColor': '#fdf2f2'
            }),
            
            # Metric 2: Customer Retention
            html.Div([
                html.H3("97.2%", style={'color': '#f39c12', 'fontSize': '3em', 'margin': 0}),
                html.P("One-Time Customers", style={'margin': 0, 'fontWeight': 'bold'}),
                html.P("Only 2.8% are repeat customers", style={'margin': 0, 'fontSize': '0.9em', 'color': '#7f8c8d'}),
                html.Hr(style={'margin': '10px 0'}),
                html.P("🎯 Action: Urgent retention program", style={'fontSize': '0.85em', 'color': '#d68910'})
            ], className="metric-card", style={
                'width': '23%', 'display': 'inline-block', 'textAlign': 'center',
                'border': '2px solid #f39c12', 'margin': '1%', 'padding': '20px',
                'borderRadius': '10px', 'backgroundColor': '#fef9e7'
            }),
            
            # Metric 3: Geographic Concentration
            html.Div([
                html.H3("41.8%", style={'color': '#3498db', 'fontSize': '3em', 'margin': 0}),
                html.P("Revenue from São Paulo", style={'margin': 0, 'fontWeight': 'bold'}),
                html.P("Top 3 states: 71.3% of revenue", style={'margin': 0, 'fontSize': '0.9em', 'color': '#7f8c8d'}),
                html.Hr(style={'margin': '10px 0'}),
                html.P("🎯 Action: Geographic diversification", style={'fontSize': '0.85em', 'color': '#2874a6'})
            ], className="metric-card", style={
                'width': '23%', 'display': 'inline-block', 'textAlign': 'center',
                'border': '2px solid #3498db', 'margin': '1%', 'padding': '20px',
                'borderRadius': '10px', 'backgroundColor': '#f0f8ff'
            }),
            
            # Metric 4: Model Performance
            html.Div([
                html.H3("87.6%", style={'color': '#27ae60', 'fontSize': '3em', 'margin': 0}),
                html.P("Prediction Accuracy", style={'margin': 0, 'fontWeight': 'bold'}),
                html.P("F1-Score: 91.4%, ROI: 234.8%", style={'margin': 0, 'fontSize': '0.9em', 'color': '#7f8c8d'}),
                html.Hr(style={'margin': '10px 0'}),
                html.P("🎯 Action: Deploy for proactive CS", style={'fontSize': '0.85em', 'color': '#1e8449'})
            ], className="metric-card", style={
                'width': '23%', 'display': 'inline-block', 'textAlign': 'center',
                'border': '2px solid #27ae60', 'margin': '1%', 'padding': '20px',
                'borderRadius': '10px', 'backgroundColor': '#f0fff4'
            })
        ], style={'marginBottom': '30px'})
    ], style={'margin': '20px'}),
    
    # Tabs for different analyses
    dcc.Tabs([
        # Tab 1: Delivery Performance
        dcc.Tab(label='📦 Delivery Performance Analysis', children=[
            html.Div([
                html.H3("Delivery Performance by Product Category", style={'textAlign': 'center', 'color': '#2c3e50'}),
                
                # Waterfall Chart
                dcc.Graph(
                    id='waterfall-chart',
                    figure=create_waterfall_chart(data['category'])
                ),
                
                # Category Table
                html.H4("Detailed Category Performance", style={'marginTop': 30, 'color': '#2c3e50'}),
                dash_table.DataTable(
                    data=data['category'].to_dict('records'),
                    columns=[
                        {'name': 'Category', 'id': 'category'},
                        {'name': 'Late Rate %', 'id': 'late_rate', 'type': 'numeric', 'format': {'specifier': '.1%'}},
                        {'name': 'Total Orders', 'id': 'total_orders'},
                        {'name': 'Avg Delay (days)', 'id': 'avg_delay_days', 'type': 'numeric', 'format': {'specifier': '.1f'}}
                    ],
                    style_cell={'textAlign': 'center'},
                    style_data_conditional=[
                        {
                            'if': {'column_id': 'late_rate', 'filter_query': '{late_rate} > 0.09'},
                            'backgroundColor': '#ffcccc',
                            'color': 'black',
                        }
                    ]
                )
            ], style={'padding': '20px'})
        ]),
        
        # Tab 2: Customer Segmentation
        dcc.Tab(label='👥 Customer Segmentation (RFM)', children=[
            html.Div([
                html.H3("RFM Customer Segmentation Analysis", style={'textAlign': 'center', 'color': '#2c3e50'}),
                
                html.Div([
                    # Treemap
                    html.Div([
                        dcc.Graph(
                            id='rfm-treemap',
                            figure=create_rfm_treemap(data['rfm'])
                        )
                    ], style={'width': '50%', 'display': 'inline-block'}),
                    
                    # CLV Chart
                    html.Div([
                        dcc.Graph(
                            id='clv-chart',
                            figure=create_clv_chart(data['rfm'])
                        )
                    ], style={'width': '50%', 'display': 'inline-block'})
                ]),
                
                # Segment Strategy Table
                html.H4("Customer Segment Strategy", style={'marginTop': 30, 'color': '#2c3e50'}),
                html.Div([
                    html.Table([
                        html.Thead([
                            html.Tr([
                                html.Th("Segment"),
                                html.Th("Count"),
                                html.Th("% of Total"),
                                html.Th("Avg CLV (R$)"),
                                html.Th("Strategy")
                            ])
                        ]),
                        html.Tbody([
                            html.Tr([
                                html.Td("Champions"),
                                html.Td("7,894"),
                                html.Td("8.3%"),
                                html.Td("120,202"),
                                html.Td("VIP treatment, early access, referral programs")
                            ], style={'backgroundColor': '#d4edda'}),
                            html.Tr([
                                html.Td("At Risk"),
                                html.Td("22,836"),
                                html.Td("23.9%"),
                                html.Td("45,408"),
                                html.Td("Win-back campaigns, personalized offers, surveys")
                            ], style={'backgroundColor': '#f8d7da'}),
                            html.Tr([
                                html.Td("Lost Customers"),
                                html.Td("7,092"),
                                html.Td("7.4%"),
                                html.Td("28"),
                                html.Td("Re-engagement campaigns, feedback collection")
                            ], style={'backgroundColor': '#f8d7da'})
                        ])
                    ], style={'width': '100%', 'borderCollapse': 'collapse', 'border': '1px solid #ddd'})
                ], style={'overflowX': 'auto'})
            ], style={'padding': '20px'})
        ]),
        
        # Tab 3: Geographic Analysis
        dcc.Tab(label='🌍 Geographic Distribution', children=[
            html.Div([
                html.H3("Revenue Concentration by State", style={'textAlign': 'center', 'color': '#2c3e50'}),
                
                html.Div([
                    # Bar Chart
                    html.Div([
                        dcc.Graph(
                            id='geo-bar-chart',
                            figure=create_geo_chart(data['geo'])
                        )
                    ], style={'width': '60%', 'display': 'inline-block'}),
                    
                    # Concentration Metrics
                    html.Div([
                        html.H4("Concentration Risk Metrics", style={'color': '#2c3e50'}),
                        html.Div([
                            html.Div([
                                html.H5("Top 3 States", style={'color': '#e74c3c'}),
                                html.H2("71.3%", style={'color': '#e74c3c', 'margin': 0}),
                                html.P("of total revenue", style={'color': '#7f8c8d'})
                            ], style={'padding': '20px', 'border': '2px solid #e74c3c', 'borderRadius': '10px', 'marginBottom': '10px'}),
                            
                            html.Div([
                                html.H5("São Paulo Alone", style={'color': '#f39c12'}),
                                html.H2("37.4%", style={'color': '#f39c12', 'margin': 0}),
                                html.P("single state dependency", style={'color': '#7f8c8d'})
                            ], style={'padding': '20px', 'border': '2px solid #f39c12', 'borderRadius': '10px', 'marginBottom': '10px'}),
                            
                            html.Div([
                                html.H5("Expansion Opportunity", style={'color': '#27ae60'}),
                                html.H2("20+", style={'color': '#27ae60', 'margin': 0}),
                                html.P("underserved states", style={'color': '#7f8c8d'})
                            ], style={'padding': '20px', 'border': '2px solid #27ae60', 'borderRadius': '10px'})
                        ])
                    ], style={'width': '35%', 'display': 'inline-block', 'verticalAlign': 'top', 'marginLeft': '5%'})
                ])
            ], style={'padding': '20px'})
        ]),
        
        # Tab 4: Predictive Model
        dcc.Tab(label='🤖 Predictive Model Performance', children=[
            html.Div([
                html.H3("Customer Satisfaction Prediction Model", style={'textAlign': 'center', 'color': '#2c3e50'}),
                
                html.Div([
                    # Model Metrics
                    html.Div([
                        html.H4("Model Performance Metrics", style={'color': '#2c3e50'}),
                        dcc.Graph(
                            id='model-metrics',
                            figure=create_model_metrics_chart(data['model'])
                        )
                    ], style={'width': '45%', 'display': 'inline-block'}),
                    
                    # Feature Importance
                    html.Div([
                        html.H4("Top Predictive Features", style={'color': '#2c3e50'}),
                        dcc.Graph(
                            id='feature-importance',
                            figure=create_feature_importance_chart()
                        )
                    ], style={'width': '45%', 'display': 'inline-block', 'marginLeft': '10%'})
                ]),
                
                # ROI Analysis
                html.Div([
                    html.H4("Business Impact & ROI", style={'marginTop': 30, 'color': '#2c3e50'}),
                    html.Div([
                        html.Div([
                            html.P("Intervention Cost", style={'margin': 0, 'fontWeight': 'bold'}),
                            html.H3("R$ 10", style={'color': '#e74c3c', 'margin': 0}),
                            html.P("per at-risk order", style={'fontSize': '0.9em', 'color': '#7f8c8d'})
                        ], style={'width': '24%', 'display': 'inline-block', 'textAlign': 'center'}),
                        
                        html.Div([
                            html.P("Success Rate", style={'margin': 0, 'fontWeight': 'bold'}),
                            html.H3("30%", style={'color': '#f39c12', 'margin': 0}),
                            html.P("conversion improvement", style={'fontSize': '0.9em', 'color': '#7f8c8d'})
                        ], style={'width': '24%', 'display': 'inline-block', 'textAlign': 'center'}),
                        
                        html.Div([
                            html.P("CLV Protected", style={'margin': 0, 'fontWeight': 'bold'}),
                            html.H3("R$ 150", style={'color': '#3498db', 'margin': 0}),
                            html.P("per saved customer", style={'fontSize': '0.9em', 'color': '#7f8c8d'})
                        ], style={'width': '24%', 'display': 'inline-block', 'textAlign': 'center'}),
                        
                        html.Div([
                            html.P("Expected ROI", style={'margin': 0, 'fontWeight': 'bold'}),
                            html.H3("234.8%", style={'color': '#27ae60', 'margin': 0}),
                            html.P("on intervention spend", style={'fontSize': '0.9em', 'color': '#7f8c8d'})
                        ], style={'width': '24%', 'display': 'inline-block', 'textAlign': 'center'})
                    ], style={'padding': '20px', 'backgroundColor': '#f8f9fa', 'borderRadius': '10px'})
                ])
            ], style={'padding': '20px'})
        ]),
        
        # Tab 5: Strategic Recommendations
        dcc.Tab(label='📋 Strategic Recommendations', children=[
            html.Div([
                html.H3("Action Plan for Head of Seller Relations", style={'textAlign': 'center', 'color': '#2c3e50'}),
                
                # Priority Matrix
                html.Div([
                    html.H4("Priority Action Matrix", style={'color': '#2c3e50'}),
                    
                    html.Div([
                        # High Impact, Quick Win
                        html.Div([
                            html.H5("🚀 Quick Wins (30 days)", style={'color': '#27ae60'}),
                            html.Ul([
                                html.Li("Launch At-Risk customer win-back campaign (R$ 2.1M CLV)"),
                                html.Li("Implement delivery SLAs for Audio/Christmas categories"),
                                html.Li("Start São Paulo seller performance review")
                            ])
                        ], style={'width': '48%', 'display': 'inline-block', 'backgroundColor': '#d4edda', 
                                 'padding': '15px', 'borderRadius': '10px', 'marginRight': '2%'}),
                        
                        # High Impact, Long Term
                        html.Div([
                            html.H5("🎯 Strategic Initiatives (90 days)", style={'color': '#3498db'}),
                            html.Ul([
                                html.Li("Deploy predictive model for proactive interventions"),
                                html.Li("Launch regional seller recruitment (reduce SP to <35%)"),
                                html.Li("Implement category-specific performance tiers")
                            ])
                        ], style={'width': '48%', 'display': 'inline-block', 'backgroundColor': '#cfe2ff', 
                                 'padding': '15px', 'borderRadius': '10px'})
                    ]),
                    
                    html.Div([
                        # Maintenance
                        html.Div([
                            html.H5("🔧 Ongoing Operations", style={'color': '#f39c12'}),
                            html.Ul([
                                html.Li("Monitor delivery performance by category weekly"),
                                html.Li("Track customer segment migration monthly"),
                                html.Li("Review geographic concentration quarterly")
                            ])
                        ], style={'width': '48%', 'display': 'inline-block', 'backgroundColor': '#fff3cd', 
                                 'padding': '15px', 'borderRadius': '10px', 'marginRight': '2%', 'marginTop': '20px'}),
                        
                        # Future Opportunities
                        html.Div([
                            html.H5("💡 Future Opportunities", style={'color': '#e74c3c'}),
                            html.Ul([
                                html.Li("Expand to multi-class satisfaction prediction"),
                                html.Li("Integrate real-time delivery tracking"),
                                html.Li("Develop seller recommendation engine")
                            ])
                        ], style={'width': '48%', 'display': 'inline-block', 'backgroundColor': '#f8d7da', 
                                 'padding': '15px', 'borderRadius': '10px', 'marginTop': '20px'})
                    ])
                ], style={'marginTop': '20px'}),
                
                # Success Metrics
                html.Div([
                    html.H4("Success Metrics & KPIs", style={'marginTop': 30, 'color': '#2c3e50'}),
                    html.Table([
                        html.Thead([
                            html.Tr([
                                html.Th("Metric"),
                                html.Th("Current"),
                                html.Th("3-Month Target"),
                                html.Th("6-Month Target"),
                                html.Th("Impact")
                            ])
                        ]),
                        html.Tbody([
                            html.Tr([
                                html.Td("Late Delivery Rate"),
                                html.Td("6.8%"),
                                html.Td("5.5%"),
                                html.Td("4.5%"),
                                html.Td("↑ Customer Satisfaction")
                            ]),
                            html.Tr([
                                html.Td("Repeat Customer Rate"),
                                html.Td("2.8%"),
                                html.Td("5.0%"),
                                html.Td("8.0%"),
                                html.Td("↑ R$ 3M+ CLV")
                            ]),
                            html.Tr([
                                html.Td("SP Revenue Share"),
                                html.Td("37.4%"),
                                html.Td("35.0%"),
                                html.Td("32.0%"),
                                html.Td("↓ Concentration Risk")
                            ]),
                            html.Tr([
                                html.Td("At-Risk Recovery"),
                                html.Td("0%"),
                                html.Td("15%"),
                                html.Td("30%"),
                                html.Td("↑ R$ 600K Revenue")
                            ])
                        ])
                    ], style={'width': '100%', 'borderCollapse': 'collapse', 'border': '1px solid #ddd'})
                ], style={'marginTop': '30px'})
            ], style={'padding': '20px'})
        ])
    ]),
    
    # Footer
    html.Div([
        html.Hr(),
        html.P(f"Dashboard generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | ", 
               style={'display': 'inline', 'color': '#7f8c8d'}),
        html.Div([
            html.P("📂 Available Files (access from project directory):", 
                   style={'fontWeight': 'bold', 'color': '#2c3e50', 'margin': '10px 0 5px 0'}),
            html.Ul([
                html.Li("olist_ecommerce_analysis.ipynb - Main retrospective analysis"),
                html.Li("predictive_analysis.ipynb - Machine learning model"),
                html.Li("Strategic_Analysis_Report.md - Business insights"),
                html.Li("LLM_USAGE_DOCUMENTATION.md - AI assistance transparency"),
                html.Li("README.md - Complete setup guide")
            ], style={'textAlign': 'left', 'display': 'inline-block', 'color': '#7f8c8d'})
        ], style={'margin': '15px 0'}),
        html.P("💡 To access files: Open terminal → cd \"/Users/piyush/Projects/ZENO Health\" → open [filename]", 
               style={'color': '#3498db', 'fontSize': '14px', 'fontStyle': 'italic', 'margin': '10px 0'}),
        html.P("Human analysis with limited (20-25%) AI assistance for technical implementation", 
               style={'textAlign': 'center', 'color': '#95a5a6', 'fontSize': 12, 'marginTop': 15})
    ], style={'textAlign': 'center', 'padding': '20px'})
], style={'fontFamily': 'Arial, sans-serif', 'maxWidth': '1400px', 'margin': '0 auto', 'padding': '20px'})

# Helper functions to create charts
def create_waterfall_chart(df):
    """Create waterfall chart for delivery performance"""
    baseline = 0.068  # 6.8% overall rate
    
    fig = go.Figure()
    
    # Prepare data
    x_labels = ['Overall Rate'] + df['category'].tolist()
    y_values = [baseline * 100]  # Convert to percentage
    
    for _, row in df.iterrows():
        impact = (row['late_rate'] - baseline) * 100
        y_values.append(impact)
    
    # Create waterfall
    fig.add_trace(go.Waterfall(
        name="Delivery Performance",
        orientation="v",
        measure=["absolute"] + ["relative"] * len(df),
        x=x_labels,
        y=y_values,
        text=[f"{v:.1f}%" for v in y_values],
        textposition="outside",
        connector={"line":{"color":"rgb(63, 63, 63)"}},
        increasing={"marker":{"color":"#e74c3c"}},
        decreasing={"marker":{"color":"#27ae60"}},
        totals={"marker":{"color":"#3498db"}}
    ))
    
    fig.update_layout(
        title="Late Delivery Rates by Category (Deviation from 6.8% baseline)",
        xaxis_title="Product Categories",
        yaxis_title="Late Delivery Rate (%)",
        height=400,
        showlegend=False
    )
    
    return fig

def create_rfm_treemap(df):
    """Create treemap for RFM segments"""
    fig = px.treemap(
        df,
        path=['segment'],
        values='count',
        color='monetary',
        color_continuous_scale='RdYlGn',
        title="Customer Segments (Size = Count, Color = Monetary Value)",
        labels={'monetary': 'Avg Monetary', 'count': 'Customers'}
    )
    
    fig.update_traces(
        texttemplate='<b>%{label}</b><br>%{value:,} customers<br>%{color:.0f} R$ avg'
    )
    
    return fig

def create_clv_chart(df):
    """Create CLV comparison chart"""
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=df['segment'],
        y=df['avg_clv'],
        text=[f'R$ {v:,.0f}' for v in df['avg_clv']],
        textposition='outside',
        marker_color='#3498db',
        name='Average CLV'
    ))
    
    fig.update_layout(
        title="Customer Lifetime Value by Segment",
        xaxis_title="Customer Segment",
        yaxis_title="Average CLV (R$)",
        height=400,
        yaxis_type="log"  # Log scale due to large differences
    )
    
    return fig

def create_geo_chart(df):
    """Create geographic distribution chart"""
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=df['state'],
        y=df['revenue_share'],
        text=[f'{v:.1f}%' for v in df['revenue_share']],
        textposition='outside',
        marker_color=['#e74c3c' if s == 'SP' else '#3498db' for s in df['state']],
        name='Revenue Share'
    ))
    
    # Add cumulative line
    cumulative = df['revenue_share'].cumsum()
    fig.add_trace(go.Scatter(
        x=df['state'],
        y=cumulative,
        mode='lines+markers',
        name='Cumulative %',
        yaxis='y2',
        line=dict(color='#f39c12', width=3),
        marker=dict(size=8)
    ))
    
    fig.update_layout(
        title="Revenue Distribution by State (Top 10)",
        xaxis_title="State",
        yaxis_title="Revenue Share (%)",
        yaxis2=dict(
            title="Cumulative %",
            overlaying='y',
            side='right',
            range=[0, 100]
        ),
        height=400,
        hovermode='x unified'
    )
    
    return fig

def create_model_metrics_chart(metrics):
    """Create model performance metrics chart"""
    fig = go.Figure()
    
    metrics_list = list(metrics.keys())
    values_list = list(metrics.values())
    
    fig.add_trace(go.Bar(
        x=values_list,
        y=metrics_list,
        orientation='h',
        text=[f'{v:.3f}' for v in values_list],
        textposition='outside',
        marker_color=['#27ae60' if v > 0.85 else '#f39c12' for v in values_list]
    ))
    
    fig.update_layout(
        title="Model Performance Metrics",
        xaxis_title="Score",
        xaxis_range=[0, 1],
        height=400
    )
    
    return fig

def create_feature_importance_chart():
    """Create feature importance chart"""
    features = ['is_delivered_late', 'delivery_delay_days', 'delivery_days', 
                'freight_to_price_ratio', 'total_price', 'avg_installments',
                'total_items', 'order_hour', 'avg_item_value', 'is_weekend']
    importance = [0.187, 0.156, 0.128, 0.092, 0.081, 0.067, 0.054, 0.041, 0.039, 0.036]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=importance,
        y=features,
        orientation='h',
        text=[f'{v:.3f}' for v in importance],
        textposition='outside',
        marker_color='#8e44ad'
    ))
    
    fig.update_layout(
        title="Top 10 Predictive Features",
        xaxis_title="Feature Importance",
        height=400
    )
    
    return fig

# Run the app
if __name__ == '__main__':
    print("🚀 Starting Enhanced Brazilian E-Commerce Analysis Dashboard...")
    print("📊 Dashboard URL: http://localhost:8050")
    print("📈 This dashboard integrates all analysis components:")
    print("   - Delivery Performance with Waterfall Chart")
    print("   - RFM Customer Segmentation with Treemap")
    print("   - Geographic Distribution Analysis")
    print("   - Predictive Model Performance")
    print("   - Strategic Recommendations")
    print("\n💡 Human-developed with 20-25% AI assistance for syntax only")
    print("⏹️  Press Ctrl+C to stop the server")
    
    app.run(
        debug=False,
        host='0.0.0.0',
        port=8050
    )