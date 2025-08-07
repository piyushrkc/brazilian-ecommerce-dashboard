#!/usr/bin/env python3
"""
Simple Dashboard Test - Brazilian E-Commerce Analysis
===================================================

Simplified version to test if dashboard works on your system.
Run this first to verify everything is working.
"""

import dash
from dash import html
import webbrowser
from threading import Timer

# Create simple app
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("🎉 Dashboard Test Successful!", style={'textAlign': 'center', 'color': 'green'}),
    html.P("If you can see this page, your dashboard setup is working correctly!", 
           style={'textAlign': 'center', 'fontSize': '18px'}),
    html.Hr(),
    html.H3("Next Steps:", style={'color': '#2c3e50'}),
    html.Ol([
        html.Li("Stop this test dashboard (Ctrl+C in terminal)"),
        html.Li("Run the full dashboard: python dashboard_app.py"),
        html.Li("Access the complete analysis at http://localhost:8050")
    ]),
    html.P("🚀 Ready to explore the Brazilian E-Commerce Analysis!", 
           style={'textAlign': 'center', 'fontWeight': 'bold', 'marginTop': '30px'})
])

def open_browser():
    webbrowser.open_new("http://localhost:8050")

if __name__ == '__main__':
    print("🧪 Testing Dashboard Setup...")
    print("🌐 Opening browser to: http://localhost:8050")
    print("⏹️  Press Ctrl+C to stop")
    
    # Auto-open browser after 1 second
    Timer(1, open_browser).start()
    
    app.run(debug=False, port=8050, host='0.0.0.0')