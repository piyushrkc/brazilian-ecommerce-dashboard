#!/bin/bash
# Enhanced Dashboard Launch Script
# ================================

echo "🚀 Brazilian E-Commerce Analysis Dashboard Launcher"
echo "================================================="
echo ""
echo "📋 Pre-flight checklist..."

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found. Please install Python 3.8 or later."
    exit 1
fi

echo "✅ Python 3 found: $(python3 --version)"

# Check current directory
echo "📂 Current directory: $(pwd)"

# Check if enhanced_dashboard.py exists
if [ ! -f "enhanced_dashboard.py" ]; then
    echo "❌ enhanced_dashboard.py not found in current directory"
    echo "   Please run this script from: /Users/piyush/Projects/ZENO Health"
    exit 1
fi

echo "✅ Dashboard file found"

# Check for required packages
echo ""
echo "📦 Checking required packages..."
packages=("dash" "plotly" "pandas" "numpy")
missing_packages=()

for package in "${packages[@]}"; do
    if python3 -c "import $package" 2>/dev/null; then
        echo "✅ $package is installed"
    else
        echo "❌ $package is missing"
        missing_packages+=($package)
    fi
done

# Install missing packages if any
if [ ${#missing_packages[@]} -gt 0 ]; then
    echo ""
    echo "📥 Installing missing packages..."
    for package in "${missing_packages[@]}"; do
        echo "Installing $package..."
        pip install $package
    done
fi

# Launch the dashboard
echo ""
echo "🎯 Launching Enhanced Dashboard..."
echo "================================="
echo ""
echo "📊 The dashboard includes:"
echo "   • Executive Summary with key metrics"
echo "   • Delivery Performance with Waterfall Chart"
echo "   • RFM Customer Segmentation with Treemap"
echo "   • Geographic Distribution Analysis"
echo "   • Predictive Model Performance"
echo "   • Strategic Recommendations"
echo ""
echo "🌐 Dashboard will be available at: http://localhost:8050"
echo ""
echo "💡 Human-developed with 20-25% AI assistance for syntax only"
echo ""
echo "⏹️  Press Ctrl+C to stop the server"
echo ""
echo "Starting server..."
echo "=================="

# Run the dashboard
python3 enhanced_dashboard.py