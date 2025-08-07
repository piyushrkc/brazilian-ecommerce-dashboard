#!/bin/bash
echo "🔧 Fixing deployment issues..."

# Ensure we have the right files
cp app_improved.py app.py
cp requirements_deploy.txt requirements.txt

# Push the fixes to GitHub
git add .
git commit -m "Fix: Updated Python version and dependencies for Render deployment

- Set Python version to 3.11.9 (stable)
- Updated package versions for compatibility
- Added explicit setuptools dependency
- Fixed pip setuptools.build_meta import issue"

git push origin main

echo "✅ Fixes pushed to GitHub"
echo "🚀 Render will automatically redeploy"
echo "⏳ Wait 5-10 minutes for new build to complete"
echo "🌐 Check your dashboard at: https://brazilian-ecommerce-dashboard.onrender.com"