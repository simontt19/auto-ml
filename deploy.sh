#!/bin/bash

# Auto ML Framework Deployment Script
# This script deploys the FastAPI application to Heroku

set -e

echo "🚀 Starting Auto ML Framework deployment to Heroku..."

# Check if Heroku CLI is installed
if ! command -v heroku &> /dev/null; then
    echo "❌ Heroku CLI is not installed. Please install it first:"
    echo "   https://devcenter.heroku.com/articles/heroku-cli"
    exit 1
fi

# Check if user is logged in to Heroku
if ! heroku auth:whoami &> /dev/null; then
    echo "❌ Not logged in to Heroku. Please run: heroku login"
    exit 1
fi

# Get app name from command line or use default
APP_NAME=${1:-"auto-ml-api-$(date +%s)"}

echo "📦 Deploying to Heroku app: $APP_NAME"

# Create Heroku app if it doesn't exist
if ! heroku apps:info $APP_NAME &> /dev/null; then
    echo "🔧 Creating new Heroku app: $APP_NAME"
    heroku create $APP_NAME
else
    echo "✅ Using existing Heroku app: $APP_NAME"
fi

# Set environment variables
echo "⚙️ Setting environment variables..."
heroku config:set PYTHON_VERSION=3.11.7 --app $APP_NAME
heroku config:set LOG_LEVEL=INFO --app $APP_NAME
heroku config:set ENVIRONMENT=production --app $APP_NAME

# Add PostgreSQL addon
echo "🗄️ Adding PostgreSQL database..."
heroku addons:create heroku-postgresql:mini --app $APP_NAME

# Deploy the application
echo "🚀 Deploying application..."
git add .
git commit -m "Deploy Auto ML API to Heroku" || true
git push heroku main

# Open the application
echo "🌐 Opening application..."
heroku open --app $APP_NAME

# Show logs
echo "📋 Showing application logs..."
heroku logs --tail --app $APP_NAME &

echo "✅ Deployment completed!"
echo "🔗 Application URL: https://$APP_NAME.herokuapp.com"
echo "📊 Health check: https://$APP_NAME.herokuapp.com/health"
echo "📚 API docs: https://$APP_NAME.herokuapp.com/docs"

# Wait for user to stop logs
echo "Press Ctrl+C to stop viewing logs..."
wait 