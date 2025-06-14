# Auto ML Framework Deployment Guide

This guide covers deploying the Auto ML Framework to production using free tier platforms.

## Overview

The Auto ML Framework can be deployed to multiple platforms:

- **Heroku**: Free tier with PostgreSQL addon
- **Railway**: Free tier with automatic scaling
- **Vercel**: For the web dashboard
- **Hugging Face**: For model sharing

## Prerequisites

### Required Accounts

- [Heroku Account](https://signup.heroku.com/) (free tier)
- [Railway Account](https://railway.app/) (free tier)
- [Vercel Account](https://vercel.com/) (free tier)
- [Hugging Face Account](https://huggingface.co/) (free tier)

### Required Tools

- Git (for version control)
- Heroku CLI (for Heroku deployment)
- Railway CLI (for Railway deployment)

## Deployment Options

### Option 1: Heroku Deployment (Recommended)

#### Quick Deploy

```bash
# Make deployment script executable
chmod +x deploy.sh

# Deploy with custom app name
./deploy.sh my-auto-ml-api

# Or deploy with auto-generated name
./deploy.sh
```

#### Manual Deploy

```bash
# 1. Login to Heroku
heroku login

# 2. Create app
heroku create auto-ml-api-$(date +%s)

# 3. Set environment variables
heroku config:set PYTHON_VERSION=3.11.7
heroku config:set LOG_LEVEL=INFO
heroku config:set ENVIRONMENT=production

# 4. Add PostgreSQL
heroku addons:create heroku-postgresql:mini

# 5. Deploy
git push heroku main

# 6. Open app
heroku open
```

#### Environment Variables

```bash
# Required
PYTHON_VERSION=3.11.7
LOG_LEVEL=INFO
ENVIRONMENT=production

# Optional
DATABASE_URL=postgresql://...  # Auto-set by Heroku PostgreSQL
```

### Option 2: Railway Deployment

#### Quick Deploy

1. Connect your GitHub repository to Railway
2. Railway will automatically detect the `railway.json` configuration
3. Deploy with one click

#### Manual Deploy

```bash
# 1. Install Railway CLI
npm install -g @railway/cli

# 2. Login to Railway
railway login

# 3. Initialize project
railway init

# 4. Deploy
railway up
```

### Option 3: Automated GitHub Actions

The framework includes automated deployment via GitHub Actions:

1. **Set up GitHub Secrets**:

   - `HEROKU_API_KEY`: Your Heroku API key
   - `HEROKU_APP_NAME`: Your Heroku app name
   - `HEROKU_EMAIL`: Your Heroku email
   - `HEROKU_APP_URL`: Your Heroku app URL

2. **Push to main branch**:

   ```bash
   git push origin main
   ```

3. **Monitor deployment**:
   - Check GitHub Actions tab
   - View deployment logs
   - Verify health checks

## API Endpoints

Once deployed, your API will be available at:

### Base URL

- **Heroku**: `https://your-app-name.herokuapp.com`
- **Railway**: `https://your-app-name.railway.app`

### Available Endpoints

- `GET /` - API information
- `GET /health` - Health check
- `GET /docs` - Interactive API documentation
- `GET /users` - List users
- `GET /projects` - List projects
- `POST /predict` - Make predictions

### Authentication

All endpoints require Bearer token authentication:

```bash
curl -H "Authorization: Bearer your-token" \
     https://your-app.herokuapp.com/health
```

## Monitoring and Health Checks

### Health Check

```bash
curl https://your-app.herokuapp.com/health
```

Expected response:

```json
{
  "status": "healthy",
  "timestamp": "2024-06-14T12:00:00",
  "models_loaded": 0,
  "users_online": 1,
  "projects_active": 2
}
```

### Logs

```bash
# Heroku
heroku logs --tail --app your-app-name

# Railway
railway logs
```

## Troubleshooting

### Common Issues

#### 1. Build Failures

**Problem**: App fails to build
**Solution**:

- Check `requirements.txt` for missing dependencies
- Verify Python version in `runtime.txt`
- Check build logs for specific errors

#### 2. Import Errors

**Problem**: Module import errors
**Solution**:

- Ensure all imports use relative paths
- Check `PYTHONPATH` configuration
- Verify file structure matches imports

#### 3. Port Issues

**Problem**: App crashes on startup
**Solution**:

- Use `$PORT` environment variable
- Check Procfile configuration
- Verify uvicorn command syntax

#### 4. Database Connection

**Problem**: Database connection errors
**Solution**:

- Check `DATABASE_URL` environment variable
- Verify PostgreSQL addon is active
- Test database connectivity

### Debug Commands

```bash
# Check app status
heroku ps --app your-app-name

# View recent logs
heroku logs --app your-app-name

# Run one-off dyno for debugging
heroku run python --app your-app-name

# Check environment variables
heroku config --app your-app-name
```

## Performance Optimization

### Free Tier Limits

- **Heroku**: 512MB RAM, 30 minutes/day sleep
- **Railway**: 512MB RAM, 500 hours/month
- **Vercel**: 100GB bandwidth/month

### Optimization Tips

1. **Memory Usage**: Monitor memory consumption
2. **Cold Starts**: Use keep-alive for frequent requests
3. **Caching**: Implement response caching
4. **Database**: Use connection pooling
5. **Logging**: Reduce log verbosity in production

## Security Considerations

### Production Security

1. **HTTPS**: All platforms provide HTTPS by default
2. **Authentication**: Implement proper JWT tokens
3. **Rate Limiting**: Add rate limiting for API endpoints
4. **Input Validation**: Validate all API inputs
5. **Secrets**: Use environment variables for sensitive data

### Environment Variables

```bash
# Never commit these to version control
JWT_SECRET_KEY=your-secret-key
DATABASE_URL=your-database-url
API_KEYS=your-api-keys
```

## Next Steps

After successful deployment:

1. **Test API Endpoints**: Verify all endpoints work
2. **Set up Monitoring**: Configure health checks and alerts
3. **Deploy Dashboard**: Deploy the web dashboard to Vercel
4. **Model Sharing**: Set up Hugging Face integration
5. **CI/CD**: Configure automated testing and deployment

## Support

For deployment issues:

1. Check the troubleshooting section
2. Review platform-specific documentation
3. Check GitHub Issues for known problems
4. Contact platform support if needed

---

**Note**: This deployment guide focuses on free tier platforms. For production-scale deployments, consider paid tiers or cloud providers like AWS, GCP, or Azure.
