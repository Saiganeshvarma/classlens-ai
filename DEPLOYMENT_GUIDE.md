# ClassLens AI - Streamlit Deployment Guide

This guide covers deploying ClassLens AI to Streamlit Community Cloud using the Google Gemini API (free tier).

## Prerequisites

- GitHub account
- Streamlit Community Cloud account (free at https://share.streamlit.io)
- Google Gemini API key (free at https://aistudio.google.com/app/apikey)

## Step 1: Get Your Free Gemini API Key

1. Visit https://aistudio.google.com/app/apikey
2. Sign in with your Google account
3. Click "Create API Key" or "Get API Key"
4. Copy the API key (starts with `AIza...`)
5. Keep this key secure - you'll need it in Step 4

**Free Tier Limits:**
- 15 requests per minute
- 1,500 requests per day
- No billing required

## Step 2: Prepare Your Repository

1. **Push to GitHub** (if not already done):
   ```bash
   cd classlens-ai
   git init
   git add .
   git commit -m "Initial commit - ClassLens AI with Gemini support"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/classlens-ai.git
   git push -u origin main
   ```

2. **Verify these files exist** (they're already configured):
   - `requirements.txt` - Python dependencies (includes `google-generativeai`)
   - `packages.txt` - System dependencies (includes `ffmpeg`)
   - `.streamlit/config.toml` - Streamlit configuration
   - `.gitignore` - Ensures `.env` and secrets are not committed

## Step 3: Deploy to Streamlit Cloud

1. Go to https://share.streamlit.io
2. Click "New app"
3. Select your repository: `YOUR_USERNAME/classlens-ai`
4. Set the main file path: `app.py`
5. Click "Advanced settings" (optional):
   - Python version: `3.12` (default)
   - Leave other settings as default

## Step 4: Configure Secrets

Before deploying, set up your API key and configuration:

1. In the Streamlit Cloud dashboard, go to your app
2. Click the menu (⋮) → "Settings" → "Secrets"
3. Paste the following configuration:

```toml
# AI Provider Configuration
AI_PROVIDER = "gemini"
GEMINI_API_KEY = "AIzaSyAb8RN6KBFhQyr_OWMB6SvAhCg3p73MZxvSeq1UgTj78mLxPtAA"

# Model Selection
EVALUATION_MODEL = "gemini-1.5-flash"
TRANSCRIPTION_MODEL = "whisper-1"

# Processing Limits
MAX_FILE_SIZE_MB = 500
MAX_DURATION_SECONDS = 3600
MAX_FRAMES = 12
FRAME_INTERVAL_SECONDS = 30

# Scoring Thresholds
THRESHOLD_POOR_MAX = 39
THRESHOLD_AVERAGE_MAX = 59
THRESHOLD_GOOD_MAX = 79

# Logging
LOG_LEVEL = "INFO"

# Timeout Settings
LLM_TIMEOUT_SECONDS = 600
```

4. **Replace** `AIzaSyAb8RN6KBFhQyr_OWMB6SvAhCg3p73MZxvSeq1UgTj78mLxPtAA` with your actual Gemini API key
5. Click "Save"

## Step 5: Deploy

1. Click "Deploy!"
2. Wait for the deployment to complete (2-5 minutes)
3. The app will automatically:
   - Install Python dependencies from `requirements.txt`
   - Install FFmpeg from `packages.txt`
   - Load secrets as environment variables
   - Start the Streamlit app

## Step 6: Verify Deployment

Once deployed, your app should:
- Show the ClassLens AI interface
- NOT show any "Setup required" warnings (if you do, check your secrets configuration)
- Be ready to accept video/audio uploads

Test the app:
1. Upload a short class recording (< 5 minutes recommended for testing)
2. Click "Analyse Class"
3. Watch the progress steps
4. Review the generated report

## Troubleshooting

### "Setup required" warning appears
- Check that `GEMINI_API_KEY` is correctly set in Secrets
- Ensure `AI_PROVIDER = "gemini"` is set in Secrets
- Restart the app: Settings → Reboot app

### "Gemini authentication failed"
- Verify your API key is valid at https://aistudio.google.com/app/apikey
- Ensure the key doesn't have extra spaces or quotes
- Check the key format starts with `AIza`

### "Gemini rate limit reached"
**Free tier limits:**
- 15 requests per minute
- 1,500 requests per day

**Solutions:**
- Wait 1-2 minutes and retry
- For higher limits, consider upgrading at https://aistudio.google.com

### "Audio extraction failed"
- Ensure the uploaded file is a valid video/audio file
- Check file size is under `MAX_FILE_SIZE_MB` (default 500MB)
- Verify FFmpeg is installed (check app logs)

### Transcription takes a long time
When using Gemini, transcription uses **local Whisper** (free, no API needed). This is slower than API-based transcription but completely free:
- Short videos (< 5 min): 30-60 seconds
- Medium videos (5-15 min): 1-3 minutes
- Long videos (15-60 min): 3-10 minutes

This is normal and expected.

### "Memory limit exceeded"
Streamlit Cloud free tier has memory limits. Solutions:
- Reduce `MAX_FILE_SIZE_MB` to 200
- Reduce `MAX_FRAMES` to 8
- Use shorter recordings for testing

### Logs and debugging
- Click the menu (⋮) → "Manage app" → "Logs"
- Look for error messages
- Check that secrets are loaded: look for "AI provider: gemini" in logs

## Monitoring Usage

**Gemini API usage:**
- Visit https://aistudio.google.com/app/apikey
- Click on your API key
- View usage statistics

**Streamlit Cloud:**
- View app analytics in the Streamlit Cloud dashboard
- Monitor app status and uptime

## Cost Optimization

ClassLens AI with Gemini is **completely free** for moderate usage:
- ✅ Gemini 1.5 Flash: Free tier (15 RPM, 1500 requests/day)
- ✅ Local Whisper transcription: Free (runs on Streamlit's servers)
- ✅ Streamlit Community Cloud: Free hosting

**Expected costs:** $0/month for typical usage

## Alternative: OpenAI Deployment

If you prefer OpenAI instead of Gemini:

1. Update secrets:
```toml
AI_PROVIDER = "openai"
OPENAI_API_KEY = "sk-..."
EVALUATION_MODEL = "gpt-4o"
```

2. Note: OpenAI is not free:
   - Whisper: ~$0.006 per minute
   - GPT-4o: ~$5 per 1M input tokens
   - Estimated cost: ~$0.50-2 per evaluation

## Security Best Practices

1. **Never commit API keys** to Git
2. **Use Streamlit Secrets** for all sensitive configuration
3. **Rotate API keys** regularly
4. **Monitor usage** to detect unauthorized access
5. **Set usage alerts** in Google Cloud Console (optional)

## Updating Your Deployment

To update your deployed app:

```bash
# Make changes to your code
git add .
git commit -m "Update: description of changes"
git push

# Streamlit Cloud auto-deploys on push
```

The app will automatically redeploy when you push to the main branch.

## Support

For issues:
- Check the troubleshooting section above
- Review Streamlit Cloud logs
- Check Gemini API status: https://status.cloud.google.com/
- Review ClassLens AI README.md for configuration details

## Next Steps

Once deployed:
1. Share your app URL with users
2. Monitor usage and performance
3. Collect feedback
4. Iterate on the evaluation rubric
5. Consider adding custom domains (Streamlit Pro)

---

**Your app is now live! 🎉**

Share your Streamlit app URL: `https://YOUR_APP_NAME.streamlit.app`
