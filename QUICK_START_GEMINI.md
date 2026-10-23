# Quick Start: ClassLens AI with Google Gemini

Get ClassLens AI running locally with Google Gemini in 5 minutes.

## Why Gemini?

✅ **Free tier** - No credit card required  
✅ **15 requests per minute** - Sufficient for testing  
✅ **1,500 requests per day** - Free forever  
✅ **Local Whisper transcription** - No API costs for transcription  
✅ **High quality** - Gemini 1.5 Flash performs excellently for evaluation  

## Step 1: Get Your Free Gemini API Key (2 minutes)

1. Visit: https://aistudio.google.com/app/apikey
2. Sign in with your Google account
3. Click "Create API Key" or "Get API Key"
4. Copy the API key (starts with `AIza...`)

**That's it!** No billing setup required.

## Step 2: Clone and Setup (2 minutes)

```bash
# Clone the repository (or navigate to your existing directory)
cd classlens-ai

# Create virtual environment
python3.12 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Step 3: Configure Environment (1 minute)

Create a `.env` file:

```bash
cp .env.example .env
```

Edit `.env` and set:

```bash
AI_PROVIDER=gemini
GEMINI_API_KEY=AIzaSyAb8RN6KBFhQyr_OWMB6SvAhCg3p73MZxvSeq1UgTj78mLxPtAA  # Replace with your key
EVALUATION_MODEL=gemini-1.5-flash
```

## Step 4: Install FFmpeg

**macOS:**
```bash
brew install ffmpeg
```

**Ubuntu/Debian:**
```bash
sudo apt-get update && sudo apt-get install -y ffmpeg
```

**Windows:**
Download from https://ffmpeg.org/download.html and add to PATH.

## Step 5: Run the App

```bash
streamlit run app.py
```

The app opens at http://localhost:8501

## Step 6: Test It

1. Upload a short class recording (video or audio)
2. Click "Analyse Class"
3. Wait for processing (transcription uses local Whisper - takes 1-2 minutes for short videos)
4. Review the evaluation report!

## What Happens Behind the Scenes?

**When you click "Analyse Class":**

1. ✅ **Video validation** - Checks format, duration, size
2. ✅ **Audio extraction** - FFmpeg extracts audio to WAV
3. 🆓 **Transcription** - Local Whisper transcribes (free, no API)
4. ✅ **Audio analysis** - Measures silence, noise, clipping
5. ✅ **Frame extraction** - Samples video frames for visual analysis
6. 🤖 **AI evaluation** - Gemini analyzes and scores (free tier)
7. ✅ **Report generation** - Creates structured JSON report

**Total cost: $0** 💰

## Gemini vs OpenAI Comparison

| Feature | Gemini (Free) | OpenAI (Paid) |
|---------|--------------|---------------|
| API Cost | Free (15 RPM, 1500/day) | ~$0.05 per request |
| Transcription | Local Whisper (free) | API Whisper (~$0.006/min) |
| Setup | Just API key | API key + billing |
| Quality | Excellent | Excellent |
| Speed | Fast | Fast |
| Context Window | 1M tokens | 128K tokens |
| **Total Cost/Evaluation** | **$0** | **~$0.50-2** |

## Troubleshooting

### "GEMINI_API_KEY is not configured"
- Check that `.env` exists and contains `GEMINI_API_KEY=AIza...`
- Ensure no extra spaces around the `=`
- Restart the app

### "Gemini authentication failed"
- Verify your API key at https://aistudio.google.com/app/apikey
- Create a new key if needed
- Check for typos

### "Gemini rate limit reached"
Free tier limits:
- 15 requests per minute
- 1,500 requests per day

Wait 1-2 minutes and retry.

### Transcription is slow
This is normal! Local Whisper runs on your CPU:
- Short video (< 5 min): 30-60 seconds
- Medium video (5-15 min): 1-3 minutes
- Long video (15-60 min): 3-10 minutes

This is the tradeoff for free transcription.

### FFmpeg not found
- Install FFmpeg (see Step 4)
- Verify: `ffmpeg -version`
- Ensure it's on your PATH

## Next Steps

### Deploy to Streamlit Cloud (Free!)
See `DEPLOYMENT_GUIDE.md` for step-by-step instructions.

### Try Different Models
Edit `.env`:
```bash
EVALUATION_MODEL=gemini-1.5-pro  # More powerful but same free tier
```

### Adjust Processing Settings
Edit `.env`:
```bash
MAX_FRAMES=8              # Fewer frames = faster processing
FRAME_INTERVAL_SECONDS=60 # Sample frames less frequently
MAX_DURATION_SECONDS=1800 # Limit to 30-minute videos
```

### Monitor Your Usage
- Visit https://aistudio.google.com/app/apikey
- Click your API key to see usage stats

## Tips for Best Results

1. **Audio quality matters** - Clear audio = better transcription
2. **Keep videos under 30 minutes** - Faster processing, better context
3. **Test with short clips first** - Verify setup before long videos
4. **Check the free tier limits** - 15 requests/minute, 1500/day
5. **Use good lighting in videos** - Better frame analysis

## Support

- **Gemini API docs:** https://ai.google.dev/docs
- **Streamlit docs:** https://docs.streamlit.io
- **ClassLens README:** See `README.md` for full documentation
- **Issues:** Open an issue on GitHub

---

**You're all set! Start evaluating class recordings with AI. 🎓🤖**

**Total setup time: ~5 minutes**  
**Total cost: $0** ✨
