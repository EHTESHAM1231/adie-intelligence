# Deploy ADIE to Render — Quick Steps (5 minutes)

## Step 1: Push to GitHub (from your terminal)

Open **Git Bash** or **PowerShell** in the `AI-Project` folder and run:

```bash
cd "C:\Users\hp\Downloads\AI-Project-1\AI-Project"

git init
git add .
git commit -m "ADIE v2.0 - ready for Render deployment"

# Create a new repo on GitHub first (https://github.com/new)
# Name it: adie-intelligence (or whatever you want)
# Then:

git remote add origin https://github.com/YOUR_USERNAME/adie-intelligence.git
git branch -M main
git push -u origin main
```

## Step 2: Deploy on Render (2 minutes)

1. Go to **https://render.com** → Sign up / Log in (use GitHub)
2. Click **"New +"** → **"Web Service"**
3. Connect your GitHub repo (`adie-intelligence`)
4. Settings:
   - **Name**: `adie-intelligence`
   - **Region**: Pick closest to you
   - **Branch**: `main`
   - **Runtime**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app --bind 0.0.0.0:$PORT --timeout 120 --workers 2`
   - **Instance Type**: `Free`
5. Click **"Create Web Service"**

## Step 3: Wait for Build (~3-5 minutes)

Render will:
- Clone your repo
- Install dependencies
- Start Gunicorn

When you see **"Your service is live"**, you're done.

## Step 4: Access Your App

Your URL will be:
```
https://adie-intelligence.onrender.com
```

Login with: `admin` / `password123`

---

## Demo Recording Tips

1. **First load is slow** (Render free tier spins down after 15 min of inactivity — first request takes ~30 seconds to wake up)
2. **Use a small dataset** for the demo (under 5000 rows works best)
3. **Pipeline flow**: Upload → Analyze → Clean → Train → Download Report
4. **Don't close the tab** between steps (session is in-memory)
5. If it times out on training, select **"Decision Tree"** instead of "All Algorithms" — it's fastest

---

## If Something Goes Wrong

| Issue | Fix |
|---|---|
| Build fails | Check Render logs → usually a missing dependency |
| App crashes on upload | Dataset too large — use a smaller CSV (<5000 rows) |
| 502 error | App is still starting — wait 30 seconds and refresh |
| Session lost | Render restarted — just log in again |
| PDF download fails | reportlab might not install — report will fall back to .txt |

---

## Files Created for Deployment

| File | Purpose |
|---|---|
| `Procfile` | Tells Render how to start the app |
| `render.yaml` | Auto-configuration for Render |
| `.gitignore` | Excludes large files from git |
| `requirements.txt` | All Python dependencies + gunicorn |

No other files were changed structurally. The app works exactly the same locally.
