# ADIE v2.0 — Production Deployment Guide

> Deploy ADIE on a Linux server (Ubuntu 22.04/24.04) accessible via public IP.
> Final result: `http://<YOUR_SERVER_IP>` serves the full ADIE application.

---

## Prerequisites

- A Linux server (Ubuntu 22.04+ recommended) with a public IP address
  - Options: AWS EC2, DigitalOcean Droplet, Azure VM, Google Cloud VM, any VPS
  - Minimum: 2 CPU cores, 4GB RAM (ML tasks need memory)
- SSH access to the server as root or a sudo user
- Your project files (this folder) transferred to the server

---

## Step 1: Transfer Project to Server

From your local Windows machine, use SCP or FileZilla:

```bash
# Option A: SCP (from PowerShell or Git Bash)
scp -r "C:\Users\hp\Downloads\AI-Project-1\AI-Project" user@YOUR_SERVER_IP:/home/user/

# Option B: Git (if you push to a repo first)
ssh user@YOUR_SERVER_IP
git clone https://github.com/YOUR_USERNAME/AI-Project.git
```

---

## Step 2: Server Setup (SSH into your server)

```bash
# Connect to your server
ssh user@YOUR_SERVER_IP

# Update system packages
sudo apt update && sudo apt upgrade -y

# Install Python 3.11+, pip, venv, and build tools
sudo apt install -y python3 python3-pip python3-venv python3-dev \
    build-essential libffi-dev libssl-dev nginx curl

# Verify Python version (should be 3.10+)
python3 --version
```

---

## Step 3: Create Virtual Environment & Install Dependencies

```bash
# Navigate to project directory
cd /home/user/AI-Project

# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install project dependencies
pip install -r requirements.txt

# Install Gunicorn (production WSGI server)
pip install gunicorn

# Verify critical imports work
python3 -c "import flask, pandas, sklearn, reportlab; print('All dependencies OK')"
```

---

## Step 4: Prepare the Application for Production

### 4.1 Create required directories with proper permissions

```bash
# Ensure uploads directory exists and is writable
mkdir -p /home/user/AI-Project/uploads
mkdir -p /home/user/AI-Project/data/default
chmod -R 755 /home/user/AI-Project/uploads

# Set ownership (replace 'user' with your actual username)
sudo chown -R user:user /home/user/AI-Project
```

### 4.2 Create production configuration

Create a file called `config.py` in the project root:

```bash
cat > /home/user/AI-Project/config.py << 'EOF'
import os

class ProductionConfig:
    SECRET_KEY = os.environ.get('ADIE_SECRET_KEY', 'change-this-to-a-random-string-in-production')
    UPLOAD_FOLDER = 'uploads'
    MAX_CONTENT_LENGTH = 100 * 1024 * 1024  # 100MB max upload
    DEBUG = False
    TESTING = False
EOF
```

### 4.3 Create a production entry point

Create `wsgi.py` in the project root:

```bash
cat > /home/user/AI-Project/wsgi.py << 'EOF'
"""
WSGI entry point for production deployment.
Gunicorn will import 'app' from this module.
"""
import os

# Production settings
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

from app import app

# Override with production config
app.config['DEBUG'] = False
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB

# Use environment variable for secret key
secret = os.environ.get('ADIE_SECRET_KEY')
if secret:
    app.secret_key = secret

if __name__ == '__main__':
    app.run()
EOF
```

### 4.4 Set environment variable for secret key

```bash
# Generate a random secret key
export ADIE_SECRET_KEY=$(python3 -c "import secrets; print(secrets.token_hex(32))")

# Add to .bashrc so it persists across sessions
echo "export ADIE_SECRET_KEY='$ADIE_SECRET_KEY'" >> ~/.bashrc
source ~/.bashrc
```

---

## Step 5: Test with Gunicorn (Quick Check)

```bash
cd /home/user/AI-Project
source venv/bin/activate

# Test run (Ctrl+C to stop after verifying it works)
gunicorn -w 4 -b 0.0.0.0:5000 --timeout 300 wsgi:app

# You should see:
# [INFO] Starting gunicorn 21.x.x
# [INFO] Listening at: http://0.0.0.0:5000
```

Open `http://YOUR_SERVER_IP:5000` in a browser to verify it loads.
Then press `Ctrl+C` to stop.

---

## Step 6: Create systemd Service (Auto-start on Boot)

```bash
sudo cat > /etc/systemd/system/adie.service << 'EOF'
[Unit]
Description=ADIE v2.0 - Automated Dataset Intelligence Engine
After=network.target

[Service]
User=user
Group=user
WorkingDirectory=/home/user/AI-Project
Environment="PATH=/home/user/AI-Project/venv/bin"
Environment="ADIE_SECRET_KEY=REPLACE_WITH_YOUR_SECRET_KEY"
ExecStart=/home/user/AI-Project/venv/bin/gunicorn \
    --workers 4 \
    --bind 127.0.0.1:5000 \
    --timeout 300 \
    --graceful-timeout 120 \
    --max-requests 1000 \
    --max-requests-jitter 50 \
    --access-logfile /home/user/AI-Project/logs/access.log \
    --error-logfile /home/user/AI-Project/logs/error.log \
    --log-level info \
    wsgi:app
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF
```

**Important**: Replace `user` with your actual username and set the secret key:

```bash
# Replace the placeholder secret key in the service file
SECRET=$(python3 -c "import secrets; print(secrets.token_hex(32))")
sudo sed -i "s/REPLACE_WITH_YOUR_SECRET_KEY/$SECRET/" /etc/systemd/system/adie.service

# Also replace 'user' with your actual username if different
# sudo sed -i 's/User=user/User=yourusername/' /etc/systemd/system/adie.service
# sudo sed -i 's/Group=user/Group=yourusername/' /etc/systemd/system/adie.service

# Create logs directory
mkdir -p /home/user/AI-Project/logs

# Enable and start the service
sudo systemctl daemon-reload
sudo systemctl enable adie
sudo systemctl start adie

# Check status
sudo systemctl status adie
```

You should see `Active: active (running)`.

---

## Step 7: Configure Nginx (Reverse Proxy)

```bash
# Remove default Nginx config
sudo rm -f /etc/nginx/sites-enabled/default

# Create ADIE Nginx config
sudo cat > /etc/nginx/sites-available/adie << 'EOF'
server {
    listen 80;
    server_name _;

    # Max upload size (100MB for large datasets)
    client_max_body_size 100M;

    # Timeouts for ML processing (5 minutes)
    proxy_connect_timeout 300s;
    proxy_send_timeout 300s;
    proxy_read_timeout 300s;
    send_timeout 300s;

    # Serve static files directly (faster than proxying)
    location /static/ {
        alias /home/user/AI-Project/static/;
        expires 7d;
        add_header Cache-Control "public, immutable";
    }

    # Proxy all other requests to Gunicorn
    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # WebSocket support (if needed in future)
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";

        # Buffering for large responses (PDF reports)
        proxy_buffering on;
        proxy_buffer_size 128k;
        proxy_buffers 4 256k;
        proxy_busy_buffers_size 256k;
    }

    # Security headers
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;

    # Deny access to hidden files
    location ~ /\. {
        deny all;
    }

    # Deny access to uploads directory directly (prevent browsing)
    location /uploads/ {
        deny all;
    }
}
EOF

# Enable the site
sudo ln -sf /etc/nginx/sites-available/adie /etc/nginx/sites-enabled/adie

# Test Nginx configuration
sudo nginx -t

# If test passes, restart Nginx
sudo systemctl restart nginx
sudo systemctl enable nginx
```

---

## Step 8: Open Firewall Ports

```bash
# If using UFW (Ubuntu's firewall)
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw allow 22/tcp    # SSH (don't lock yourself out!)
sudo ufw enable
sudo ufw status

# If using AWS/GCP/Azure, also open ports 80 and 443 in the
# Security Group / Firewall Rules via the cloud console.
```

---

## Step 9: Verify Deployment

```bash
# Check all services are running
sudo systemctl status adie
sudo systemctl status nginx

# Test locally on the server
curl -I http://localhost

# Should return: HTTP/1.1 200 OK
```

Now open in your browser:
```
http://YOUR_SERVER_IP
```

You should see the ADIE splash page. Log in with `admin` / `password123`.

### Full Pipeline Test:
1. Login → Dashboard
2. Upload a CSV file → verify preview + target detection works
3. Click "Execute Pipeline" → verify cleaning completes
4. Click "Run Benchmark" → verify ML training completes
5. Click "Download PDF Report" → verify PDF downloads
6. Click "Run System Evaluation" → verify evaluation runs

---

## Step 10 (Optional): HTTPS with Let's Encrypt

If you have a domain name pointing to your server:

```bash
# Install Certbot
sudo apt install -y certbot python3-certbot-nginx

# Get SSL certificate (replace with your domain)
sudo certbot --nginx -d yourdomain.com

# Certbot will automatically:
# - Obtain the certificate
# - Modify your Nginx config to use HTTPS
# - Set up auto-renewal

# Test auto-renewal
sudo certbot renew --dry-run
```

After this, your app will be accessible at:
```
https://yourdomain.com
```

---

## Configuration Files Summary

### File: `/home/user/AI-Project/wsgi.py`
Production WSGI entry point (created in Step 4.3)

### File: `/etc/systemd/system/adie.service`
Systemd service for auto-start (created in Step 6)

### File: `/etc/nginx/sites-available/adie`
Nginx reverse proxy config (created in Step 7)

---

## Useful Commands

```bash
# Restart the app after code changes
sudo systemctl restart adie

# View live application logs
tail -f /home/user/AI-Project/logs/error.log
tail -f /home/user/AI-Project/logs/access.log

# View Nginx logs
tail -f /var/log/nginx/error.log
tail -f /var/log/nginx/access.log

# Check if Gunicorn is running
ps aux | grep gunicorn

# Check which ports are in use
sudo ss -tlnp | grep -E '80|5000'

# Restart everything
sudo systemctl restart adie && sudo systemctl restart nginx
```

---

## Troubleshooting

| Problem | Solution |
|---|---|
| 502 Bad Gateway | Gunicorn not running: `sudo systemctl restart adie` |
| 504 Gateway Timeout | Increase `proxy_read_timeout` in Nginx and `--timeout` in Gunicorn |
| 413 Request Entity Too Large | Increase `client_max_body_size` in Nginx config |
| Static files not loading | Check `alias` path in Nginx matches your actual project path |
| Permission denied on uploads | `chmod -R 755 uploads/` and check ownership |
| Module not found | Activate venv: `source venv/bin/activate && pip install -r requirements.txt` |
| App crashes on large dataset | Increase server RAM or add swap: `sudo fallocate -l 4G /swapfile` |
| PDF download fails | Verify reportlab installed: `pip show reportlab` |

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────┐
│                    INTERNET                               │
│                  (Public IP)                              │
└────────────────────────┬────────────────────────────────┘
                         │ Port 80 (HTTP) / 443 (HTTPS)
                         ▼
┌─────────────────────────────────────────────────────────┐
│                      NGINX                               │
│              (Reverse Proxy)                              │
│                                                          │
│  /static/* ──→ Serve directly from filesystem            │
│  /*        ──→ Proxy to Gunicorn (127.0.0.1:5000)       │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│                    GUNICORN                               │
│            (4 worker processes)                           │
│            Timeout: 300 seconds                          │
│            Bind: 127.0.0.1:5000                          │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│                   FLASK APP (ADIE)                        │
│                                                          │
│  wsgi.py → app.py → utils/*.py                          │
│                                                          │
│  Reads/Writes: uploads/ (CSV, JSON, PKL, PDF)           │
│  Templates:    templates/*.html                          │
│  Static:       static/style.css                          │
└─────────────────────────────────────────────────────────┘
```

---

## Final Checklist

- [ ] Server has 4GB+ RAM
- [ ] Python 3.10+ installed
- [ ] Virtual environment created with all dependencies
- [ ] `wsgi.py` created (production entry point)
- [ ] Gunicorn installed in venv
- [ ] systemd service created and enabled
- [ ] Nginx configured with correct paths
- [ ] Firewall ports 80/443 open
- [ ] `uploads/` directory exists and is writable
- [ ] `logs/` directory exists
- [ ] Secret key set via environment variable
- [ ] Debug mode disabled
- [ ] App accessible at `http://YOUR_SERVER_IP`
- [ ] File upload works
- [ ] Full pipeline works (analyze → clean → train)
- [ ] PDF report downloads correctly
- [ ] System evaluation runs

---

*Generated for ADIE v2.0 — April 2026*
