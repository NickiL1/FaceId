#!/bin/bash

# ----------------------------
# Full installation & run script
# ----------------------------

# 1️⃣ Set your repository URL
REPO_URL="https://github.com/NickiL1/FaceId"
APP_DIR="Main"
REP_DIR="FaceId"

echo "Cloning repository..."
if [ -d "$REP_DIR" ]; then
    echo "Directory $REP_DIR already exists. Removing it..."
    rm -rf $REP_DIR
fi

git clone $REPO_URL
cd $REP_DIR
export PYTHONPATH=$(pwd)
cd $APP_DIR

# 🔥 Remove Git history so it's just a clean folder
rm -rf .git

# 2️⃣ Check Python
if ! command -v python3 &> /dev/null; then
    echo "Python3 not found. Please install Python 3.10+"
    exit 1
fi

# 3️⃣ Create virtual environment
python3 -m venv .venv

# 4️⃣ Activate virtual environment
source .venv/bin/activate

# 5️⃣ Upgrade pip
pip install --upgrade pip

# 6️⃣ Install dependencies
echo "Installing dependencies..."
pip install --no-cache-dir -r ../requirements.txt

# 7️⃣ Run FastAPI app
echo "Starting FastAPI app..."
# Optional: open browser automatically
echo "App running at: http://localhost:8000/video_feed"
python3 app.py
