# ----------------------------
# Full installation & run script (Windows PowerShell)
# ----------------------------

# 1️⃣ Set your repository URL
$REPO_URL = "https://github.com/NickiL1/FaceId"
$APP_DIR = "Main"
$REP_DIR = "FaceId"

Write-Output "Cloning repository..."
if (Test-Path $REP_DIR) {
    Write-Output "Directory $REP_DIR already exists. Removing it..."
    Remove-Item -Recurse -Force $REP_DIR
}

git clone $REPO_URL
Set-Location $REP_DIR

# Set PYTHONPATH so modules can be found
$env:PYTHONPATH = (Get-Location).Path
Set-Location $APP_DIR

# 🔥 Remove Git history so it's just a clean folder
if (Test-Path ".git") {
    Remove-Item -Recurse -Force .git
}

# 2️⃣ Check Python
if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Output "Python not found. Please install Python 3.10+"
    exit 1
}

# 3️⃣ Create virtual environment
python -m venv .venv

# 4️⃣ Activate virtual environment
. .\.venv\Scripts\Activate.ps1

# 5️⃣ Upgrade pip
pip install --upgrade pip

# 6️⃣ Install dependencies
Write-Output "Installing dependencies..."
pip install --no-cache-dir -r ..\requirements.txt

# 7️⃣ Run FastAPI app
Write-Output "Starting FastAPI app..."
Write-Output "App running at: http://localhost:8000/video_feed"
python app.py
