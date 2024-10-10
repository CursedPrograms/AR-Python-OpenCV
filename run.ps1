$VENV_DIR = "psdenv"

# Check if the virtual environment directory exists
if (-not (Test-Path $VENV_DIR)) {
    # Create the virtual environment
    python -m venv $VENV_DIR
}

# Activate the virtual environment and run the Python script
& "$VENV_DIR\Scripts\Activate.ps1"
python main.py

# Pause for user input before closing (optional)
Read-Host -Prompt "Press Enter to continue..."