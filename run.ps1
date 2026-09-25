Set-Location $PSScriptRoot
$VENV_DIR = "psdenv"
$PY = Join-Path $VENV_DIR "Scripts\python.exe"

try {
    # Create the virtual environment (py launcher first: "python" is often not on PATH on Windows)
    if (-not (Test-Path $PY)) {
        Write-Host "Creating virtual environment..."
        if (Get-Command py -ErrorAction SilentlyContinue) { py -3 -m venv $VENV_DIR } else { python -m venv $VENV_DIR }
        if ($LASTEXITCODE -ne 0) { throw "could not create the virtual environment" }
    }

    # Install requirements when requirements.txt changed since the last install
    $marker = Join-Path $VENV_DIR "requirements.installed"
    if (-not (Test-Path $marker) -or (Get-FileHash requirements.txt).Hash -ne (Get-FileHash $marker).Hash) {
        Write-Host "Installing requirements..."
        & $PY -m pip install -r requirements.txt
        if ($LASTEXITCODE -ne 0) { throw "could not install requirements" }
        Copy-Item requirements.txt $marker -Force
    }

    & $PY main.py
}
catch {
    Write-Host "Setup failed: $_"
}

Read-Host -Prompt "Press Enter to continue..."
