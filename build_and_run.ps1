param (
    [string]$Command = "demo-neurons",
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$RemainingArgs
)

$RepoRoot = $PSScriptRoot
Set-Location $RepoRoot

New-Item -ItemType Directory -Force -Path "$RepoRoot\bin\core", "$RepoRoot\bin\app" | Out-Null

Write-Host "[1/2] Compiling java-core..." -ForegroundColor Cyan
$coreFiles = (Get-ChildItem -Path "$RepoRoot\java-core\src\main\java\com\traffic\core\*.java").FullName
javac -d "$RepoRoot\bin\core" $coreFiles
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Compilation of java-core failed." -ForegroundColor Red
    exit $LASTEXITCODE
}

Write-Host "[2/2] Compiling java-app..." -ForegroundColor Cyan
$appFiles = (Get-ChildItem -Path "$RepoRoot\java-app\src\main\java\com\traffic\app\*.java").FullName
javac -cp "$RepoRoot\bin\core" -d "$RepoRoot\bin\app" $appFiles
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Compilation of java-app failed." -ForegroundColor Red
    exit $LASTEXITCODE
}

Write-Host "[SUCCESS] Compilation completed successfully!`n" -ForegroundColor Green

$cp = "$RepoRoot\bin\core;$RepoRoot\bin\app"
$fullArgs = @("-cp", $cp, "com.traffic.app.Main", $Command) + $RemainingArgs
java @fullArgs
