@echo off
setlocal enabledelayedexpansion

cd /d "%~dp0"

if not exist "bin\core" mkdir "bin\core"
if not exist "bin\app" mkdir "bin\app"

echo [1/2] Compiling java-core...
dir /b /s java-core\src\main\java\com\traffic\core\*.java > core_sources.txt
javac -d bin\core @core_sources.txt
del core_sources.txt
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Failed to compile java-core.
    exit /b %ERRORLEVEL%
)

echo [2/2] Compiling java-app...
dir /b /s java-app\src\main\java\com\traffic\app\*.java > app_sources.txt
javac -cp bin\core -d bin\app @app_sources.txt
del app_sources.txt
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Failed to compile java-app.
    exit /b %ERRORLEVEL%
)

echo [SUCCESS] Compilation completed!
echo.

if "%~1"=="" (
    echo Running default 'demo-neurons' command...
    java -cp "bin\core;bin\app" com.traffic.app.Main demo-neurons
) else (
    java -cp "bin\core;bin\app" com.traffic.app.Main %*
)
