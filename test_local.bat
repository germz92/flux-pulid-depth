@echo off
echo Testing ComfyUI Flux+PuLID+Depth workflow locally on Windows...
echo.

REM Check if cog.exe exists
if not exist "cog.exe" (
    echo ERROR: cog.exe not found in current directory
    echo Please download it first with:
    echo Invoke-WebRequest -Uri "https://github.com/replicate/cog/releases/latest/download/cog_Windows_x86_64.exe" -OutFile "cog.exe"
    pause
    exit /b 1
)

REM Check if reference image exists
if not exist "reference.jpg" (
    echo ERROR: reference.jpg not found in current directory
    echo Please add your reference image as "reference.jpg" in this folder
    pause
    exit /b 1
)

echo Running basic test...
cog.exe predict -i prompt="wearing an elegant dress at a fancy restaurant" -i reference_image=@reference.jpg -i width=768 -i height=1024

echo.
echo Basic test completed! Check the output image.
echo.

echo Running advanced test with all parameters...
cog.exe predict ^
  -i prompt="wearing a cowboy outfit with hat at a western saloon" ^
  -i negative_prompt="ugly, blurry, low quality, deformed, extra limbs" ^
  -i reference_image=@reference.jpg ^
  -i width=768 ^
  -i height=1024 ^
  -i steps=25 ^
  -i cfg=1.5 ^
  -i guidance=3.5 ^
  -i pulid_weight=1.2 ^
  -i controlnet_strength=0.7 ^
  -i seed=12345 ^
  -i enable_face_swap=true

echo.
echo Advanced test completed!
echo.
pause 