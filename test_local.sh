#!/bin/bash

# Test script for local ComfyUI workflow testing
# Make sure you have a reference image in the same directory

echo "Testing ComfyUI Flux+PuLID+Depth workflow locally..."

# Basic test
echo "Running basic test..."
cog predict \
  -i prompt="wearing a elegant dress at a fancy restaurant" \
  -i reference_image=@reference.jpg \
  -i width=768 \
  -i height=1024

echo "Test completed! Check the output image."

# Advanced test with all parameters
echo "Running advanced test with all parameters..."
cog predict \
  -i prompt="wearing a cowboy outfit with hat at a western saloon" \
  -i negative_prompt="ugly, blurry, low quality, deformed, extra limbs" \
  -i reference_image=@reference.jpg \
  -i width=768 \
  -i height=1024 \
  -i steps=25 \
  -i cfg=1.5 \
  -i guidance=3.5 \
  -i pulid_weight=1.2 \
  -i controlnet_strength=0.7 \
  -i seed=12345 \
  -i enable_face_swap=true

echo "Advanced test completed!" 