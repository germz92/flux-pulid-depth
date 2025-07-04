# ComfyUI Workflow Deployment on Replicate

This repository contains the necessary files to deploy your ComfyUI workflows on Replicate.

## Quick Start

1. **Add your ComfyUI workflow JSON**:
   - Export your workflow from ComfyUI (right-click → "Save (API Format)")
   - Either replace the default workflow in `predict.py` or pass it as a parameter

2. **Customize the model**:
   - Update the `cog.yaml` file with any additional dependencies
   - Modify the `predict.py` file to match your workflow requirements

3. **Deploy to Replicate**:
   ```bash
   # Install cog
   sudo curl -o /usr/local/bin/cog -L https://github.com/replicate/cog/releases/latest/download/cog_`uname -s`_`uname -m`
   sudo chmod +x /usr/local/bin/cog

   # Test locally
   cog predict -i prompt="a beautiful landscape"

   # Deploy to Replicate
   cog push r8.im/your-username/your-model-name
   ```

## File Structure

- `cog.yaml` - Replicate configuration file
- `predict.py` - Main prediction script that runs ComfyUI workflows
- `workflow.json` - Example ComfyUI workflow (optional)
- `README.md` - This file

## Customization

### Adding Your Workflow

1. **Method 1: Replace default workflow**
   - Edit the `get_default_workflow()` method in `predict.py`
   - Paste your workflow JSON there

2. **Method 2: Pass workflow as parameter**
   - Keep the current setup
   - Pass your workflow JSON as the `workflow_json` parameter when calling the API

### Adding Custom Nodes

If your workflow uses custom nodes:

1. Update `cog.yaml` to install the custom nodes:
   ```yaml
   run:
     - "git clone https://github.com/comfyanonymous/ComfyUI.git"
     - "cd ComfyUI && pip install -r requirements.txt"
     - "cd ComfyUI/custom_nodes && git clone https://github.com/your-custom-node-repo.git"
   ```

### Adding Models

For custom models, update `cog.yaml`:

```yaml
run:
  - "git clone https://github.com/comfyanonymous/ComfyUI.git"
  - "cd ComfyUI && pip install -r requirements.txt"
  - "wget -O ComfyUI/models/checkpoints/your-model.safetensors https://your-model-url"
```

## API Usage

Once deployed, you can use your model like this:

```python
import replicate

output = replicate.run(
    "your-username/your-model-name",
    input={
        "prompt": "wearing a red dress at a party",
        "negative_prompt": "ugly, blurry, low quality",
        "reference_image": open("path/to/reference.jpg", "rb"),
        "width": 768,
        "height": 1024,
        "steps": 20,
        "cfg": 1.5,
        "guidance": 3.5,
        "pulid_weight": 1.0,
        "controlnet_strength": 0.6,
        "seed": 42,
        "enable_face_swap": True
    }
)
```

## Input Parameters

- `workflow_json`: Your ComfyUI workflow JSON (optional, uses default Flux+PuLID+Depth workflow)
- `prompt`: Text prompt for generation
- `negative_prompt`: Negative prompt to avoid unwanted elements
- `reference_image`: Reference image for PuLID face preservation (required)
- `width`: Width of the generated image (64-2048, default: 768)
- `height`: Height of the generated image (64-2048, default: 1024)
- `steps`: Number of inference steps (1-100, default: 20)
- `cfg`: CFG scale (0.1-20.0, default: 1.5)
- `guidance`: Flux guidance scale (0.1-10.0, default: 3.5)
- `pulid_weight`: PuLID weight for face preservation (0.0-2.0, default: 1.0)
- `controlnet_strength`: ControlNet strength for depth control (0.0-1.0, default: 0.6)
- `seed`: Random seed for reproducibility (-1 for random)
- `enable_face_swap`: Enable ReActor face swap (default: True)

## Notes

- The default workflow uses Flux with PuLID for face preservation and depth ControlNet
- A reference image is required for the PuLID face preservation to work
- The workflow includes ReActor face swap for better face consistency
- GPU is required for this workflow (uses CUDA)
- Make sure to test locally with `cog predict` before deploying 