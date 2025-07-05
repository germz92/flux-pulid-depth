import os
import sys
import json
import tempfile
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
import shutil
import requests
from PIL import Image
import base64
import io

# Add ComfyUI to Python path
sys.path.append('/src/ComfyUI')

from cog import BasePredictor, Input, Path as CogPath

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        self.comfy_dir = Path("/src/ComfyUI")
        self.models_dir = self.comfy_dir / "models"
        self.output_dir = self.comfy_dir / "output"
        self.temp_dir = self.comfy_dir / "temp"
        
        # Ensure ComfyUI directory exists
        if not self.comfy_dir.exists():
            raise FileNotFoundError(f"ComfyUI directory not found at {self.comfy_dir}")
        
        # Create necessary directories
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Download models if they don't exist
        self.download_models()
        
        # Start ComfyUI server
        self.start_comfyui_server()
    
    def download_models(self):
        """Download required models if they don't exist"""
        models_to_download = [
            {
                "url": "https://huggingface.co/Kijai/flux-fp8/resolve/main/flux1-dev-fp8.safetensors",
                "path": "ComfyUI/models/checkpoints/flux1-dev-fp8.safetensors"
            },
            {
                "url": "https://huggingface.co/ffxvs/vae-flux/resolve/main/ae.safetensors",
                "path": "ComfyUI/models/vae/ae.safetensors"
            },
            {
                "url": "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors",
                "path": "ComfyUI/models/clip/clip_l.safetensors"
            },
            {
                "url": "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp8_e4m3fn.safetensors",
                "path": "ComfyUI/models/clip/t5xxl_fp8_e4m3fn.safetensors"
            },
            {
                "url": "https://huggingface.co/Shakker-Labs/FLUX.1-dev-ControlNet-Depth/resolve/main/diffusion_pytorch_model.safetensors",
                "path": "ComfyUI/models/controlnet/flux-depth-controlnet-v3.safetensors"
            },
            {
                "url": "https://huggingface.co/guozinan/PuLID/resolve/main/pulid_flux_v0.9.1.safetensors",
                "path": "ComfyUI/models/pulid/pulid_flux_v0.9.1.safetensors"
            },
            {
                "url": "https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx",
                "path": "ComfyUI/models/insightface/inswapper_128.onnx"
            },
            {
                "url": "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth",
                "path": "ComfyUI/models/facerestore_models/GFPGANv1.4.pth"
            }
        ]
        
        for model in models_to_download:
            model_path = Path(f"/src/{model['path']}")
            if not model_path.exists():
                print(f"Downloading {model_path.name}...")
                self.download_file(model['url'], model_path)
                print(f"Downloaded {model_path.name}")
            else:
                print(f"Model {model_path.name} already exists, skipping download")
    
    def download_file(self, url: str, path: Path):
        """Download a file from URL to the specified path"""
        import subprocess
        path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["wget", "-O", str(path), url], check=True)
    
    def start_comfyui_server(self):
        """Start the ComfyUI server"""
        os.chdir(str(self.comfy_dir))
        
        # Start ComfyUI server in the background
        self.server_process = subprocess.Popen([
            sys.executable, "main.py", 
            "--listen", "127.0.0.1", 
            "--port", "8188",
            "--disable-auto-launch"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait for server to start
        time.sleep(10)
        
        # Check if server is running
        try:
            response = requests.get("http://127.0.0.1:8188")
            print("ComfyUI server is running")
        except Exception as e:
            print(f"Error starting ComfyUI server: {e}")
            raise
    
    def predict(
        self,
        workflow_json: str = Input(
            description="ComfyUI workflow JSON string (leave empty to use default Flux+PuLID+Depth workflow)",
            default=""
        ),
        prompt: str = Input(
            description="Text prompt for generation",
            default="wearing western outfit with a cowboy hat at a bar in nashville"
        ),
        negative_prompt: str = Input(
            description="Negative prompt",
            default="nsfw, nude, deformed, ugly, extra limbs, blurry, extra fingers, helmet"
        ),
        reference_image: CogPath = Input(
            description="Reference image for PuLID face preservation (required)",
        ),
        width: int = Input(
            description="Width of the generated image",
            default=768,
            ge=64,
            le=2048
        ),
        height: int = Input(
            description="Height of the generated image", 
            default=1024,
            ge=64,
            le=2048
        ),
        steps: int = Input(
            description="Number of inference steps",
            default=20,
            ge=1,
            le=100
        ),
        cfg: float = Input(
            description="CFG scale",
            default=1.5,
            ge=0.1,
            le=20.0
        ),
        guidance: float = Input(
            description="Flux guidance scale",
            default=3.5,
            ge=0.1,
            le=10.0
        ),
        pulid_weight: float = Input(
            description="PuLID weight for face preservation",
            default=1.0,
            ge=0.0,
            le=2.0
        ),
        controlnet_strength: float = Input(
            description="ControlNet strength for depth control",
            default=0.6,
            ge=0.0,
            le=1.0
        ),
        seed: int = Input(
            description="Random seed for reproducibility",
            default=-1
        ),
        enable_face_swap: bool = Input(
            description="Enable ReActor face swap",
            default=True
        )
    ) -> CogPath:
        """Run a ComfyUI workflow and return the generated image"""
        
        # If no workflow JSON provided, use a default one
        if not workflow_json:
            workflow_json = self.get_default_workflow()
        
        # Parse workflow
        try:
            workflow = json.loads(workflow_json)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in workflow: {e}")
        
        # Update workflow with user inputs
        workflow = self.update_workflow_inputs(
            workflow, prompt, negative_prompt, reference_image, width, height, 
            steps, cfg, guidance, pulid_weight, controlnet_strength, seed, enable_face_swap
        )
        
        # Queue the workflow
        response = self.queue_workflow(workflow)
        
        # Wait for completion and get results
        output_path = self.wait_for_completion(response)
        
        return CogPath(output_path)
    
    def get_default_workflow(self) -> str:
        """Return the Flux+PuLID+Depth workflow JSON"""
        default_workflow = {
            "108": {
                "inputs": {
                    "vae_name": "ae.sft"
                },
                "class_type": "VAELoader",
                "_meta": {
                    "title": "Load VAE"
                }
            },
            "113": {
                "inputs": {
                    "width": 768,
                    "height": 1024,
                    "batch_size": 1
                },
                "class_type": "EmptySD3LatentImage",
                "_meta": {
                    "title": "EmptySD3LatentImage"
                }
            },
            "114": {
                "inputs": {
                    "pulid_file": "pulid_flux_v0.9.1.safetensors"
                },
                "class_type": "PulidFluxModelLoader",
                "_meta": {
                    "title": "Load PuLID Flux Model"
                }
            },
            "119": {
                "inputs": {},
                "class_type": "PulidFluxEvaClipLoader",
                "_meta": {
                    "title": "Load Eva Clip (PuLID Flux)"
                }
            },
            "120": {
                "inputs": {
                    "image": "DS World 2305280 (1) (1).jpg"
                },
                "class_type": "LoadImage",
                "_meta": {
                    "title": "Load Image"
                }
            },
            "123": {
                "inputs": {
                    "clip_name1": "t5xxl_fp8_e4m3fn.safetensors",
                    "clip_name2": "clip_l.safetensors",
                    "type": "flux",
                    "device": "default"
                },
                "class_type": "DualCLIPLoader",
                "_meta": {
                    "title": "DualCLIPLoader"
                }
            },
            "124": {
                "inputs": {
                    "provider": "CUDA"
                },
                "class_type": "PulidFluxInsightFaceLoader",
                "_meta": {
                    "title": "Load InsightFace (PuLID Flux)"
                }
            },
            "178": {
                "inputs": {
                    "weight": 1.0,
                    "start_at": 0,
                    "end_at": 1,
                    "model": [
                        "204",
                        0
                    ],
                    "pulid_flux": [
                        "114",
                        0
                    ],
                    "eva_clip": [
                        "119",
                        0
                    ],
                    "face_analysis": [
                        "124",
                        0
                    ],
                    "image": [
                        "120",
                        0
                    ]
                },
                "class_type": "ApplyPulidFlux",
                "_meta": {
                    "title": "Apply PuLID Flux"
                }
            },
            "196": {
                "inputs": {
                    "guidance": 3.5,
                    "conditioning": [
                        "203",
                        0
                    ]
                },
                "class_type": "FluxGuidance",
                "_meta": {
                    "title": "FluxGuidance"
                }
            },
            "201": {
                "inputs": {
                    "samples": [
                        "247",
                        0
                    ],
                    "vae": [
                        "108",
                        0
                    ]
                },
                "class_type": "VAEDecode",
                "_meta": {
                    "title": "VAE Decode"
                }
            },
            "203": {
                "inputs": {
                    "text": "wearing western outfit with a cowboy hat at a bar in nashville",
                    "clip": [
                        "123",
                        0
                    ]
                },
                "class_type": "CLIPTextEncode",
                "_meta": {
                    "title": "CLIP Text Encode (Prompt)"
                }
            },
            "204": {
                "inputs": {
                    "unet_name": "flux1-dev-fp8.safetensors",
                    "weight_dtype": "fp8_e4m3fn"
                },
                "class_type": "UNETLoader",
                "_meta": {
                    "title": "Load Diffusion Model"
                }
            },
            "239": {
                "inputs": {
                    "text": "",
                    "clip": [
                        "123",
                        0
                    ]
                },
                "class_type": "CLIPTextEncode",
                "_meta": {
                    "title": "CLIP Text Encode (Prompt)"
                }
            },
            "247": {
                "inputs": {
                    "seed": 270086661936310,
                    "steps": 20,
                    "cfg": 1.5,
                    "sampler_name": "dpmpp_2m",
                    "scheduler": "sgm_uniform",
                    "denoise": 1,
                    "model": [
                        "178",
                        0
                    ],
                    "positive": [
                        "263",
                        0
                    ],
                    "negative": [
                        "263",
                        1
                    ],
                    "latent_image": [
                        "113",
                        0
                    ]
                },
                "class_type": "KSampler",
                "_meta": {
                    "title": "KSampler"
                }
            },
            "248": {
                "inputs": {
                    "text": "nsfw, nude, deformed, ugly, extra limbs, blurry, extra fingers, helmet",
                    "clip": [
                        "123",
                        0
                    ]
                },
                "class_type": "CLIPTextEncode",
                "_meta": {
                    "title": "CLIP Text Encode (Prompt)"
                }
            },
            "257": {
                "inputs": {
                    "enabled": True,
                    "swap_model": "inswapper_128.onnx",
                    "facedetection": "retinaface_resnet50",
                    "face_restore_model": "GFPGANv1.4.pth",
                    "face_restore_visibility": 1,
                    "codeformer_weight": 0.5,
                    "detect_gender_input": "no",
                    "detect_gender_source": "no",
                    "input_faces_index": "0",
                    "source_faces_index": "0",
                    "console_log_level": 1,
                    "input_image": [
                        "201",
                        0
                    ],
                    "source_image": [
                        "120",
                        0
                    ]
                },
                "class_type": "ReActorFaceSwap",
                "_meta": {
                    "title": "ReActor 🌌 Fast Face Swap"
                }
            },
            "260": {
                "inputs": {
                    "a": 6.283185307179586,
                    "bg_threshold": 0.1,
                    "resolution": 512,
                    "image": [
                        "120",
                        0
                    ]
                },
                "class_type": "MiDaS-DepthMapPreprocessor",
                "_meta": {
                    "title": "MiDaS Depth Map"
                }
            },
            "263": {
                "inputs": {
                    "strength": 0.6,
                    "start_percent": 0,
                    "end_percent": 0.6,
                    "positive": [
                        "196",
                        0
                    ],
                    "negative": [
                        "248",
                        0
                    ],
                    "control_net": [
                        "264",
                        0
                    ],
                    "image": [
                        "260",
                        0
                    ]
                },
                "class_type": "ControlNetApplyAdvanced",
                "_meta": {
                    "title": "Apply ControlNet"
                }
            },
            "264": {
                "inputs": {
                    "control_net_name": "flux-depth-controlnet-v3.safetensors"
                },
                "class_type": "ControlNetLoader",
                "_meta": {
                    "title": "Load ControlNet Model"
                }
            },
            "265": {
                "inputs": {
                    "images": [
                        "257",
                        0
                    ]
                },
                "class_type": "PreviewImage",
                "_meta": {
                    "title": "Preview Image"
                }
            },
            "266": {
                "inputs": {
                    "filename_prefix": "ComfyUI",
                    "images": [
                        "257",
                        0
                    ]
                },
                "class_type": "SaveImage",
                "_meta": {
                    "title": "Save Image"
                }
            }
        }
        
        return json.dumps(default_workflow)
    
    def update_workflow_inputs(self, workflow: Dict, prompt: str, negative_prompt: str, reference_image: CogPath, 
                             width: int, height: int, steps: int, cfg: float, guidance: float, 
                             pulid_weight: float, controlnet_strength: float, seed: int, enable_face_swap: bool) -> Dict:
        """Update workflow with user inputs"""
        
        # Save reference image to temp location
        import shutil
        temp_image_path = self.temp_dir / "reference_image.jpg"
        shutil.copy(str(reference_image), str(temp_image_path))
        
        # Update specific nodes by ID for this workflow
        # Node 203: Main text prompt (positive)
        if "203" in workflow:
            workflow["203"]["inputs"]["text"] = prompt
        
        # Node 248: Negative prompt
        if "248" in workflow:
            workflow["248"]["inputs"]["text"] = negative_prompt
        
        # Node 247: KSampler settings
        if "247" in workflow:
            workflow["247"]["inputs"]["steps"] = steps
            workflow["247"]["inputs"]["cfg"] = cfg
            workflow["247"]["inputs"]["seed"] = seed if seed >= 0 else int(time.time())
        
        # Node 113: Image dimensions
        if "113" in workflow:
            workflow["113"]["inputs"]["width"] = width
            workflow["113"]["inputs"]["height"] = height
        
        # Node 178: PuLID weight
        if "178" in workflow:
            workflow["178"]["inputs"]["weight"] = pulid_weight
        
        # Node 196: Flux guidance
        if "196" in workflow:
            workflow["196"]["inputs"]["guidance"] = guidance
        
        # Node 263: ControlNet strength
        if "263" in workflow:
            workflow["263"]["inputs"]["strength"] = controlnet_strength
            workflow["263"]["inputs"]["end_percent"] = controlnet_strength
        
        # Node 120: Reference image
        if "120" in workflow:
            workflow["120"]["inputs"]["image"] = str(temp_image_path.name)
        
        # Node 257: ReActor face swap enable/disable
        if "257" in workflow:
            workflow["257"]["inputs"]["enabled"] = enable_face_swap
        
        return workflow
    
    def queue_workflow(self, workflow: Dict) -> Dict:
        """Queue a workflow for execution"""
        url = "http://127.0.0.1:8188/prompt"
        
        payload = {
            "prompt": workflow,
            "client_id": "replicate_client"
        }
        
        response = requests.post(url, json=payload)
        response.raise_for_status()
        
        return response.json()
    
    def wait_for_completion(self, response: Dict) -> str:
        """Wait for workflow completion and return output path"""
        prompt_id = response["prompt_id"]
        
        while True:
            # Check if workflow is complete
            history_url = f"http://127.0.0.1:8188/history/{prompt_id}"
            history_response = requests.get(history_url)
            
            if history_response.status_code == 200:
                history = history_response.json()
                if prompt_id in history:
                    # Workflow is complete
                    break
            
            time.sleep(1)
        
        # Find the output image (look for most recent files)
        output_files = []
        for ext in ["*.png", "*.jpg", "*.jpeg"]:
            output_files.extend(self.output_dir.glob(ext))
        
        if output_files:
            # Return the most recent file
            latest_file = max(output_files, key=os.path.getctime)
            return str(latest_file)
        else:
            raise RuntimeError("No output images found")
    
    def __del__(self):
        """Clean up resources"""
        if hasattr(self, 'server_process'):
            self.server_process.terminate()
            self.server_process.wait() 