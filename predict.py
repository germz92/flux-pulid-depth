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
        self.workflow_path = "/src/workflow.json"
        self.server_address = "127.0.0.1:8188"
        self.server_process = None
        self.setup_complete = False
        
    async def setup_async(self):
        """Async setup for downloading models and starting ComfyUI"""
        if self.setup_complete:
            return
            
        print("* Verifying build-time models exist...")
        if not self.verify_build_time_models():
            raise RuntimeError("ERROR: Build-time models missing! Check build logs.")
        
        print("* Downloading large models at runtime...")
        await self.download_large_models()
        
        print("* Verifying PuLID custom nodes are installed...")
        if not self.verify_pulid_nodes():
            raise RuntimeError("ERROR: PuLID custom nodes not found! Check installation and ComfyUI logs.")
        
        print("SUCCESS: Setup complete - PuLID nodes are available")
        self.setup_complete = True
        
    def verify_build_time_models(self) -> bool:
        """Verify that build-time models were downloaded correctly"""
        required_models = [
            "/src/ComfyUI/models/vae/ae.safetensors",
            "/src/ComfyUI/models/clip/clip_l.safetensors", 
            "/src/ComfyUI/models/insightface/models/antelopev2/inswapper_128.onnx"
        ]
        
        missing_models = []
        for model_path in required_models:
            if not os.path.exists(model_path):
                missing_models.append(model_path)
        
        if missing_models:
            print(f"ERROR: Missing build-time models: {missing_models}")
            return False
        
        print("SUCCESS: All build-time models found")
        return True
        
    def verify_pulid_nodes(self) -> bool:
        """Verify that PuLID custom nodes are properly installed"""
        try:
            # Check if PuLID nodes are available
            required_nodes = [
                "PulidFluxModelLoader",
                "PulidFluxInsightFaceLoader", 
                "PulidFluxEvaClipLoader",
                "ApplyPulidFlux"
            ]
            
            # Get available nodes from ComfyUI
            all_nodes = self.get_available_nodes()
            
            # Check for required nodes
            found_nodes = [node for node in required_nodes if node in all_nodes]
            
            if len(found_nodes) < len(required_nodes):
                print(f"ERROR: Missing PuLID nodes! Required: {required_nodes}")
                print(f"* PuLID-related nodes found: {[n for n in all_nodes if 'pulid' in n.lower() or 'PuLID' in n or 'Pulid' in n]}")
                print(f"* Total ComfyUI nodes available: {len(all_nodes)}")
                return False
            
            print(f"SUCCESS: All required PuLID nodes found: {found_nodes}")
            return True
            
        except Exception as e:
            print(f"ERROR: Error checking PuLID nodes: {e}")
            return False
            
    def get_available_nodes(self) -> List[str]:
        """Get list of available ComfyUI nodes"""
        try:
            if not self.is_server_running():
                self.start_server()
                
            response = requests.get(f"http://{self.server_address}/object_info")
            if response.status_code == 200:
                return list(response.json().keys())
            return []
        except Exception as e:
            print(f"ERROR: Could not get available nodes: {e}")
            return []
            
    def restart_server(self):
        """Restart ComfyUI server"""
        print("* Restarting ComfyUI server...")
        if self.server_process and self.server_process.poll() is None:
            self.server_process.terminate()
            self.server_process.wait()
        self.start_server()
        
    def start_server(self):
        """Start ComfyUI server"""
        comfyui_path = "/src/ComfyUI"
        
        print(f"* Checking custom nodes directory: {comfyui_path}/custom_nodes")
        custom_nodes_dir = Path(f"{comfyui_path}/custom_nodes")
        if custom_nodes_dir.exists():
            for item in custom_nodes_dir.iterdir():
                if item.is_dir():
                    print(f"  * Found custom node directory: {item.name}")
                    # Check for Python files
                    py_files = list(item.glob("*.py"))
                    if py_files:
                        print(f"    * Python files: {[f.name for f in py_files[:5]]}")  # Show first 5
                        
        # Check models directory
        models_dir = Path(f"{comfyui_path}/models")
        print(f"* Checking models directory: {models_dir}")
        if models_dir.exists():
            for subdir in models_dir.iterdir():
                if subdir.is_dir():
                    files = list(subdir.glob("*"))
                    print(f"  * {subdir.name}: {[f.name for f in files[:3]]}")  # Show first 3
                    
        print("* Starting ComfyUI server...")
        
        # Start ComfyUI server
        self.server_process = subprocess.Popen(
            [sys.executable, "main.py", "--listen", "127.0.0.1", "--port", "8188"],
            cwd=comfyui_path,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        print("* Waiting for ComfyUI server to start...")
        
        # Wait for server to start
        max_attempts = 30
        for attempt in range(max_attempts):
            time.sleep(2)
            try:
                response = requests.get(f"http://{self.server_address}/system_stats", timeout=5)
                if response.status_code == 200:
                    break
            except:
                pass
                
            # Check if process is still running
            if self.server_process.poll() is not None:
                print("SUCCESS: ComfyUI server is running")
                break
                
        # Show some server output for debugging
        if self.server_process.poll() is None:
            # Server is running, capture some output
            time.sleep(1)
            try:
                # Non-blocking read of stdout
                import select
                if select.select([self.server_process.stdout], [], [], 0) == ([self.server_process.stdout], [], []):
                    lines = []
                    while len(lines) < 10:  # Read up to 10 lines
                        line = self.server_process.stdout.readline()
                        if not line:
                            break
                        lines.append(line.strip())
                        print(f"* ComfyUI: {line.strip()}")
            except:
                pass
                
        # Final check
        if self.server_process.poll() is not None:
            print("ERROR: ComfyUI server failed to start")
            print("* Server logs:")
            try:
                stdout, stderr = self.server_process.communicate(timeout=5)
                print(stdout)
                print(stderr)
            except:
                pass
            raise RuntimeError("ComfyUI server failed to start")
            
        print("SUCCESS: ComfyUI server is responding")
        
    def is_server_running(self) -> bool:
        """Check if ComfyUI server is running"""
        try:
            response = requests.get(f"http://{self.server_address}/system_stats", timeout=5)
            return response.status_code == 200
        except:
            return False
            
    def debug_server_state(self):
        """Debug server state and available nodes"""
        try:
            # Get all available nodes
            all_nodes = self.get_available_nodes()
            print(f"* Total nodes available: {len(all_nodes)}")
            
            # Check for PuLID related nodes
            pulid_related = [n for n in all_nodes if 'pulid' in n.lower() or 'PuLID' in n or 'Pulid' in n]
            print(f"* PuLID related nodes found: {pulid_related}")
            
            # Check for expected nodes
            expected_nodes = ["PulidFluxModelLoader", "PulidFluxInsightFaceLoader", "PulidFluxEvaClipLoader", "ApplyPulidFlux"]
            found_expected = [n for n in expected_nodes if n in all_nodes]
            print(f"SUCCESS: Expected PuLID nodes found: {found_expected}")
            
            if len(found_expected) == len(expected_nodes):
                print(f"* PuLID custom nodes loaded successfully: {found_expected}")
            else:
                print("WARNING: Expected PuLID custom nodes not found")
                print("* Available nodes sample:", all_nodes[:20])  # Show first 20 nodes
                
            # Check for other relevant nodes
            flux_nodes = [n for n in all_nodes if 'flux' in n.lower() or 'Flux' in n]
            print(f"* Flux related nodes: {flux_nodes}")
            
            # Check UNET nodes
            unet_nodes = [n for n in all_nodes if 'unet' in n.lower() or 'UNET' in n]
            print(f"* UNET related nodes: {unet_nodes}")
            
            # Show recent server output
            print("* Recent server output:")
            if self.server_process and self.server_process.stdout:
                # Try to read recent output
                pass
                
        except Exception as e:
            print(f"ERROR: Error checking ComfyUI server: {e}")
            print("* Server logs:")
            if self.server_process:
                try:
                    stdout, stderr = self.server_process.communicate(timeout=2)
                    print(stdout[-1000:])  # Show last 1000 chars
                    print(stderr[-1000:])
                except:
                    pass

    def predict(
        self,
        prompt: str = Input(description="Text prompt for image generation", default="A beautiful portrait of a person"),
        negative_prompt: str = Input(description="Negative prompt", default="blurry, low quality, distorted"),
        reference_image: CogPath = Input(description="Reference image for face preservation with PuLID"),
        width: int = Input(description="Width of the generated image", default=1024, ge=512, le=2048),
        height: int = Input(description="Height of the generated image", default=1024, ge=512, le=2048),
        steps: int = Input(description="Number of inference steps", default=25, ge=1, le=50),
        cfg: float = Input(description="Classifier-free guidance scale", default=7.0, ge=1.0, le=20.0),
        guidance: float = Input(description="Guidance scale for Flux", default=3.5, ge=0.0, le=10.0),
        pulid_weight: float = Input(description="Weight for PuLID face preservation", default=0.9, ge=0.0, le=2.0),
        controlnet_strength: float = Input(description="ControlNet strength for depth control", default=0.6, ge=0.0, le=2.0),
        seed: int = Input(description="Random seed for reproducibility", default=-1),
        enable_face_swap: bool = Input(description="Enable ReActor face swap", default=True)
    ) -> CogPath:
        """Run a single prediction on the model"""
        
        # Ensure async setup is complete
        import asyncio
        if not self.setup_complete:
            asyncio.create_task(self.setup_async())
            
        # Set random seed
        if seed == -1:
            seed = int(time.time())
            
        # Ensure ComfyUI server is running
        if not self.is_server_running():
            self.start_server()
            
        # Debug server state
        self.debug_server_state()
        
        # Load workflow
        with open(self.workflow_path, 'r') as f:
            workflow = json.load(f)
        
        print("SUCCESS: Using Flux+PuLID+Depth workflow")
        
        # Read and encode the reference image
        with open(reference_image, 'rb') as f:
            reference_image_data = f.read()
            reference_image_base64 = base64.b64encode(reference_image_data).decode('utf-8')
        
        # Update workflow with user inputs
        workflow = self.update_workflow_inputs(
            workflow,
            prompt=prompt,
            negative_prompt=negative_prompt,
            reference_image=reference_image,
            reference_image_base64=reference_image_base64,
            width=width,
            height=height,
            steps=steps,
            cfg=cfg,
            guidance=guidance,
            pulid_weight=pulid_weight,
            controlnet_strength=controlnet_strength,
            seed=seed,
            enable_face_swap=enable_face_swap
        )
        
        # Queue the workflow
        try:
            response = requests.post(
                f"http://{self.server_address}/prompt",
                json={"prompt": workflow}
            )
            response.raise_for_status()
            
            prompt_data = response.json()
            prompt_id = prompt_data['prompt_id']
            
            # Wait for completion
            while True:
                time.sleep(1)
                
                # Check status
                history_response = requests.get(f"http://{self.server_address}/history/{prompt_id}")
                if history_response.status_code == 200:
                    history = history_response.json()
                    if prompt_id in history:
                        break
                        
            # Get the generated image
            history_data = history[prompt_id]
            
            # Find the output image
            output_images = []
            for node_id, node_data in history_data['outputs'].items():
                if 'images' in node_data:
                    for image_data in node_data['images']:
                        filename = image_data['filename']
                        subfolder = image_data.get('subfolder', '')
                        
                        # Download the image
                        image_url = f"http://{self.server_address}/view"
                        params = {'filename': filename}
                        if subfolder:
                            params['subfolder'] = subfolder
                            
                        img_response = requests.get(image_url, params=params)
                        if img_response.status_code == 200:
                            # Save to temporary file
                            output_path = f"/tmp/output_{int(time.time())}.png"
                            with open(output_path, 'wb') as f:
                                f.write(img_response.content)
                            return CogPath(output_path)
                            
            raise RuntimeError("No output image found")
            
        except Exception as e:
            print(f"Error during prediction: {e}")
            raise
        
    def update_workflow_inputs(self, workflow, **kwargs):
        """Update the Flux+PuLID+Depth workflow with user inputs"""
        
        # Complete workflow structure from flux pulid + depth (3).json
        workflow_structure = {
            "33": {
                "inputs": {
                    "filename_prefix": "ComfyUI",
                    "images": ["32", 0]
                },
                "class_type": "SaveImage",
                "_meta": {
                    "title": "Save Image"
                }
            },
            "32": {
                "inputs": {
                    "swap_model": "inswapper_128.onnx",
                    "det_model": "retinaface_resnet50",
                    "save_original": False,
                    "CodeFormer_fidelity": 0.5,
                    "source_image": ["25", 0],
                    "input_image": ["30", 0],
                    "enabled": kwargs.get('enable_face_swap', True)
                },
                "class_type": "ReActorFaceSwap",
                "_meta": {
                    "title": "ReActor * Fast Face Swap"
                }
            },
            "30": {
                "inputs": {
                    "samples": ["29", 0],
                    "vae": ["4", 0]
                },
                "class_type": "VAEDecode",
                "_meta": {
                    "title": "VAE Decode"
                }
            },
            "29": {
                "inputs": {
                    "noise": ["3", 0],
                    "guider": ["28", 0],
                    "sampler": ["16", 0],
                    "sigmas": ["17", 0],
                    "latent_image": ["5", 0]
                },
                "class_type": "SamplerCustomAdvanced",
                "_meta": {
                    "title": "SamplerCustomAdvanced"
                }
            },
            "28": {
                "inputs": {
                    "model": ["27", 0],
                    "conditioning": ["26", 0]
                },
                "class_type": "CFGGuider",
                "_meta": {
                    "title": "CFGGuider"
                }
            },
            "27": {
                "inputs": {
                    "model": ["21", 0],
                    "pulid": ["20", 0],
                    "eva_clip": ["19", 0],
                    "face_analysis": ["18", 0],
                    "image": ["25", 0],
                    "weight": kwargs.get('pulid_weight', 0.9),
                    "start_at": 0,
                    "end_at": 1
                },
                "class_type": "ApplyPulidFlux",
                "_meta": {
                    "title": "Apply PuLID Flux"
                }
            },
            "26": {
                "inputs": {
                    "guidance": kwargs.get('guidance', 3.5),
                    "conditioning": ["11", 0]
                },
                "class_type": "FluxGuidance",
                "_meta": {
                    "title": "FluxGuidance"
                }
            },
            "25": {
                "inputs": {
                    "image": kwargs.get('reference_image_base64', ''),
                    "upload": "image"
                },
                "class_type": "LoadImage",
                "_meta": {
                    "title": "Load Image"
                }
            },
            "24": {
                "inputs": {
                    "positive": ["11", 0],
                    "negative": ["12", 0],
                    "control_net": ["23", 0],
                    "image": ["22", 0],
                    "strength": kwargs.get('controlnet_strength', 0.6),
                    "start_percent": 0,
                    "end_percent": 1
                },
                "class_type": "ControlNetApplyAdvanced",
                "_meta": {
                    "title": "Apply ControlNet (Advanced)"
                }
            },
            "23": {
                "inputs": {
                    "control_net_name": "flux-depth-controlnet-v3.safetensors"
                },
                "class_type": "ControlNetLoader",
                "_meta": {
                    "title": "Load ControlNet"
                }
            },
            "22": {
                "inputs": {
                    "image": ["25", 0],
                    "resolution": 512
                },
                "class_type": "DepthAnythingV2",
                "_meta": {
                    "title": "Depth Anything V2"
                }
            },
            "21": {
                "inputs": {
                    "positive": ["24", 0],
                    "negative": ["24", 1],
                    "control_net": ["24", 2],
                    "model": ["1", 0]
                },
                "class_type": "ControlNetApplyAdvanced",
                "_meta": {
                    "title": "Apply ControlNet (Advanced)"
                }
            },
            "20": {
                "inputs": {
                    "pulid_file": "pulid_flux_v0.9.0.safetensors"
                },
                "class_type": "PulidFluxModelLoader",
                "_meta": {
                    "title": "PuLID Flux Model Loader"
                }
            },
            "19": {
                "inputs": {
                    "eva_clip_path": "EVA02_CLIP_L_336_psz14_s6B.pt"
                },
                "class_type": "PulidFluxEvaClipLoader",
                "_meta": {
                    "title": "PuLID Flux EVA-CLIP Loader"
                }
            },
            "18": {
                "inputs": {
                    "provider": "CPU"
                },
                "class_type": "PulidFluxInsightFaceLoader",
                "_meta": {
                    "title": "PuLID Flux InsightFace Loader"
                }
            },
            "17": {
                "inputs": {
                    "model": ["1", 0],
                    "steps": kwargs.get('steps', 25)
                },
                "class_type": "BasicScheduler",
                "_meta": {
                    "title": "BasicScheduler"
                }
            },
            "16": {
                "inputs": {
                    "sampler_name": "euler"
                },
                "class_type": "KSamplerSelect",
                "_meta": {
                    "title": "KSamplerSelect"
                }
            },
            "12": {
                "inputs": {
                    "text": kwargs.get('negative_prompt', ''),
                    "clip": ["10", 0]
                },
                "class_type": "CLIPTextEncode",
                "_meta": {
                    "title": "CLIP Text Encode (Prompt)"
                }
            },
            "11": {
                "inputs": {
                    "text": kwargs.get('prompt', ''),
                    "clip": ["10", 0]
                },
                "class_type": "CLIPTextEncode",
                "_meta": {
                    "title": "CLIP Text Encode (Prompt)"
                }
            },
            "10": {
                "inputs": {
                    "clip_name1": "t5/google_t5-v1_1-xxl_encoderonly-fp8_e4m3fn.safetensors",
                    "clip_name2": "clip_l.safetensors",
                    "type": "flux"
                },
                "class_type": "DualCLIPLoader",
                "_meta": {
                    "title": "DualCLIPLoader"
                }
            },
            "5": {
                "inputs": {
                    "width": kwargs.get('width', 1024),
                    "height": kwargs.get('height', 1024),
                    "batch_size": 1
                },
                "class_type": "EmptyLatentImage",
                "_meta": {
                    "title": "Empty Latent Image"
                }
            },
            "4": {
                "inputs": {
                    "vae_name": "ae.safetensors"
                },
                "class_type": "VAELoader",
                "_meta": {
                    "title": "Load VAE"
                }
            },
            "3": {
                "inputs": {
                    "noise_seed": kwargs.get('seed', int(time.time()))
                },
                "class_type": "RandomNoise",
                "_meta": {
                    "title": "RandomNoise"
                }
            },
            "1": {
                "inputs": {
                    "unet_name": "flux1-dev-fp8.safetensors",
                    "weight_dtype": "fp8_e4m3fn"
                },
                "class_type": "UNETLoader",
                "_meta": {
                    "title": "Load Diffusion Model"
                }
            }
        }
        
        print("* Updating Flux+PuLID+Depth workflow inputs")
        
        # Update the workflow structure with the new inputs
        workflow.update(workflow_structure)
        
        return workflow
        
    async def download_large_models(self):
        """Download large models at runtime to minimize Docker image size"""
        
        models_to_download = [
            {
                "url": "https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/flux1-dev-fp8.safetensors",
                "path": "/src/ComfyUI/models/diffusion_models/flux1-dev-fp8.safetensors",
                "size": "~12GB"
            },
            {
                "url": "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp8_e4m3fn.safetensors",
                "path": "/src/ComfyUI/models/clip/t5/google_t5-v1_1-xxl_encoderonly-fp8_e4m3fn.safetensors",
                "size": "~9GB"
            },
            {
                "url": "https://huggingface.co/Kijai/flux-fp8/resolve/main/flux-depth-controlnet-v3.safetensors",
                "path": "/src/ComfyUI/models/controlnet/FLUX/flux-depth-controlnet-v3.safetensors",
                "size": "~3GB"
            },
            {
                "url": "https://huggingface.co/guozinan/PuLID/resolve/main/pulid_flux_v0.9.0.safetensors",
                "path": "/src/ComfyUI/models/pulid/pulid_flux_v0.9.0.safetensors",
                "size": "~900MB"
            },
            {
                "url": "https://huggingface.co/TencentARC/GFPGAN/resolve/main/GFPGANv1.4.pth",
                "path": "/src/ComfyUI/models/facerestore_models/GFPGANv1.4.pth",
                "size": "~300MB"
            },
            {
                "url": "https://huggingface.co/QuanSun/EVA-CLIP/resolve/main/EVA02_CLIP_L_336_psz14_s6B.pt",
                "path": "/src/ComfyUI/models/clip/EVA02_CLIP_L_336_psz14_s6B.pt",
                "size": "~850MB"
            }
        ]
        
        for model in models_to_download:
            model_path = Path(model['path'])
            if not model_path.exists():
                print(f"* Downloading {model['size']} model: {model_path.name}")
                
                try:
                    # Create directory if needed
                    model_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    # Download with wget
                    result = subprocess.run([
                        'wget', '-O', str(model_path), model['url']
                    ], capture_output=True, text=True)
                    
                    if result.returncode != 0:
                        print(f"ERROR: Download failed: {result.stderr}")
                        raise RuntimeError(f"Failed to download {model_path.name}")
                        
                    print(f"SUCCESS: Downloaded {model_path.name}")
                except Exception as e:
                    print(f"ERROR: Error downloading {model_path.name}: {e}")
                    raise
            else:
                print(f"SUCCESS: Large model already exists: {model_path.name}")
    
    def __del__(self):
        """Clean up resources"""
        if hasattr(self, 'server_process'):
            self.server_process.terminate()
            self.server_process.wait() 