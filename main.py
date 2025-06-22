from smolagents import Tool, CodeAgent, LiteLLMModel
import replicate
import json
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

class GenerateImageTool(Tool):
    name = "generate_image"
    description = "Generate image with AI model based on original cloth img & text prompt"
    inputs = {
        "original_cloth_img_url": {
            "type": "string",
            "description": "URL of the original clothing image"
        },
        "prompt": {
            "type": "string",
            "description": "Text prompt for image generation"
        }
    }
    output_type = "string"

    def forward(self, original_cloth_img_url: str, prompt: str) -> str:
        data = {
    "242": {
        "inputs": {
            "ckpt_name": "Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors"
        },
        "class_type": "CheckpointLoaderSimple",
        "_meta": {
            "title": "Load Checkpoint"
        }
    },
    "243": {
        "inputs": {
            "text": "hyperdetailed photography, soft light, head portrait, (white background:1.3), skin details, sharp and in focus, \ngirl chinese student, \nshort (blue: 1.4) wavey hair, \nbig eyes, \nnarrow nose, \nslim, \ncute, \nbeautiful",
            "clip": [
                "242",
                1
            ]
        },
        "class_type": "CLIPTextEncode",
        "meta": {
            "title": "CLIP Text Encode (Prompt)"
        }
    },
    "244": {
        "inputs": {
            "text": "(worst quality, low quality, normal quality, lowres, low details, oversaturated, undersaturated, overexposed, underexposed, grayscale, bw, bad photo, bad photography, bad art:1.4), (watermark, signature, text font, username, error, logo, words, letters, digits, autograph, trademark, name:1.2), (blur, blurry, grainy), morbid, ugly, asymmetrical, mutated malformed, mutilated, poorly lit, bad shadow, draft, cropped, out of frame, cut off, censored, jpeg artifacts, out of focus, glitch, duplicate, (airbrushed, cartoon, anime, semi-realistic, cgi, render, blender, digital art, manga, amateur:1.3), (3D, 3D Game, 3D Scene, 3D Character:1.1), acne",
            "clip": [
                "242",
                1
            ]
        },
        "class_type": "CLIPTextEncode",
        "meta": {
            "title": "CLIP Text Encode (Prompt)"
        }
    },
    "245": {
        "inputs": {
            "seed": 3829697690114223,
            "steps": 35,
            "cfg": 8,
            "sampler_name": "dpmpp_2m",
            "scheduler": "karras",
            "denoise": 1,
            "model": [
                "242",
                0
            ],
            "positive": [
                "243",
                0
            ],
            "negative": [
                "244",
                0
            ],
            "latent_image": [
                "246",
                0
            ]
        },
        "class_type": "KSampler",
        "_meta": {
            "title": "KSampler"
        }
    },
    "246": {
        "inputs": {
            "width": "1024",
            "height": "1024",
            "batch_size": 1
        },
        "class_type": "EmptyLatentImage",
        "meta": {
            "title": "Empty Latent Image"
        }
    },
    "247": {
        "inputs": {
            "samples": [
                "245",
                0
            ],
            "vae": [
                "242",
                2
            ]
        },
        "class_type": "VAEDecode",
        "_meta": {
            "title": "VAE Decode"
        }
    },
    "248": {
        "inputs": {
            "image": "$248-0",
            "images": [
                "247",
                0
            ]
        },
        "class_type": "PreviewBridge",
        "meta": {
            "title": "Preview Bridge (Image)"
        }
    },
    "250": {
        "inputs": {
            "resolution": 1024,
            "image": [
                "248",
                0
            ]
        },
        "class_type": "OneFormer-COCO-SemSegPreprocessor",
        "_meta": {
            "title": "OneFormer COCO Segmentor"
        }
    },
    "251": {
        "inputs": {
            "image": "$251-0",
            "images": [
                "248",
                0
            ]
        },
        "class_type": "PreviewBridge",
        "meta": {
            "title": "Preview Bridge (Image)"
        }
    },
    "252": {
        "inputs": {
            "image": "$252-0",
            "images": [
                "250",
                0
            ]
        },
        "class_type": "PreviewBridge",
        "meta": {
            "title": "Preview Bridge (Image)"
        }
    },
    "254": {
        "inputs": {
            "channel": "red",
            "image": [
                "250",
                0
            ]
        },
        "class_type": "ImageToMask",
        "_meta": {
            "title": "Convert Image to Mask"
        }
    },
    "256": {
        "inputs": {
            "ipadapter_file": "ip-adapter-faceid_sdxl.bin"
        },
        "class_type": "IPAdapterModelLoader",
        "_meta": {
            "title": "Load IPAdapter Model"
        }
    },
    "257": {
        "inputs": {
            "clip_name": "Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors"
        },
        "class_type": "CLIPVisionLoader",
        "_meta": {
            "title": "Load CLIP Vision"
        }
    },
    "260": {
        "inputs": {
            "ipadapter_file": "ip-adapter-plus-face_sdxl_vit-h.safetensors"
        },
        "class_type": "IPAdapterModelLoader",
        "_meta": {
            "title": "Load IPAdapter Model"
        }
    },
    "261": {
        "inputs": {
            "text": prompt,
            "clip": [
                "242",
                1
            ]
        },
        "class_type": "CLIPTextEncode",
        "_meta": {
            "title": "CLIP Text Encode (Prompt)"
        }
    },
    "262": {
        "inputs": {
            "text": "(worst quality, low quality, normal quality, lowres, low details, oversaturated, undersaturated, overexposed, underexposed, grayscale, bw, bad photo, bad photography, bad art:1.4), (watermark, signature, text font, username, error, logo, words, letters, digits, autograph, trademark, name:1.2), (blur, blurry, grainy), morbid, ugly, asymmetrical, mutated malformed, mutilated, poorly lit, bad shadow, draft, cropped, out of frame, cut off, censored, jpeg artifacts, out of focus, glitch, duplicate, (airbrushed, cartoon, anime, semi-realistic, cgi, render, blender, digital art, manga, amateur:1.3), (3D,3D Game, 3D Game Scene, 3D Character:1.1), (bad hands. bad anatomy, bad body. bad face. bad teeth, bad arms, bad legs. deformities:1.3)",
            "clip": [
                "242",
                1
            ]
        },
        "class_type": "CLIPTextEncode",
        "_meta": {
            "title": "CLIP Text Encode (Prompt)"
        }
    },
    "263": {
        "inputs": {
            "seed": 55003239682810,
            "steps": 35,
            "cfg": 8,
            "sampler_name": "dpmpp_2m",
            "scheduler": "karras",
            "denoise": 1,
            "model": [
                "277",
                0
            ],
            "positive": [
                "261",
                0
            ],
            "negative": [
                "262",
                0
            ],
            "latent_image": [
                "264",
                0
            ]
        },
        "class_type": "KSampler",
        "_meta": {
            "title": "KSampler"
        }
    },
    "264": {
        "inputs": {
            "width": "1024",
            "height": "1024",
            "batch_size": 1
        },
        "class_type": "EmptyLatentImage",
        "_meta": {
            "title": "Empty Latent Image"
        }
    },
    "265": {
        "inputs": {
            "samples": [
                "263",
                0
            ],
            "vae": [
                "242",
                2
            ]
        },
        "class_type": "VAEDecode",
        "_meta": {
            "title": "VAE Decode"
        }
    },
    "266": {
        "inputs": {
            "image": "$266-0",
            "images": [
                "265",
                0
            ]
        },
        "class_type": "PreviewBridge",
        "_meta": {
            "title": "Preview Bridge (Image)"
        }
    },
    "267": {
        "inputs": {
            "image": original_cloth_img_url,
            "upload": "image"
        },
        "class_type": "LoadImage",
        "_meta": {
            "title": "Load Image"
        }
    },
    "268": {
        "inputs": {
            "resolution": 1024,
            "image": [
                "267",
                0
            ]
        },
        "class_type": "UniFormer-SemSegPreprocessor",
        "_meta": {
            "title": "UniFormer Segmentor"
        }
    },
    "269": {
        "inputs": {
            "channel": "red",
            "image": [
                "268",
                0
            ]
        },
        "class_type": "ImageToMask",
        "_meta": {
            "title": "Convert Image to Mask"
        }
    },
    "270": {
        "inputs": {
            "ipadapter_file": "ip-adapter-plus_sdxl_vit-h.safetensors"
        },
        "class_type": "IPAdapterModelLoader",
        "_meta": {
            "title": "Load IPAdapter Model"
        }
    },
    "272": {
        "inputs": {
            "images": [
                "268",
                0
            ]
        },
        "class_type": "PreviewImage",
        "_meta": {
            "title": "Preview Image"
        }
    },
    "274": {
        "inputs": {
            "provider": "CPU"
        },
        "class_type": "InsightFaceLoader",
        "_meta": {
            "title": "Load InsightFace"
        }
    },
    "275": {
        "inputs": {
            "weight": 0.5,
            "noise": 0,
            "weight_type": "original",
            "start_at": 0,
            "end_at": 1,
            "faceid_v2": True,
            "weight_v2": 1.5,
            "unfold_batch": False,
            "ipadapter": [
                "256",
                0
            ],
            "clip_vision": [
                "257",
                0
            ],
            "insightface": [
                "274",
                0
            ]
        },
        "class_type": "IPAdapterApply",
        "meta": {
            "title": "Apply IPAdapter"
        }
    },
    "276": {
        "inputs": {
        "weight": 0.5,
        "noise": 0,
        "weight_type": "original",
        "start_at": 0,
        "end_at": 1,
        "faceid_v2": True,
        "weight_v2": 1.5,
        "unfold_batch": False,
        "ipadapter": [
            "260",
            0
        ],
        "clip_vision": [
            "257",
            0
        ],
        "model": [
            "242",
            0
        ]
        },
        "class_type": "IPAdapterApply",
        "_meta": {
        "title": "Apply IPAdapter"
        }
    },
    "277": {
        "inputs": {
            "weight": 0.5,
            "noise": 0,
            "weight_type": "original",
            "start_at": 0,
            "end_at": 1,
            "unfold_batch": False,
            "ipadapter": [
                "270",
                0
            ],
            "clip_vision": [
                "257",
                0
            ],
            "image": [
                "267",
                0
            ],
            "model": [
                "276",
                0
            ],
            "attn_mask": [
                "269",
                0
            ]
        },
        "class_type": "IPAdapterApply",
        "_meta": {
            "title": "Apply IPAdapter"
        }
    },
    "279": {
        "inputs": {
            "filename_prefix": "final",
            "images": [
                "265",
                0
            ]
        },
        "class_type": "SaveImage",
        "_meta": {
            "title": "Save Image"
        }
    }
        }
        json_string = json.dumps(data, indent=2)
        output = replicate.run(
            "stability-ai/sdxl:39ed52f2a78e934b3ba6e2a89f5b1c712de7dfea535525255b1aa35c5565e08b",
            input={
                "workflow_json": json_string,
                "prompt": prompt,
                "randomise_seeds": True,
                "return_temp_files": False
            }
        )
        if hasattr(output, 'url'): return output.url
        elif isinstance(output, list): return output[0]
        else: return str(output)

class ReviewImageTool(Tool):
    name = "review_image"
    description = "Compare and review original clothing image with AI generated image"
    inputs = {
        "original_cloth_img_url": {
            "type": "string",
            "description": "URL of the original clothing image"
        },
        "ai_generated_img_url": {
            "type": "string",
            "description": "URL of the AI generated image"
        },
        "text_prompt": {
            "type": "string",
            "description": "Original text prompt used for generation"
        }
    }
    output_type = "string"

    def forward(self, original_cloth_img_url: str, ai_generated_img_url: str, text_prompt: str) -> str:
        client = OpenAI()
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"""
original text prompt: {text_prompt}
1st img is the reference clothing item, 2nd img is an AI generated image of person wearing this clothing.

Compare the clothing in both images with extreme attention to detail. Analyze:

1. Core Style Elements (25% of evaluation):
- Exact garment type/category
- Overall silhouette and cut
- Basic structure (e.g., collar type, sleeve length, closure type)

2. Design Details (50% of evaluation):
- Colors (exact shade matches)
- Patterns/prints (precise pattern type, size, placement)
- Textures and materials
- All embellishments (buttons, zippers, stitching)
- Logos/branding (size, placement, color)
- Unique design features (pockets, panels, seams)

3. Fit & Construction (25% of evaluation):
- How it sits on the body
- Draping/folding behavior
- Proportions and measurements
- Construction details (seams, hems, cuffs)

Only mark as 95% match if ALL elements are identical. Even minor variations should reduce the score.

Output format:
MATCH SCORE: [X%]
- Include specific calculation breakdown based on the three categories above

DEVIATIONS:
- List every difference found, no matter how small
- Group by category (Style/Design/Fit)
- Include specific measurements/colors where possible

MISSING ELEMENTS:
- Detailed list of reference image features absent in generated image
- Include exact specifications (measurements, color codes, material types)

IMPROVED PROMPT:
[Original prompt] + [All missing specific details] organized as:
- Garment type & structure
- Materials & textures
- Colors & patterns
- Design details & embellishments
- Fit & proportions
""",
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": original_cloth_img_url,
                            },
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": ai_generated_img_url,
                            },
                        },
                    ],
                }
            ],
            max_tokens=1000,
        )
        return response.choices[0].message.content

class FixHandsTool(Tool):
    name = "fix_hands"
    description = "Fix hand distortions in AI generated images"
    inputs = {
        "ai_generated_img_url": {
            "type": "string",
            "description": "URL of the AI generated image"
        }
    }
    output_type = "string"

    def forward(self, ai_generated_img_url: str) -> str:
        output = replicate.run(
            "973398769/hands-restoration:721dd0c8bc13dcd514b8aa0d7530f814fc1f01caff8d5bc92b6ecc0856a7ad20",
            input={
                "input_file": ai_generated_img_url,
                "function_name": "hand_restoration",
                "randomise_seeds": True,
                "return_temp_files": False
            },
        )
        return f"iterated img with hand distortion fixed: {output}"

class UpscaleImageTool(Tool):
    name = "upscale_image"
    description = "Upscale image to improve quality"
    inputs = {
        "latest_ai_generated_img_url": {
            "type": "string",
            "description": "URL of the latest AI generated image"
        },
        "prompt": {
            "type": "string",
            "description": "Text prompt for upscaling"
        }
    }
    output_type = "string"

    def forward(self, latest_ai_generated_img_url: str, prompt: str) -> str:
        output = replicate.run(
            "juergengunz/ultimate-portrait-upscale:f7fdace4ec7adab7fa02688a160eee8057f070ead7fbb84e0904864fd2324be5",
            input={
                "cfg": 8,
                "image": latest_ai_generated_img_url,
                "steps": 20,
                "denoise": 0.1,
                "upscaler": "4x-UltraSharp",
                "mask_blur": 8,
                "mode_type": "Linear",
                "scheduler": "normal",
                "tile_width": 512,
                "upscale_by": 2,
                "tile_height": 512,
                "sampler_name": "euler",
                "tile_padding": 32,
                "seam_fix_mode": "None",
                "seam_fix_width": 64,
                "negative_prompt": "(worst quality, low quality, normal quality, lowres, low details, oversaturated, undersaturated, overexposed)",
                "positive_prompt": prompt,
                "seam_fix_denoise": 1,
                "seam_fix_padding": 16,
                "seam_fix_mask_blur": 8,
                "controlnet_strength": 1,
                "force_uniform_tiles": True,
                "use_controlnet_tile": True
            },
        )
        return f"upscaled img with fix: {output}"

generate_image_tool = GenerateImageTool()
review_image_tool = ReviewImageTool()
fix_hands_tool = FixHandsTool()
upscale_image_tool = UpscaleImageTool()

model = LiteLLMModel(
    model_id="gpt-4o-mini",
    api_key=os.environ["OPENAI_API_KEY"]
)

image_generator = CodeAgent(
    tools=[generate_image_tool, review_image_tool],
    model=model,
    max_steps=7,
    verbosity_level=2  
)

image_finetuner = CodeAgent(
    tools=[fix_hands_tool, upscale_image_tool],
    model=model,
    max_steps=2,
    verbosity_level=2  
)

def process_image_request(cloth_img_url: str, base_prompt: str) -> str:
    generation_result = image_generator.run(f"""
    Task: Generate and review a realistic photo based on a reference clothing image.
    
    Input:
    - Original cloth image: {cloth_img_url}
    - Base prompt: {base_prompt}
    
    Steps:
    1. Use generate_image tool to create initial image
    2. Use review_image tool to evaluate the match
    3. If not 95% match, iterate with new prompt
    4. Continue until 95% match is achieved
    
    Return the result in format: generated_image_url|||final_prompt.
    """)

    generated_img, final_prompt = generation_result.split("|||")
    
    final_result = image_finetuner.run(f"""
    Task: Enhance the quality of the generated image.
    
    Input image: {generated_img}
    Original prompt: {final_prompt}
    
    Steps:
    1. Use fix_hands tool to correct any hand distortions
    2. Use upscale_image tool to improve image quality
    
    Return the final enhanced image URL.
    """)
    return f"original_cloth_img: {cloth_img_url}, original_prompt: {base_prompt}, final_prompt: {final_prompt}, final_image_url: {final_result}"

if __name__ == "__main__":
    cloth_img = "https://media.istockphoto.com/id/830738010/photo/blue-baby-coat-child-fashion-wear-isolated-nobody.jpg?s=1024x1024&w=is&k=20&c=WYuBes-tOCAF5vXLh8RP7L_6w6a62TvFXopghhDVe7s="
    base_prompt = "woman with blue hair, in a blue cloth, in a cafe in paris"
    
    result = process_image_request(cloth_img, base_prompt)
    print("Final result:", result)