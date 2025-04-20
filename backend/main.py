# from fastapi import FastAPI, HTTPException, Request
# from fastapi.responses import HTMLResponse, JSONResponse
# from fastapi.templating import Jinja2Templates
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel, Field
# import os
# from dotenv import load_dotenv
# import openai
# from typing import Optional, List
# import logging
#from pathlib import Path

# # Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # Load environment variables
# load_dotenv()

# # Initialize FastAPI app
# app = FastAPI(title="AI Recipe Assistant")

# # Add CORS middleware
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Setup templates
#current_dir = Path(__file__).parent
#templates_path = current_dir / "templates"
#templates =  Jinja2Templates(directory=str(templates_path))

# # Configure OpenAI
# # openai.api_key = os.getenv("OPENAI_API_KEY")
# # if not openai.api_key:
# #     raise ValueError("OPENAI_API_KEY environment variable is not set")

# class RecipeRequest(BaseModel):
#     query: str = Field(..., min_length=1, description="The recipe to generate")
#     diet_preference: Optional[str] = Field(None, description="Dietary preference (e.g., vegetarian, vegan)")
#     cuisine_type: Optional[str] = Field(None, description="Type of cuisine (e.g., Italian, Mexican)")

#     class Config:
#         schema_extra = {
#             "example": {
#                 "query": "chocolate chip cookies",
#                 "diet_preference": "vegetarian",
#                 "cuisine_type": "italian"
#             }
#         }

# class LearningResource(BaseModel):
#     title: str
#     url: str
#     type: str

# class RecipeResponse(BaseModel):
#     recipe: str
#     image_url: str
#     learning_resources: List[LearningResource]

# # def generate_recipe(query: str, diet_preference: Optional[str] = None, cuisine_type: Optional[str] = None) -> dict:
# #     logger.info(f"Generating recipe for query: {query}, diet: {diet_preference}, cuisine: {cuisine_type}")
    
# #     if not query:
# #         raise HTTPException(status_code=400, detail="Recipe query is required")

# #     # Create a detailed prompt for the recipe
# #     prompt = f"""Create a detailed recipe for {query}"""
# #     if diet_preference:
# #         prompt += f" that is {diet_preference}"
# #     if cuisine_type:
# #         prompt += f" in {cuisine_type} style"
    
# #     prompt += """\n\nFormat the recipe in markdown with the following sections:
# #     1. Brief Description
# #     2. Ingredients (as a bulleted list)
# #     3. Instructions (as numbered steps)
# #     4. Tips (as a bulleted list)
# #     5. Nutritional Information (as a bulleted list)
    
# #     Use markdown formatting like:
# #     - Headers (###)
# #     - Bold text (**)
# #     - Lists (- and 1.)
# #     - Sections (>)
# #     """

# #     try:
# #         logger.info(f"Sending prompt to OpenAI: {prompt}")
        
# #         # Generate recipe text
# #         completion = openai.chat.completions.create(
# #             model="gpt-3.5-turbo",
# #             messages=[
# #                 {"role": "system", "content": "You are a professional chef who provides detailed recipes with ingredients, instructions, nutritional information, and cooking tips. Format your responses in markdown."},
# #                 {"role": "user", "content": prompt}
# #             ],
# #             temperature=0.7
# #         )
# #         recipe_text = completion.choices[0].message.content
# #         logger.info("Successfully generated recipe text")

# #         # Generate recipe image
# #         logger.info("Generating recipe image")
# #         image_response = openai.images.generate(
# #             model="dall-e-3",
# #             prompt=f"Professional food photography of {query}, appetizing, high-quality, restaurant style",
# #             n=1,
# #             size="1024x1024"
# #         )
# #         image_url = image_response.data[0].url
# #         logger.info("Successfully generated recipe image")

# #         # Get learning resources
# #         learning_resources = get_learning_resources(query)
# #         logger.info("Successfully generated learning resources")

# #         response_data = {
# #             "recipe": recipe_text,
# #             "image_url": image_url,
# #             "learning_resources": learning_resources
# #         }
        
# #         return response_data
# #     except Exception as e:
# #         logger.error(f"Error generating recipe: {str(e)}")
# #         raise HTTPException(status_code=500, detail=str(e))
# def generate_recipe(query: str, diet_preference: Optional[str] = None, cuisine_type: Optional[str] = None) -> dict:
#     logger.info(f"Generating mock recipe for query: {query}, diet: {diet_preference}, cuisine: {cuisine_type}")

#     mock_recipe = f"""
#     ### {query.title()} Recipe

#     > **A quick and easy mock recipe!**

#     #### Ingredients
#     - 1 cup flour
#     - 2 eggs
#     - 1/2 cup milk
#     - Salt to taste

#     #### Instructions
#     1. Mix all ingredients.
#     2. Cook on medium heat.
#     3. Serve hot.

#     #### Tips
#     - Use fresh ingredients.
#     - Adjust salt as per taste.

#     #### Nutritional Info
#     - Calories: ~200
#     - Protein: 5g
#     - Carbs: 30g
#     """

#     mock_image_url = "https://via.placeholder.com/600x400.png?text=Recipe+Image"

#     mock_learning_resources = [
#         {
#             "title": "Mock Cooking Basics",
#             "url": "https://example.com/mock-cooking",
#             "type": "video"
#         },
#         {
#             "title": "Mock Recipe Tips",
#             "url": "https://example.com/mock-tips",
#             "type": "article"
#         }
#     ]

#     return {
#         "recipe": mock_recipe,
#         "image_url": mock_image_url,
#         "learning_resources": mock_learning_resources
#     }

# def get_learning_resources(recipe_name: str) -> list:
#     return [
#         {
#             "title": f"Master the Art of {recipe_name}",
#             "url": f"https://cooking-school.example.com/learn/{recipe_name.lower().replace(' ', '-')}",
#             "type": "video"
#         },
#         {
#             "title": f"Tips and Tricks for Perfect {recipe_name}",
#             "url": f"https://recipes.example.com/tips/{recipe_name.lower().replace(' ', '-')}",
#             "type": "article"
#         }
#     ]

# @app.post("/recipe", response_model=RecipeResponse)
# async def get_recipe(request: RecipeRequest):
#     logger.info(f"Received recipe request: {request}")
#     try:
#         result = generate_recipe(request.query, request.diet_preference, request.cuisine_type)
#         logger.info("Successfully generated recipe response")
#         return result
#     except Exception as e:
#         logger.error(f"Error processing recipe request: {str(e)}")
#         return JSONResponse(
#             status_code=500,
#             content={"detail": str(e)}
#         )

# @app.get("/", response_class=HTMLResponse)
# async def root(request: Request):
#     return templates.TemplateResponse("index.html", {"request": request})

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8080)
# import os
# import requests
# from fastapi import FastAPI, Request, HTTPException
# from fastapi.responses import JSONResponse, HTMLResponse
# from fastapi.staticfiles import StaticFiles
# from fastapi.templating import Jinja2Templates
# from pydantic import BaseModel
# from typing import Optional
# from dotenv import load_dotenv

# # Load environment variables
# load_dotenv()

# app = FastAPI()

# # Setup static + templates
# os.makedirs("static", exist_ok=True)
# os.makedirs("templates", exist_ok=True)

# app.mount("/static", StaticFiles(directory="static"), name="static")
# templates = Jinja2Templates(directory="templates")

# # Hugging Face config
# TEXT_MODEL = "facebook/bart-large-cnn"
# HF_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

# class RecipeRequest(BaseModel):
#     ingredients: str
#     diet: Optional[str] = None
#     cuisine: Optional[str] = None

# @app.get("/", response_class=HTMLResponse)
# async def home(request: Request):
#     return templates.TemplateResponse("index.html", {"request": request})

# @app.post("/api/generate-recipe")
# async def generate_recipe(request: RecipeRequest):
#     try:
#         # 👨‍🍳 Smart Prompt
#         prompt = f"""
#         You are a professional chef and recipe writer.
#         Create a full, detailed cooking recipe using the following ingredients: {request.ingredients}.
#         {f"Make sure it is suitable for a {request.diet} diet." if request.diet else ""}
#         {f"The recipe should follow {request.cuisine} cuisine style." if request.cuisine else ""}

#         Format the response in markdown with the following sections:
#         ### Title
#         ### Description
#         ### Ingredients (as a bulleted list)
#         ### Instructions (as numbered steps)
#         ### Tips
#         ### Nutritional Information (if possible)

#         Be friendly and helpful in tone.
#         """

#         # Debug logs
#         print("📤 Prompt Sent:", prompt)
#         print("🧠 Model:", TEXT_MODEL)

#         headers = {"Authorization": f"Bearer {HF_API_KEY}"}
#         payload = {
#             "inputs": prompt,
#             "parameters": {
#                 "max_new_tokens": 250,
#                 "temperature": 0.8,
#                 "do_sample": True
#             }
#         }

#         # Send to Hugging Face
#         response = requests.post(
#             f"https://api-inference.huggingface.co/models/{TEXT_MODEL}",
#             headers=headers,
#             json=payload,
#             timeout=30
#         )

#         # Try JSON parse or show raw text
#         try:
#             result = response.json()
#         except Exception as e:
#             print("❌ Could not parse JSON. Raw response:")
#             print(response.text)
#             raise HTTPException(status_code=500, detail="Invalid response from Hugging Face API")

#         print("✅ HF JSON Response:", result)

#         # Handle errors or loading message
#         if "error" in result:
#             raise HTTPException(status_code=503, detail=result["error"])
#         if isinstance(result, dict) and "generated_text" in result:
#             generated = result["generated_text"]
#         elif isinstance(result, list) and "generated_text" in result[0]:
#             generated = result[0]["generated_text"]
#         else:
#             raise HTTPException(status_code=500, detail="No recipe generated by the model.")

#         return {
#             "recipe": generated.strip(),
#             "image_url": "/static/placeholder.jpg"
#         }

#     except Exception as e:
#         print(f"🔥 Error: {str(e)}")
#         raise HTTPException(status_code=500, detail=str(e))


# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
import os
import requests
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import Optional
from dotenv import load_dotenv

load_dotenv()
app = FastAPI()

# Setup directories
os.makedirs("static", exist_ok=True)
os.makedirs("templates", exist_ok=True)

app.mount("/static", StaticFiles(directory="backend/templates/static"), name="static")
# templates = Jinja2Templates(directory="templates")
templates = Jinja2Templates(directory="backend/templates")  # Now points to GitHub Pages dir

# More reliable model that fits within free tier
TEXT_MODEL = "mistralai/Mixtral-8x7B-Instruct-v0.1"  # Smaller model that works reliably
HF_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

class RecipeRequest(BaseModel):
    ingredients: str
    diet: Optional[str] = None
    cuisine: Optional[str] = None

# def generate_fallback_recipe(ingredients: str) -> str:
#     """Generate a simple recipe when API fails"""
#     return f"""### {ingredients} Recipe

# ### Ingredients
# - {ingredients.replace(',', '\n-')}

# ### Instructions
# SORRY MODEL IS CURRENTLY NOT ACCEPTING RESPONSES TRY THIS-
# 1. Wash and prepare all ingredients
# 2. Combine in a mixing bowl
# 3. Cook/blend as needed
# 4. Season to taste
# 5. Serve and enjoy!

# ### Tips
# - Add spices according to your preference
# - Adjust consistency with water or milk"""
def generate_fallback_recipe(ingredients: str) -> str:
    """Generate a simple recipe when API fails"""
    # First process the ingredients list separately
    ingredients_list = ingredients.replace(',', '\n-')
    
    return f"""### {ingredients} Recipe

### Ingredients
- {ingredients_list}

### Instructions
SORRY MODEL IS CURRENTLY NOT ACCEPTING RESPONSES TRY THIS-
1. Wash and prepare all ingredients
2. Combine in a mixing bowl
3. Cook/blend as needed
4. Season to taste
5. Serve and enjoy!

### Tips
- Add spices according to your preference
- Adjust consistency with water or milk"""

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/api/generate-recipe")
async def generate_recipe(request: RecipeRequest):
    try:
        # First check if API key exists
        if not HF_API_KEY:
            return {
                "recipe": generate_fallback_recipe(request.ingredients),
                "image_url": "/static/placeholder.jpg"
            }

        prompt = f"Create a detailed recipe using: {request.ingredients}. "
        if request.diet:
            prompt += f"({request.diet} diet) "
        if request.cuisine:
            prompt += f"({request.cuisine} style). "
        prompt += "Include ingredients list,how to prepare and step-by-step instructions."

        headers = {"Authorization": f"Bearer {HF_API_KEY}"}
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 300,
                "temperature": 0.7
            }
        }

        try:
            response = requests.post(
                f"https://api-inference.huggingface.co/models/{TEXT_MODEL}",
                headers=headers,
                json=payload,
                timeout=10
            )
            
            # Check if response is HTML (error page)
            if "text/html" in response.headers.get("content-type", ""):
                raise HTTPException(status_code=503, detail="Hugging Face API is currently unavailable")
                
            result = response.json()
            
            if response.status_code != 200:
                error_msg = result.get("error", "API request failed")
                raise HTTPException(status_code=response.status_code, detail=error_msg)

            recipe_text = result[0].get("generated_text", "")
            
            # Ensure we got a valid recipe
            if not recipe_text or "sorry" in recipe_text.lower() or "can't" in recipe_text.lower():
                raise ValueError("Invalid recipe generated")

            return {
                "recipe": recipe_text,
                "image_url": "/static/placeholder.jpg"
            }

        except requests.exceptions.RequestException as e:
            print(f"API Request failed: {str(e)}")
            return {
                "recipe": generate_fallback_recipe(request.ingredients),
                "image_url": "/static/placeholder.jpg"
            }

    except Exception as e:
        print(f"Error: {str(e)}")
        return {
            "recipe": generate_fallback_recipe(request.ingredients),
            "image_url": "/static/placeholder.jpg"
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
