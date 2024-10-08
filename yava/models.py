"""

    Access to models through LiteLLM Router

"""

import google.auth

from litellm import Router


_, project_id = google.auth.default()

model_list = [{
    "model_name": "gemini-1.5-flash", # model alias 
    "litellm_params": { 
        "model": "vertex_ai/gemini-1.5-flash-001", # actual model name
        "vertex_project": project_id,
        "vertex_location": "us-central1"
    }
},
{
    "model_name": "gemini-1.5-flash", # model alias 
    "litellm_params": { 
        "model": "vertex_ai/gemini-1.5-flash-001", # actual model name
        "vertex_project": project_id,
        "vertex_location": "us-east5"
    }
}]

flash = Router(model_list=model_list)
