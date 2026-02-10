
import sys, signal, base64, re, io, json, time
from io import BytesIO
from pathlib import Path
from typing import Dict
import subprocess
import requests
import gradio as gr
from PIL import Image
import os 

from app import MAPPER, save_run

api_key = API_KEY = "jina_981e851b2dee47ba834256269776c26dF2quAX8oYUwN-8M7_jQapBtD-9As"

MAPPER.model.set_api_key(api_key)

inputs = [
    ("A group of cyclists riding nearby the ocean", "https://cdn.duvine.com/wp-content/uploads/2016/04/17095703/Slides_mallorca_FOR-WEB.jpg"),
    ("Computer Science jobs in USA", "https://www.ayresassociates.com/wp-content/uploads/2019/02/Career-Expo-Pie-Chart-Crop.jpg"),
    ("Graph of profession choices in Bangladesh", "https://notepadacademy.com/wp-content/uploads/2023/08/image.png")
]

for input in inputs:
    img_proc, *_ = MAPPER.process_image(input[1])
    toks, maps = MAPPER.get_token_similarity_maps(input[0], img_proc)
    save_run(input[0], input[1], img_proc, maps)

