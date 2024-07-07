# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 15:32:27 2024

@author: amine
"""

from dotenv import load_dotenv
import os
from mistralai.client import MistralClient

load_dotenv()

client = MistralClient(api_key=os.getenv("MISTRAL_API_KEY"))

retrieved_jobs = client.jobs.retrieve("2efafdc4-2fdc-4c22-a76b-7383b4d654a1")
print(retrieved_jobs)
