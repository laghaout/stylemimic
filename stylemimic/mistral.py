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

jobs = client.jobs.list()
print(jobs)
