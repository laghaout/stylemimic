# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 18:21:40 2024

@author: amine
"""

from dataclasses import dataclass

@dataclass
class AzureLocation:
    account_url: str
    container_name: str
    credential: str
    
A = AzureLocation('amine', 'laghaout', 42)    