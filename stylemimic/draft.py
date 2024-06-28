# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 15:56:57 2024

@author: Amine Laghaout
"""

from openai import OpenAI
import pandas as pd

client = OpenAI()

prompts = pd.Series(
    [
        "He came home in a hurry as if it was on fire. I can only imagine how bad it must have been.",
        "No wonder why he is always late. His alarm clock doesn't work",
        "King Bernadotte accessed the throne of Sweden after many battles.",
    ]
)


def get_chatgpt_response(prompt):
    try:
        completion = client.chat.completions.create(
            model="gpt-4o",
            # response_format={ "type": "json_object" },
            temperature=0.2,
            max_tokens=500,
            seed=42,
            messages=[
                {"role": "system", 
                 "content": "You are a writing assistant that re-writes texts to sound as if they were written by the classical authors of the 17th and 18th century."},
                {"role": "user", 
                 "content": prompt}
            ],
        )

        return completion
        # return completion.choices[0].message.content.strip()

    except Exception as e:
        return str(e)

# Display the results
results = pd.DataFrame(
    {
        "prompt": prompts,
        "completion_object": prompts.apply(get_chatgpt_response),
    }
)
results["response"] = results["completion_object"].apply(
    lambda x: x.choices[0].message.content.strip())
print(results)
