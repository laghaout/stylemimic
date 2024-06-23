# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 15:56:57 2024

@author: amine
"""

from openai import OpenAI
import pandas as pd

client = OpenAI()


def get_chatgpt_response(prompt):
    try:
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                # {"role": "system", "content": "You are a poetic assistant, skilled in explaining complex programming concepts with creative flair."},
                {"role": "user", "content": prompt}
            ],
        )

        return completion
        # return completion.choices[0].message.content.strip()

    except Exception as e:
        return str(e)


prompts = pd.Series(
    [
        "He came home in a hurry as if it was on fire. I can only imagine how bad it must have been.",
        "No wonder why he is always late. His alarm clock doesn't work",
        "King Bernadotte accessed the throne of Sweden after many battles.",
    ]
)

# Display the results
results = pd.DataFrame(
    {
        "prompt": prompts,
        "completion_object": prompts.apply(get_chatgpt_response),
    }
)
results["response"] = results.apply(
    lambda x: x.choices[0].message.content, axis=1
)
print(results)
