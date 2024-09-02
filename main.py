# -*- coding: utf-8 -*-
"""
Created on Sun Jun 30 17:23:14 2024

@author: amine
"""

import stylemimic.utilities as util
import stylemimic.wrangler as wra


def main(author: str | int):
    config = util.assemble_config(author)

    oneoffwrangler = wra.OneoffWrangler(**config["oneoffwrangler"])

    # Generate all the re-writes
    for k in oneoffwrangler.user_prompt.keys():
        oneoffwrangler(k)

    return oneoffwrangler


# %%
if __name__ == "__main__":
    oneoffwrangler = {k: main(k) for k in range(2)}
    oneoffwrangler = {v.author: v for _, v in oneoffwrangler.items()}
    data = {k: v.data for k, v in oneoffwrangler.items()}

# %%

# oneoffwrangler[author].disp_proses(1)
