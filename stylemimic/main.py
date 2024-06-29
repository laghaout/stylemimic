# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 21:38:36 2024

@author: Amine Laghaout
"""

import utilities as util
import wrangler as wra

env_vars = util.get_env_vars(("DATA_DIR_DOCKER", "DATA_DIR_LOCAL", "AUTHOR"))

data_params = dict(
    data_dir=env_vars["DATA_DIR"],
    # files=('Part 1 Ghettokungen.txt',),
    nrows=200,  # 200
    shuffle_seed=42,
    author=env_vars["AUTHOR"],
    system=dict(
        prose2beat="Du är en skrivassistent som sammanfattar de viktigaste delarna i en text.",
        beat2prose=f"Du är skrivassistent. När du får en sammanfattning av en scen eller av en berättelse utvecklar du den sammanfattningen till en fullständig scen eller berättelse på 500 ord. Texten du skriver imiterar {env_vars['AUTHOR']}s skrivstil.",
    ),
    user=dict(
        beat2prose=f"Skriv en berättelse på cirka 500 ord i {env_vars['AUTHOR']}s skrivstil baserat på följande sammanfattning.\n\nSammanfattning: "
    ),
)


oneoff_wrangler = wra.OneOffWrangler(**data_params)
oneoff_wrangler()
