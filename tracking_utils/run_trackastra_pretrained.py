from trackastra.model import Trackastra
import zarr
import numpy as np
from pathlib import Path
from typing import List

device = "automatic"  # explicit choices: [cuda, mps, cpu]


def run_trackastra_pretrained(
    zarr_container: str,
    sequences: List[str],
    img_dataset_name: str,
    mask_dataset_name: str,
    output_csv_file_name,
    name: str = "general_2d",
    threshold: float = 0.01,
):
    output_csv_file_name = Path(output_csv_file_name)
    output_csv_file_name.parent.mkdir(parents=True, exist_ok=True)

    with open(output_csv_file_name, "w", encoding="utf-8") as f:
        f.write("#sequence id_u t_u id_v t_v association\n")

    model = Trackastra.from_pretrained(name, device=device)

    store = zarr.open(zarr_container, mode="r")

    for sequence_name in sequences:
        # C T Y X -> T Y X C
        imgs = np.moveaxis(np.array(store[sequence_name][img_dataset_name]), 0, -1)
        # 1 T Y X -> T Y X
        masks = np.array(store[sequence_name][mask_dataset_name])[0]
        predictions = model._predict(imgs, masks, edge_threshold=threshold)
        id_time_dictionary = {}
        for node in predictions["nodes"]:
            id_time_dictionary[node["id"]] = (int(node["time"]), int(node["label"]))

        for edge, association in predictions["weights"]:
            source, target = edge
            t_u, id_u = id_time_dictionary[source]
            t_v, id_v = id_time_dictionary[target]

            if t_v == t_u + 1:
                with open(output_csv_file_name, "a") as file:
                    file.write(
                        f"{sequence_name} {id_u} {t_u} {id_v} {t_v} {association}\n"
                    )
