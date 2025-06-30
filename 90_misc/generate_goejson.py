import sys
import json
from pathlib import Path
from ultralytics import YOLO
from pyproj import Transformer

sys.path.append(
    "/workspaces/sip-deliverables/0_preprocessing/")
try:
    from split import split_tif_file
except ImportError:
    print("Error importing split module. Ensure the path is correct and the module exists.")


def main():
    transformer = Transformer.from_crs(
        "EPSG:2056", "EPSG:4326", always_xy=True)
    trees = []
    geojson = {
        "type": "FeatureCollection",
        "name": "Detected Trees",
        "features": []
    }

    model = YOLO(
        "/workspaces/sip-deliverables/10_data/models/experiments/model_architecture/data_subset-1__enable_threshold-False__threshold-0__split_ratio-80__epochs-500__model_name-yolo11n__optimizer-auto__augment-True__run_number-3.pt"
    )

    split_tif_file(
        file=Path("/workspaces/sip-deliverables/10_data/input/470555.tif"),
        output_dir=Path("/workspaces/sip-deliverables/10_data/output/"),
        chunk_size=(640, 640),
        overlap_factor=0,
    )

    for img_path in Path("/workspaces/sip-deliverables/10_data/output/470555").glob("**/*.jpg"):
        results = model.predict(source=img_path, conf=0.75)

        print("results", results)

        trees.extend(results[0].boxes.xywh.cpu().numpy().tolist())

        for i, box in enumerate(results[0].boxes.xywh.cpu().numpy().tolist()):
            x = box[0]
            y = box[1]

            x_corr = (int(img_path.stem.split("_")[1].replace(
                "x", "")) + x) * 0.05 + 2747000.0
            y_corr = (int(img_path.stem.split("_")[3].replace(
                "y", "")) + y) * -0.05 + 1256000.0

            longitude, latitude = transformer.transform(x_corr, y_corr)

            geojson["features"].append({
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [longitude, latitude]
                },
                "properties": {
                    "id": len(geojson) + 1,
                    "confidence": results[0].boxes.conf.cpu().numpy().tolist()[i],
                    "model_name": "yolo11n",
                },
            })

    print(trees)
    print(geojson)

    with open("/workspaces/sip-deliverables/90_misc/detected_trees_75.geojson", "w") as f:
        json.dump(geojson, f, indent=4)


if __name__ == "__main__":
    main()
