def extract_data_from_model_name(model_name: str, extractor: str) -> str | None:
    parts = model_name.split("__")
    for part in parts:
        if extractor in part:
            value = part.split("-")[-1]
            return value
    return None
