def gen_run_name(params: dict) -> str:
    configs = []
    for key, value in params.items():
        if isinstance(value, float):
            configs.append(f"{key}-{int(value * 100)}")
        else:
            configs.append(f"{key}-{value}")

    return f"{'__'.join(configs)}__run_number-"
