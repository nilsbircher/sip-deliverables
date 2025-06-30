from pathlib import Path


def subset_filter(all_files: list[Path], _: Path, partial: float) -> list[Path]:
    """
    Subset the files based on the partial ratio.
    """

    print(f"Total files: {len(all_files)}")
    if partial < 1:
        num_partial = int(len(all_files) * partial)
        all_files = all_files[:num_partial]
        print(f"Partial files: {len(all_files)}")

    return all_files


def threshold_filter(
    all_files: list[Path], input_label_folder: Path, threshold: float
) -> list[Path]:
    """
    Subset the files based on the threshold.
    """

    print(f"Total files: {len(all_files)}")
    if threshold > 0:
        filtered_files = []

        for file in all_files:
            threshold_violation = False

            with open(input_label_folder / f"{file.stem}.txt", "r") as f:
                lines = f.readlines()

            if len(lines) == 0:
                threshold_violation = True

            for line in lines:
                width, height = line.split(" ")[-2:]
                if float(width) < threshold and float(height) < threshold:
                    threshold_violation = True
                    break

            if not threshold_violation:
                filtered_files.append(file)

        all_files = filtered_files
        print(f"Partial files: {len(filtered_files)}")

    return all_files


filter_function_lookup = {
    "subset_filter": subset_filter,
    "threshold_filter": threshold_filter,
}
