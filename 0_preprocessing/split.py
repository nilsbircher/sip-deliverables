from pathlib import Path
from PIL import Image
import tifffile
from multiprocessing import Pool


def split_tif(
    input_dir: Path, output_dir: Path, chunk_size: tuple, overlap_factor: float
) -> None:
    file_list = [file for file in input_dir.glob("*.tif")]
    with Pool() as pool:
        pool.starmap(
            split_tif_file,
            [(file, output_dir, chunk_size, overlap_factor)
             for file in file_list],
        )
    pool.close()
    pool.join()


def split_tif_file(
    file: Path | str,
    output_dir: Path | str,
    chunk_size: tuple[int, int] | tuple[str, str],
    overlap_factor: float | str,
) -> None:
    if not isinstance(file, Path):
        file = Path(file)

    if not isinstance(output_dir, Path):
        output_dir = Path(output_dir)

    if not isinstance(chunk_size, tuple) or not all(
        isinstance(i, int) for i in chunk_size
    ):
        chunk_size = tuple(map(int, chunk_size))

    if not isinstance(overlap_factor, float):
        overlap_factor = float(overlap_factor)

    try:
        image = tifffile.imread(file)
    except Exception as e:
        print(f"Error reading TIFF file: {e}")
        return

    image_height, image_width = image.shape[-2:]

    height_start = 0
    while height_start < image_height:
        height_end = min(height_start + chunk_size[0], image_height)

        width_start = 0
        while width_start < image_width:
            width_end = min(width_start + chunk_size[1], image_width)

            chunk_image = Image.fromarray(
                # Remove A from RGBA, set array to height, width, channels instead of channels, height, width
                image[:3, height_start:height_end, width_start:width_end].transpose(
                    1, 2, 0
                )
            )

            output_path = output_dir / file.stem
            output_path.mkdir(parents=True, exist_ok=True)
            chunk_image.save(
                output_path
                / f"{file.stem}_x{width_start}_x{width_end}_y{height_start}_y{height_end}_{overlap_factor * 100:.0f}.jpg",
                "JPEG",
            )

            width_start += int(chunk_size[0] - chunk_size[0] * overlap_factor)
        height_start += int(chunk_size[1] - chunk_size[1] * overlap_factor)


def main():
    CHUNK_SIZE = 640, 640
    OVERLAP_FACTOR = 0
    cwd = Path(__file__).parent.parent

    print(f"Current working directory: {cwd}")

    input_dir = cwd / "10_data" / "input"
    output_dir = cwd / "10_data" / "output"

    split_tif(
        input_dir, output_dir, chunk_size=CHUNK_SIZE, overlap_factor=OVERLAP_FACTOR
    )

    print(f"Splitting completed. Output saved to {output_dir}")


if __name__ == "__main__":
    main()
