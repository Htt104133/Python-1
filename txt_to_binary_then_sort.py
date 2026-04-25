import sys
import multiprocessing as mp
from pathlib import Path
from array import array

from external_binary_sort_u32 import external_parallel_sort, ARRAY_TYPE


MAX_UINT32 = 2 ** 32 - 1


def make_binary_path(txt_path: Path) -> Path:
    return txt_path.with_name(txt_path.stem + "_binary.bin")


def write_buffer(file_obj, buffer) -> None:
    a = array(ARRAY_TYPE, buffer)

    if sys.byteorder != "little":
        a.byteswap()

    a.tofile(file_obj)


def convert_txt_to_binary(txt_filename: str, max_numbers_in_memory: int) -> Path:
    txt_path = Path(txt_filename).resolve()

    if not txt_path.exists():
        raise FileNotFoundError(f"Input text file does not exist: {txt_path}")

    if not txt_path.is_file():
        raise ValueError(f"Input path is not a file: {txt_path}")

    if max_numbers_in_memory <= 0:
        raise ValueError("The second parameter must be a positive integer.")

    bin_path = make_binary_path(txt_path)

    if bin_path.exists():
        bin_path.unlink()

    buffer_limit = min(max_numbers_in_memory, 1_000_000)
    buffer = []
    count = 0

    with open(txt_path, "r", encoding="utf-8") as src, open(bin_path, "wb") as dst:
        for line_number, line in enumerate(src, start=1):
            s = line.strip()

            if not s:
                continue

            try:
                value = int(s)
            except ValueError:
                raise ValueError(f"Line {line_number} is not an integer: {s!r}")

            if not (0 <= value <= MAX_UINT32):
                raise ValueError(
                    f"Line {line_number}: {value} is outside unsigned 32-bit range "
                    f"0..{MAX_UINT32}."
                )

            buffer.append(value)
            count += 1

            if len(buffer) >= buffer_limit:
                write_buffer(dst, buffer)
                buffer.clear()

        if buffer:
            write_buffer(dst, buffer)

    print(f"Text file converted to binary: {bin_path}")
    print(f"Converted numbers: {count}")

    return bin_path


def main() -> None:
    if len(sys.argv) != 3:
        print("Usage:")
        print("  python txt_to_binary_then_sort.py <txt_file> <max_numbers_in_memory>")
        print()
        print("Example:")
        print("  python txt_to_binary_then_sort.py random_numbers.txt 1000000")
        sys.exit(1)

    txt_filename = sys.argv[1]

    try:
        max_numbers_in_memory = int(sys.argv[2])
    except ValueError:
        print("Error: second parameter must be an integer.")
        sys.exit(1)

    try:
        bin_path = convert_txt_to_binary(txt_filename, max_numbers_in_memory)
        external_parallel_sort(str(bin_path), max_numbers_in_memory)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    mp.freeze_support()
    main()