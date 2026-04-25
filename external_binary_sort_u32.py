import os
import sys
import heapq
import shutil
import struct
import tempfile
import multiprocessing as mp
from pathlib import Path
from array import array


INT_SIZE = 4
ARRAY_TYPE = "I"          # unsigned 32-bit integer
STRUCT_FORMAT = "<I"      # little-endian unsigned int32
PACKER = struct.Struct(STRUCT_FORMAT)
MAX_OPEN_FILES = 128


def check_environment() -> None:
    a = array(ARRAY_TYPE)
    if a.itemsize != INT_SIZE:
        raise RuntimeError("array('I') is not 4 bytes on this system.")


def make_output_path(input_path: Path) -> Path:
    return input_path.with_name(input_path.stem + "_sorted" + input_path.suffix)


def write_uint32_values(file_obj, values) -> None:
    a = array(ARRAY_TYPE, values)

    if sys.byteorder != "little":
        a.byteswap()

    a.tofile(file_obj)


def read_one_uint32(file_obj):
    data = file_obj.read(INT_SIZE)

    if not data:
        return None

    if len(data) != INT_SIZE:
        raise ValueError("Broken binary file: incomplete 32-bit integer.")

    return PACKER.unpack(data)[0]


def sort_chunk(task):
    input_filename, temp_dir, chunk_id, start_index, count = task

    temp_path = Path(temp_dir) / f"run_{chunk_id:09d}.bin"

    numbers = array(ARRAY_TYPE)

    with open(input_filename, "rb") as f:
        f.seek(start_index * INT_SIZE)
        numbers.fromfile(f, count)

    if len(numbers) != count:
        raise ValueError("Unexpected end of file while reading a chunk.")

    if sys.byteorder != "little":
        numbers.byteswap()

    values = numbers.tolist()
    del numbers

    values.sort()

    with open(temp_path, "wb") as out:
        write_uint32_values(out, values)

    return str(temp_path)


def create_tasks(input_path: Path, total_numbers: int, chunk_size: int, temp_dir: str):
    chunk_id = 0
    start = 0

    while start < total_numbers:
        count = min(chunk_size, total_numbers - start)
        yield (str(input_path), temp_dir, chunk_id, start, count)

        start += count
        chunk_id += 1


def merge_group(run_paths, output_path, max_numbers_in_memory: int) -> None:
    heap = []
    files = []

    try:
        for i, run_path in enumerate(run_paths):
            f = open(run_path, "rb")
            files.append(f)

            value = read_one_uint32(f)

            if value is not None:
                heapq.heappush(heap, (value, i))

        buffer_capacity = max_numbers_in_memory - len(run_paths)

        with open(output_path, "wb") as out:
            buffer = []

            while heap:
                value, file_index = heapq.heappop(heap)

                if buffer_capacity > 0:
                    buffer.append(value)

                    if len(buffer) >= buffer_capacity:
                        write_uint32_values(out, buffer)
                        buffer.clear()
                else:
                    out.write(PACKER.pack(value))

                next_value = read_one_uint32(files[file_index])

                if next_value is not None:
                    heapq.heappush(heap, (next_value, file_index))

            if buffer:
                write_uint32_values(out, buffer)

    finally:
        for f in files:
            f.close()


def merge_all_runs(run_paths, output_path: Path, temp_dir: str, max_numbers_in_memory: int) -> None:
    runs = list(run_paths)

    if not runs:
        open(output_path, "wb").close()
        return

    if len(runs) == 1:
        shutil.copyfile(runs[0], output_path)
        os.remove(runs[0])
        return

    if max_numbers_in_memory < 2:
        raise ValueError("Memory limit is too small: at least 2 numbers are needed.")

    round_id = 0

    while len(runs) > 1:
        if max_numbers_in_memory >= 3:
            fan_in = min(len(runs), MAX_OPEN_FILES, max_numbers_in_memory - 1)
        else:
            fan_in = 2

        new_runs = []
        group_id = 0

        for i in range(0, len(runs), fan_in):
            group = runs[i:i + fan_in]

            if len(group) == 1:
                new_runs.append(group[0])
                continue

            is_final_merge = len(runs) <= fan_in and i == 0

            if is_final_merge:
                merged_path = output_path
            else:
                merged_path = Path(temp_dir) / f"merge_{round_id:03d}_{group_id:06d}.bin"

            merge_group(group, merged_path, max_numbers_in_memory)

            for old_path in group:
                os.remove(old_path)

            new_runs.append(str(merged_path))
            group_id += 1

        runs = new_runs
        round_id += 1

    if Path(runs[0]) != output_path:
        shutil.copyfile(runs[0], output_path)
        os.remove(runs[0])


def external_parallel_sort(input_filename: str, max_numbers_in_memory: int) -> Path:
    check_environment()

    input_path = Path(input_filename).resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"Input file does not exist: {input_path}")

    if not input_path.is_file():
        raise ValueError(f"Input path is not a file: {input_path}")

    if max_numbers_in_memory <= 0:
        raise ValueError("The second parameter must be a positive integer.")

    file_size = input_path.stat().st_size

    if file_size % INT_SIZE != 0:
        raise ValueError("Binary file size must be divisible by 4 bytes.")

    total_numbers = file_size // INT_SIZE

    output_path = make_output_path(input_path)

    if output_path.exists():
        output_path.unlink()

    cpu_count = os.cpu_count() or 1

    if total_numbers > 0 and max_numbers_in_memory < 2 * cpu_count:
        raise ValueError(
            f"Memory limit is too small for strict parallel run. "
            f"Need at least {2 * cpu_count} numbers for {cpu_count} CPU cores."
        )

    chunk_size = max(1, max_numbers_in_memory // (2 * cpu_count))

    print(f"Input binary file: {input_path}")
    print(f"Output binary file: {output_path}")
    print(f"Total numbers: {total_numbers}")
    print(f"CPU cores used: {cpu_count}")
    print(f"Max numbers in memory: {max_numbers_in_memory}")
    print(f"Numbers per sorted chunk: {chunk_size}")

    with tempfile.TemporaryDirectory(
        prefix="external_sort_",
        dir=str(input_path.parent)
    ) as temp_dir:

        tasks = create_tasks(
            input_path=input_path,
            total_numbers=total_numbers,
            chunk_size=chunk_size,
            temp_dir=temp_dir
        )

        sorted_runs = []

        with mp.Pool(processes=cpu_count) as pool:
            for run_path in pool.imap_unordered(sort_chunk, tasks, chunksize=1):
                sorted_runs.append(run_path)

        merge_all_runs(
            run_paths=sorted_runs,
            output_path=output_path,
            temp_dir=temp_dir,
            max_numbers_in_memory=max_numbers_in_memory
        )

    print("Sorting finished.")
    print(f"Created file: {output_path}")

    return output_path


def main() -> None:
    if len(sys.argv) != 3:
        print("Usage:")
        print("  python external_binary_sort_u32.py <binary_file> <max_numbers_in_memory>")
        print()
        print("Example:")
        print("  python external_binary_sort_u32.py random_numbers_binary.bin 1000000")
        sys.exit(1)

    input_filename = sys.argv[1]

    try:
        max_numbers_in_memory = int(sys.argv[2])
    except ValueError:
        print("Error: second parameter must be an integer.")
        sys.exit(1)

    try:
        external_parallel_sort(input_filename, max_numbers_in_memory)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    mp.freeze_support()
    main()