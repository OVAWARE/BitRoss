import json
import base64
import os
import requests
from PIL import Image
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import math
import time
import io
import uuid

# Logging setup
logging.basicConfig(
    filename="error.log",
    level=logging.ERROR,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# Globals
image_size = 1024
processed_files = 0
file_count = 0
write_lock = threading.Lock()
stop_display = threading.Event()


# Encode in-memory image to base64
def encode_image_to_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


# Resize base64-decoded image
def resize_image(base64_data, size):
    image_data = base64.b64decode(base64_data)
    with Image.open(io.BytesIO(image_data)).convert("RGBA") as img:
        return img.resize(size, resample=Image.NEAREST)


# Save per-thread metadata
def write_metadata(file_name, project_id, description, keywords, thread_id):
    metadata = {
        "file_name": file_name,
        "project_id": project_id,
        "mod": "existing_mod",
        "tags": ", ".join(keywords),
        "description": description,
    }
    with open(f"./labeled/metadata_{thread_id}.json", "a") as f:
        json.dump(metadata, f)
        f.write("\n")


# Describe image using local Ollama model
def describe_image(file_name, project_id, base64_data):
    global processed_files
    try:
        resized_image = resize_image(base64_data, (image_size, image_size))
        encoded_image = encode_image_to_base64(resized_image)

        prompt = (
            f"Describe this image in one sentence and a set of keywords. As a hint, the filename is {file_name}. "
            f"Be descriptive and focus on the contents. Do not mention 'minecraft', 'pixel art', 'blocky style', or 'low-resolution style'.\n"
            f"Respond strictly in JSON format:\n"
            f"\ninterface Answer {{\n  description: string;\n  keywords: string[];\n}}\n"
        )

        payload = {
            "model": "gemma3:12b-it-qat",
            "prompt": prompt,
            "images": [encoded_image],
            "stream": False,
        }

        response = requests.post("http://localhost:11434/api/generate", json=payload)
        response.raise_for_status()

        full_output = json.loads(response.text)["response"]

        json_start = full_output.find("{")
        json_end = full_output.rfind("}") + 1
        response_data = json.loads(full_output[json_start:json_end])

        with write_lock:
            processed_files += 1

        return response_data
    except Exception as e:
        logging.error(f"Error describing {file_name}: {e}")
        return None


# Handle a group of image items
def process_image_group(items):
    thread_id = uuid.uuid4()
    for item in items:
        file_name = item.get("file_name")
        project_id = item.get("project_id")
        base64_data = item.get("png_file_base64")

        if not file_name or not base64_data:
            continue

        description_data = describe_image(file_name, project_id, base64_data)
        if description_data:
            try:
                write_metadata(
                    file_name,
                    project_id,
                    description_data.get("description", ""),
                    description_data.get("keywords", []),
                    thread_id,
                )
            except Exception as e:
                logging.error(f"Failed writing metadata for {file_name}: {e}")
    return str(thread_id)


lastUpdateTime = time.time()
lastAmount = 0


# Periodic display update
def update_display():
    global processed_files, file_count
    while not stop_display.is_set():
        with write_lock:
            processed = processed_files - lastAmount
            remaining = file_count - processed
            elapsed = time.time() - lastUpdateTime

            avg_time_per_image = (elapsed / processed) if processed > 0 else 0
            remaining_seconds = int(remaining * avg_time_per_image)

            days, rem = divmod(remaining_seconds, 86400)
            hours, rem = divmod(rem, 3600)
            minutes, seconds = divmod(rem, 60)
            print(
                f"Images processed: {processed_files}/{file_count} | Progress: {processed_files / file_count * 100:.2f}% | Estimated time remaining: {days}d {hours:02}h {minutes:02}m {seconds:02}s",
                end="\r",
            )
        time.sleep(1)


# Merge metadata files
def combine_metadata_files(thread_ids):
    combined_metadata = []
    for thread_id in thread_ids:
        with open(f"./labeled/metadata_{thread_id}.json", "r") as f:
            for line in f:
                combined_metadata.append(json.loads(line))
    with open("./labeled/metadata.json", "w") as f:
        for metadata in combined_metadata:
            json.dump(metadata, f)
            f.write("\n")
    for thread_id in thread_ids:
        os.remove(f"./labeled/metadata_{thread_id}.json")


# Entry point
def main():
    global file_count

    os.makedirs("./labeled", exist_ok=True)

    with open("./dataset/items.json", "r") as f:
        items = json.load(f)

    file_count = len(items)
    num_groups = min(4, file_count)
    group_size = math.ceil(file_count / num_groups)
    item_groups = [items[i : i + group_size] for i in range(0, len(items), group_size)]

    display_thread = threading.Thread(target=update_display)
    display_thread.start()

    thread_ids = []
    with ThreadPoolExecutor(max_workers=num_groups) as executor:
        futures = [executor.submit(process_image_group, group) for group in item_groups]
        for future in as_completed(futures):
            thread_ids.append(future.result())

    stop_display.set()
    display_thread.join()

    combine_metadata_files(thread_ids)


if __name__ == "__main__":
    main()
