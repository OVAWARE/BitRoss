import glob
import json
import base64
import os
import requests
from PIL import Image
from openai import OpenAI
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import math
import time
import io
import shutil
import uuid

# Initialize OpenAI client
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="sk-or-v1-XXXX",
)
logging.basicConfig(filename='error.log', level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

# Pixel-art items are typically 16x16; upscaling to 1024 burns tokens/cost
# without improving captions. 128 is plenty for Gemini Flash vision.
image_size = 128
total_cost = 0.0
average_cost = 0.0
processed_files = 0
file_count = 0

# Lock for thread-safe writing to shared resources
write_lock = threading.Lock()

# Event to signal the display thread to stop
stop_display = threading.Event()

def encode_image_to_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def resize_image(file_path, size):
    with Image.open(file_path).convert('RGBA') as img:
        resized_image = img.resize(size, resample=Image.NEAREST)
        return resized_image

def write_metadata(file_name, description, keywords, thread_id):
    metadata = {
        "file_name": file_name,
        "mod": "existing_mod",
        "tags": ", ".join(keywords),
        "description": description
    }
    with open(f'./labeled/metadata_{thread_id}.json', 'a') as f:
        json.dump(metadata, f)
        f.write('\n')

def describe_image(file):
    global total_cost, processed_files
    resized_image = resize_image(file, (image_size, image_size))
    encoded_image = encode_image_to_base64(resized_image)
    response = client.chat.completions.create(
        extra_headers={
            "HTTP-Referer": "PixelCNN",
            "X-Title": "PixelCNN",
        },
        model="google/gemini-flash-1.5",
        messages=[
            {
                "role": "system",
                "content": "Describe this image in one sentence as well as through keywords. Be descriptive. Focus on the contents of the image, not presentation. Don't mention \"minecraft\", \"pixel art\", \"blocky style\", \"low-resolution style\".\n\nRespond in JSON format:\n```\ninterface Answer {\n  description: string;\n  keywords: string[];\n}\n```"
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Write the JSON, Note the items name is {file.split('/')[-1].split('.')[0].replace('_', ' ')}. Some of these may not be meaningful (e.g. name of the resource pack itself)"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{encoded_image}",
                            "detail": "auto"
                        }
                    },
                ]
            }
        ],
        response_format={"type": "json_object"},
        temperature=0
    )

    generation_id = response.id
    headers = {
        "Authorization": f"Bearer {client.api_key}"
    }
    generation_response = requests.get(
        f"https://openrouter.ai/api/v1/generation?id={generation_id}",
        headers=headers
    ).json()

    try:
        cost = generation_response['data']['total_cost']
        with write_lock:
            total_cost += cost
            processed_files += 1

        response_data = response.choices[0].message.content
        return json.loads(response_data), cost
    except:
        logging.error(f"Error processing file: {file} {response}")
        shutil.move(file, './error/')
        return None, 0

def process_image_group(file_group):
    thread_id = uuid.uuid4()
    for file in file_group:
        description_data, cost = describe_image(file)
        try:
            description = description_data.get("description", "")
            keywords = description_data.get("keywords", [])
            write_metadata(os.path.basename(file), description, keywords, thread_id)
            shutil.move(file, './labeled/')
        except:
            logging.error(f"Error processing file: {file}")
            shutil.move(file, './error/')
            continue
    return str(thread_id)

def update_display():
    global total_cost, average_cost, processed_files, file_count
    while not stop_display.is_set():
        os.system('cls' if os.name == 'nt' else 'clear')
        with write_lock:
            if processed_files > 0:
                average_cost = total_cost / processed_files
            print(f"Total cost so far: ${total_cost:.10f}")
            print(f"Average cost per image: ${average_cost:.10f}")
            print(f"Total images processed: {processed_files}/{file_count}")
            print(f"Total expected price: ${file_count * average_cost:.10f}")
            print(f"Progress: {processed_files/file_count*100:.2f}%")
        time.sleep(1)  # Update every second

def combine_metadata_files(thread_ids):
    combined_metadata = []
    for thread_id in thread_ids:
        with open(f'./labeled/metadata_{thread_id}.json', 'r') as f:
            for line in f:
                combined_metadata.append(json.loads(line))
    
    with open('./labeled/metadata.json', 'w') as f:
        for metadata in combined_metadata:
            json.dump(metadata, f)
            f.write('\n')
    
    # Clean up individual thread metadata files
    for thread_id in thread_ids:
        os.remove(f'./labeled/metadata_{thread_id}.json')

def main():
    global file_count, total_cost, average_cost, processed_files

    files = glob.glob('./items/*.png')
    file_count = len(files)

    # Number of groups (adjust this value as needed)
    num_groups = 50
    group_size = math.ceil(file_count / num_groups)

    # Split files into groups
    file_groups = [files[i:i + group_size] for i in range(0, len(files), group_size)]

    # Start the display update thread
    display_thread = threading.Thread(target=update_display)
    display_thread.start()

    # Process groups in parallel
    thread_ids = []
    with ThreadPoolExecutor(max_workers=num_groups) as executor:
        futures = [executor.submit(process_image_group, group) for group in file_groups]

        for future in as_completed(futures):
            thread_ids.append(future.result())

    # Stop the display update thread
    stop_display.set()
    display_thread.join()

    # Combine metadata files
    combine_metadata_files(thread_ids)

    # Final display update
    update_display()

if __name__ == "__main__":
    main()
