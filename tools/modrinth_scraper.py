import io
import zipfile
import json
import requests
import base64
import os
import threading
from PIL import Image  # Import the Image module from PIL

# Ensure the dataset directory exists
os.makedirs('./dataset', exist_ok=True)

# Prepare JSON files

# Open the file in append mode with proper handling
ModrinthFile = open("modrinth.csv", "a+")

# Set the search parameters
facets = json.dumps([
    ["project_type:mod"],
    ["client_side:required"],
    ["server_side:required"]
])

def ModrinthSearch(offset):
    print(f"Scraping Offset {offset}")
    SearchParams = {
        "query": "",
        "limit": 100,
        "index": "newest",
        "offset": offset,
        "facets": facets
    }

    # Make a request to the API
    response = requests.get("https://api.modrinth.com/v2/search", params=SearchParams)

    # Check if the response is successful and contains JSON data
    if response.status_code == 200:
        try:
            response_json = response.json()
        except json.JSONDecodeError:
            print(f"Failed to decode JSON at offset {offset}: {response.text}")
            return  # Exit the function if decoding fails
    else:
        print(f"Failed request at offset {offset}: {response.status_code} - {response.text}")
        return  # Exit the function if the request fails

    # Process each item in the response
    for item in response_json.get("hits", []):
        project_id = item["project_id"]
        blocks_data = []
        items_data = []

        # Check if project_id already exists in the file
        ModrinthFile.seek(0)
        existing_ids = [line.split(",")[0] for line in ModrinthFile.readlines()]
        if project_id in existing_ids:
            continue

        # Fetch version information
        versions_response = requests.get(f"https://api.modrinth.com/v2/project/{item['project_id']}/version")
        if versions_response.status_code == 200:
            versions = versions_response.json()
        else:
            print(f"Failed to fetch versions for project {project_id}: {versions_response.status_code}")
            continue

        if versions:
            version_url = versions[0]['files'][0]['url']
            response = requests.get(version_url)

            # Open and process the ZIP file
            try:

                with io.BytesIO(response.content) as jar_file:
                    with zipfile.ZipFile(jar_file) as zip_file:
                        for file_info in zip_file.infolist():
                            # Check if the file is a PNG and if it's in the relevant directories
                            if file_info.filename.endswith(".png"):
                                if "items" in file_info.filename or "item" in file_info.filename:
                                    subdir = 'items'
                                elif "blocks" in file_info.filename or "block" in file_info.filename:
                                    subdir = 'blocks'
                                else:
                                    continue


                                # Read the PNG file
                                with zip_file.open(file_info) as png_file:
                                    png_data = png_file.read()

                                # Check the size of the image
                                try:
                                    with Image.open(io.BytesIO(png_data)) as img:
                                        if img.size != (16, 16):
                                            continue  # Skip images that are not 16x16
                                except:
                                    continue  # Skip images that cannot be opened

                                # Encode the PNG file to base64
                                png_base64 = base64.b64encode(png_data).decode('utf-8')

                                # Prepare data entry
                                data_entry = {
                                    "file_name": os.path.basename(file_info.filename),
                                    "project_id": project_id,
                                    "png_file_base64": png_base64
                                }

                                # Store the data entry in the appropriate list
                                if subdir == 'items':
                                    items_data.append(data_entry)
                                elif subdir == 'blocks':
                                    blocks_data.append(data_entry)

            except Exception as e:
                print(f"Error processing file {file_info.filename}: {e}")

        # Append the data to the JSON files
        if len(items_data) > 0:
            #print(f"Found {len(items_data)} items")
            try:
                with open('./dataset/items.json', 'r+') as items_file:
                    try:
                        items_data_existing = json.load(items_file)
                    except json.JSONDecodeError:
                        items_data_existing = []
                    items_data_existing.extend(items_data)
                    items_file.seek(0)
                    json.dump(items_data_existing, items_file, indent=4)
                    items_file.truncate()
            except FileNotFoundError:
                with open('./dataset/items.json', 'w') as items_file:
                    json.dump(items_data, items_file, indent=4)

        if len(blocks_data) > 0:
            #print(f"Found {len(blocks_data)} blocks")
            try:
                with open('./dataset/blocks.json', 'r+') as blocks_file:
                    try:
                        blocks_data_existing = json.load(blocks_file)
                    except json.JSONDecodeError:
                        blocks_data_existing = []
                    blocks_data_existing.extend(blocks_data)
                    blocks_file.seek(0)
                    json.dump(blocks_data_existing, blocks_file, indent=4)
                    blocks_file.truncate()
            except FileNotFoundError:
                with open('./dataset/blocks.json', 'w') as blocks_file:
                    json.dump(blocks_data, blocks_file, indent=4)

        if len(items_data) + len(blocks_data) > 0:
            ModrinthFile.write(f"{item['project_id']},{item['project_type']},{item['slug'].replace(',', '.')},{item['author'].replace(',', '.')},{len(items_data)},{len(blocks_data)},{item['license'].replace(',', '.')},{version_url},{offset}\n")

    # Check if there are more pages
    if len(response_json["hits"]) > 0:
        ModrinthSearch(offset + 100)

ModrinthSearch(0)
