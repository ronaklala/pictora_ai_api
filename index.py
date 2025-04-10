import json
import logging
import io
import os
import re
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import boto3
from PIL import Image  # Ensure Pillow is included in your Lambda Layer or deployment package
import decimal
from dotenv import load_dotenv
from boto3.dynamodb.conditions import Key

load_dotenv()

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")
CORS(app)
# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# AWS configurations
BUCKET_NAME = 'rekognition3103'
COLLECTION_ID = 'pictora_lala'

# Initialize AWS clients
rekognition_client = boto3.client(
    'rekognition',
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=os.getenv("AWS_REGION")
)

s3_client = boto3.client(
    's3',
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=os.getenv("AWS_REGION")
)

dynamodb = boto3.resource(
    'dynamodb',
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=os.getenv("AWS_REGION")
)

table = dynamodb.Table('FaceRecognitionTable')

def delete_collection():
    """Delete the Rekognition collection and remove all metadata from DynamoDB."""
    try:
        rekognition_client.delete_collection(CollectionId=COLLECTION_ID)
        logging.info(f"✅ Collection '{COLLECTION_ID}' deleted")
        return jsonify({'message': f'Collection {COLLECTION_ID} deleted successfully'}), 200
    except Exception as e:
        logging.error(f"❌ Error deleting collection: {str(e)}")
        return jsonify({'error': 'Failed to delete collection'}), 500

def ensure_collection_exists():
    """
    Check if the Rekognition collection exists; if not, create it.
    """
    try:
        rekognition_client.describe_collection(CollectionId=COLLECTION_ID)
        logging.info(f"Collection '{COLLECTION_ID}' exists.")
    except rekognition_client.exceptions.ResourceNotFoundException:
        logging.warning(f"Collection '{COLLECTION_ID}' not found. Creating it now...")
        rekognition_client.create_collection(CollectionId=COLLECTION_ID)
        logging.info(f"Collection '{COLLECTION_ID}' created successfully.")

def convert_to_decimal(obj):
    """Recursively convert float values to Decimal for DynamoDB."""
    if isinstance(obj, float):
        return decimal.Decimal(str(obj))  # Convert float to Decimal
    elif isinstance(obj, dict):
        return {k: convert_to_decimal(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_decimal(i) for i in obj]
    return obj


def get_last_noface_id():
    """Fetch the last NOFACE ID stored in DynamoDB, handling pagination."""
    noface_ids = []
    last_evaluated_key = None

    while True:
        # Perform scan with pagination support
        scan_params = {
            "FilterExpression": "begins_with(FaceID, :prefix)",
            "ExpressionAttributeValues": {":prefix": "NOFACE"}
        }
        if last_evaluated_key:
            scan_params["ExclusiveStartKey"] = last_evaluated_key  # Get next page

        response = table.scan(**scan_params)

        # Collect FaceIDs
        noface_ids.extend(item['FaceID'] for item in response.get('Items', []))

        # Check if there are more pages
        last_evaluated_key = response.get("LastEvaluatedKey")
        if not last_evaluated_key:
            break  # Stop if no more pages

    # Debugging: Print all retrieved NOFACE IDs
    print("Retrieved NOFACE IDs:", noface_ids)

    # If no NOFACE IDs found, return default
    if not noface_ids:
        return "NOFACE_0"

    # Extract numeric part and get max
    max_noface = max(noface_ids, key=lambda x: int(x.split("_")[1]))
    return max_noface

def increment_noface_id(last_noface_id):
    """Correctly increment NOFACE ID with debugging."""
    print("Incrementing from:", last_noface_id)

    if last_noface_id == "NOFACE_0":
        return "NOFACE_1"
    
    parts = last_noface_id.split("_")
    print("Split Parts:", parts)

    if len(parts) == 2 and parts[0] == "NOFACE":
        next_number = int(parts[1]) + 1
        new_id = f"NOFACE_{next_number}"
        print("Generated New ID:", new_id)
        return new_id
    
    print("Returning default NOFACE_0 due to unexpected format")
    return "NOFACE_0"

@app.route('/fetch-all-data', methods=['GET'])
def fetch_all_data_from_dynamodb():
    """
    Fetches all records from a DynamoDB table and returns them in JSON format.

    :param table_name: Name of the DynamoDB table
    :return: JSON-formatted data
    """
    try:

        # Scan the entire table (Note: Not recommended for very large tables)
        response = table.scan(ProjectionExpression="CategoryName")
        
        # Extract data
        categories = {item["CategoryName"] for item in response.get("Items", []) if "CategoryName" in item}

        # Handle pagination if there are more records
        while "LastEvaluatedKey" in response:
            response = table.scan(
                ProjectionExpression="CategoryName",
                ExclusiveStartKey=response["LastEvaluatedKey"]
            )
            categories.update(item["CategoryName"] for item in response.get("Items", []) if "CategoryName" in item)

        return json.dumps(list(categories), indent=2)  # Convert to JSON format

    except Exception as e:
        return json.dumps({"error": str(e)})  # Return error as JSON

@app.route('/delete_collection', methods=['GET'])
def delete_collection_route():
    """
    Delete the Rekognition collection and remove all metadata from DynamoDB.
    """
    return delete_collection() 


@app.route("/upload", methods=["POST"])
def upload_images():
     # Assuming this function is defined elsewhere
    ensure_collection_exists()  # Assuming this function is defined elsewhere

    # Get the uploaded file and category name from the request
    file = request.files['images']
    category_name = request.form.get('category_name', '')  # Get category name from frontend

    original_s3_key = f'uploads/{file.filename}'
    low_res_s3_key = f'uploads/low_res_{file.filename}'
    webp_s3_key = f'uploads/{os.path.splitext(file.filename)[0]}.webp'

    # Read the file into memory
    file_content = file.read()  # Read the entire file into memory
    file_buffer = io.BytesIO(file_content)  # Create a BytesIO buffer for Pillow and S3 operations

    # Process image: Create a low-resolution version first
    try:
        # Open the image using Pillow
        image = Image.open(file_buffer)

        # Create a low-resolution copy (e.g., 200x200)
        low_res_image = image.copy()
        low_res_image.thumbnail((720, 720))
        low_res_buffer = io.BytesIO()
        low_res_image.save(low_res_buffer, format='JPEG')  # Save as JPEG for low-res copy
        low_res_buffer.seek(0)

        # Upload low-resolution image to S3
        s3_client.put_object(Bucket=BUCKET_NAME, Key=low_res_s3_key, Body=low_res_buffer.getvalue(), ContentType='image/jpeg')
        socketio.emit('upload_progress', {'filename': file.filename, 'stage': 'Low-resolution uploaded'})
        logging.info(f"✅ Low-resolution image '{low_res_s3_key}' uploaded successfully.")

    except Exception as e:
        logging.error(f"❌ Error processing and uploading low-res image: {str(e)}")
        return jsonify({'error': 'Failed to process and upload low-res image'}), 500

    # Reset buffer position for original image upload
    file_buffer.seek(0)

    # Upload original image to S3
    try:
        s3_client.upload_fileobj(file_buffer, BUCKET_NAME, original_s3_key)
        socketio.emit('upload_progress', {'filename': file.filename, 'stage': 'Original uploaded'})
        logging.info(f"✅ Original image '{original_s3_key}' successfully uploaded to S3.")
        
    except Exception as e:
        logging.error(f"❌ Error uploading '{original_s3_key}' to S3: {str(e)}")
        return jsonify({'error': f'Failed to upload {original_s3_key} to S3'}), 500

    # Convert original image to WebP format and upload it
    try:
        webp_buffer = io.BytesIO()
        image.save(webp_buffer, format='WEBP')  # Convert to WebP format
        webp_buffer.seek(0)

        # Upload WebP image to S3
        s3_client.put_object(Bucket=BUCKET_NAME, Key=webp_s3_key, Body=webp_buffer.getvalue(), ContentType='image/webp')
        socketio.emit('upload_progress', {'filename': file.filename, 'stage': 'WebP uploaded'})
        logging.info(f"✅ WebP image '{webp_s3_key}' uploaded successfully.")

    except Exception as e:
        logging.error(f"❌ Error processing images: {str(e)}")
        return jsonify({'error': 'Failed to process images'}), 500

    # Index faces using Rekognition
    try:
        print(original_s3_key, COLLECTION_ID, BUCKET_NAME)
        response = rekognition_client.index_faces(
            CollectionId=COLLECTION_ID,
            Image={'S3Object': {'Bucket': BUCKET_NAME, 'Name': original_s3_key}},
            DetectionAttributes=['ALL']
        )

        
        
        logging.info(f"✅ Faces from '{original_s3_key}' indexed successfully in Rekognition.")

        face_id_to_store = None  # Default face ID if no faces are detected

        if response['FaceRecords']:
            for face_record in response['FaceRecords']:
                face_id_to_store = face_record['Face']['FaceId']  # Get the actual Face ID if detected

                bounding_box = convert_to_decimal(face_record['Face']['BoundingBox'])  # Convert to Decimal
                confidence = decimal.Decimal(str(face_record['Face']['Confidence']))  # Convert to Decimal

                table.put_item(Item={
                    'FaceID': face_id_to_store,
                    'ImageID': original_s3_key,
                    'ImageURL': f's3://{BUCKET_NAME}/{original_s3_key}',
                    'LowResImageURL': f's3://{BUCKET_NAME}/{low_res_s3_key}',  # Store URL of low-res image
                    'WebPImageURL': f's3://{BUCKET_NAME}/{webp_s3_key}',       # Store URL of WebP image
                    'BoundingBox': bounding_box,
                    'Confidence': confidence,
                    'CategoryName': category_name   # Store the category name in DynamoDB
                })
                socketio.emit('upload_progress', {'filename': file.filename, 'stage': 'Database updated'})
                logging.info(f"✅ Face ID '{face_id_to_store}' saved to DynamoDB with confidence {confidence}%.")
        
        else:
            # If no faces were detected, generate a new NOFACE ID
            last_noface_id = get_last_noface_id()
            print(last_noface_id)  # Function to get last NOFACE ID from DB
            print(increment_noface_id(last_noface_id))  # Function to increment NOFACE ID
            new_noface_id = increment_noface_id(last_noface_id)  # Increment it

            table.put_item(Item={
                'FaceID': new_noface_id,
                'ImageID': original_s3_key,
                'ImageURL': f's3://{BUCKET_NAME}/{original_s3_key}',
                'LowResImageURL': f's3://{BUCKET_NAME}/{low_res_s3_key}',  # Store URL of low-res image
                'WebPImageURL': f's3://{BUCKET_NAME}/{webp_s3_key}',       # Store URL of WebP image
                'BoundingBox': None,   # No bounding box available since no face was detected
                'Confidence': None,     # No confidence available since no face was detected
                'CategoryName': category_name   # Store the category name in DynamoDB
            })
            socketio.emit('upload_progress', {'filename': file.filename, 'stage': 'Database updated (no faces)'})
            logging.info(f"✅ No faces detected for '{original_s3_key}'. Stored with Face ID as {new_noface_id}.")

    except rekognition_client.exceptions.ResourceNotFoundException:
        logging.error(f"❌ Rekognition Collection '{COLLECTION_ID}' not found.")
        return jsonify({'error': f'Rekognition Collection "{COLLECTION_ID}" not found'}), 500
    
    except Exception as e:
        logging.error(f"❌ Error indexing faces from '{original_s3_key}' in Rekognition: {str(e)}")
        return jsonify({'error': f'Failed to index faces from {original_s3_key}'}), 500

    return jsonify({'message': f'Image {file.filename} uploaded successfully!'}), 200


@app.route("/fetch_count", methods=["GET"])
def fetch_image_count():
    unique_images = set()
    last_evaluated_key = None

    while True:
        query_params = {
            "IndexName": "CategoryIndex",
            "KeyConditionExpression": "CategoryName = :category",
            "ExpressionAttributeValues": {":category": "Mata-Ki-Chowki"}
        }
        if last_evaluated_key:
            query_params["ExclusiveStartKey"] = last_evaluated_key

        response = table.query(**query_params)

        for item in response.get("Items", []):
            if "ImageID" in item:
                unique_images.add(item["ImageID"])  # Add to unique set

        last_evaluated_key = response.get("LastEvaluatedKey")
        if not last_evaluated_key:
            break

    print(f"Total unique ImageID count: {len(unique_images)}")
    return jsonify({"unique_image_count": len(unique_images)}), 200

@app.route("/fetch_category_wise_data/<category_name>", methods=["GET"])
def get_images_by_category(category_name):
    """
    Fetch all unique image rows from DynamoDB using query pagination.
    """
    try:
        unique_images = {}  # Store unique images
        last_evaluated_key = None  # For pagination

        while True:
            query_params = {
                "IndexName": "CategoryIndex",  # Your GSI Name
                "KeyConditionExpression": "#category = :category",
                "ExpressionAttributeNames": {"#category": "CategoryName"},
                "ExpressionAttributeValues": {":category": category_name}
            }

            if last_evaluated_key:
                query_params["ExclusiveStartKey"] = last_evaluated_key  # Continue from last key

            response = table.query(**query_params)

            # Add only unique ImageID rows
            for item in response.get("Items", []):
                if "ImageID" in item:
                    unique_images[item["ImageID"]] = item  # Store unique images

            last_evaluated_key = response.get("LastEvaluatedKey")  # Get next page key
            if not last_evaluated_key:
                break  # Stop if no more pages

        return jsonify({"status": "success", "data": list(unique_images.values())}), 200

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

    
@app.route("/search", methods=["POST"])
def search_face():
    """
    Search for a face in Rekognition and return all matching images from the database
    with S3 URLs converted to public HTTP URLs (both full-resolution and thumbnail versions).
    """
    file = request.files.get("image")
    if not file:
        return jsonify({"error": "No image file provided"}), 400

    try:
        image_bytes = file.read()  # Read image as bytes

        # Search for matching faces in Rekognition (Directly using image bytes)
        response = rekognition_client.search_faces_by_image(
            CollectionId=COLLECTION_ID,
            Image={"Bytes": image_bytes},  # Directly passing image bytes
            FaceMatchThreshold=98,
            MaxFaces=1000
        )

    except Exception as e:
        logging.error(f"❌ Error searching face in Rekognition: {str(e)}")
        return jsonify({"error": "Failed to search face"}), 500

    matched_faces = response.get("FaceMatches", [])
    if not matched_faces:
        return jsonify({"MatchFound": False, "Message": "No matching face found."}), 404

    face_ids = [match["Face"]["FaceId"] for match in matched_faces]
    all_matched_images = []

    try:
        # Fetch all matching records using Query instead of Scan
        for face_id in face_ids:
            response = table.query(
                KeyConditionExpression=Key("FaceID").eq(face_id)
            )
            items = response.get("Items", [])

            # ✅ Convert S3 paths to public URLs
            for item in items:
                if "ImageURL" in item:
                    s3_path = item["ImageURL"].replace(f"s3://{BUCKET_NAME}/", "")
                    full_image_url = f"https://{BUCKET_NAME}.s3.us-east-2.amazonaws.com/{s3_path}"

                    # Generate thumbnail URL (assuming thumbnail images have '_thumb' suffix)
                    thumbnail_s3_path = s3_path.replace(".", "_thumb.")
                    thumbnail_url = f"https://{BUCKET_NAME}.s3.us-east-2.amazonaws.com/{thumbnail_s3_path}"

                    item["OriginalURL"] = full_image_url
                    item["ThumbnailURL"] = thumbnail_url  # ✅ Adding thumbnail URL

            all_matched_images.extend(items)

    except Exception as e:
        logging.error(f"❌ Error retrieving face metadata from DynamoDB: {str(e)}")
        return jsonify({"error": "Failed to retrieve face metadata"}), 500

    return jsonify({"MatchFound": True, "MatchedImages": all_matched_images})


@app.route('/update-data', methods=['GET'])
def update_data():
    try:
        # Query using categoryIndex instead of Scan
        response = table.query(
            IndexName="CategoryIndex",  # Use your Global Secondary Index (GSI)
            KeyConditionExpression="#category = :oldVal",
            ExpressionAttributeNames={"#category": "CategoryName"},  
            ExpressionAttributeValues={":oldVal": "Mata Ki Chowki"}
        )

        items = response.get("Items", [])

        if not items:
            return jsonify({"message": "No records found"}), 404

        # Update each item
        for item in items:
            face_id = item["FaceID"]  # Primary Key

            table.update_item(
                Key={"FaceID": face_id},
                UpdateExpression="SET #category = :newVal",
                ExpressionAttributeNames={"#category": "CategoryName"},
                ExpressionAttributeValues={":newVal": "Mata-Ki-Chowki"}
            )
            print("Updated item with FaceID: ", face_id)

        return jsonify({"message": f"Updated {len(items)} items successfully"}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    socketio.run(app, debug=True)
