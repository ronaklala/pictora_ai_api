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
from dotenv import dotenv_values



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
    aws_access_key_id=os.environ.get("aws_access_key_id"),
    aws_secret_access_key=os.environ.get("aws_secret_access_key"),
    region_name=os.environ.get("aws_region")
)

s3_client = boto3.client(
    's3',
    aws_access_key_id=os.environ.get("aws_access_key_id"),
    aws_secret_access_key=os.environ.get("aws_secret_access_key"),
    region_name=os.environ.get("aws_region")
)

dynamodb = boto3.resource(
    'dynamodb',
    aws_access_key_id=os.environ.get("aws_access_key_id"),
    aws_secret_access_key=os.environ.get("aws_secret_access_key"),
    region_name=os.environ.get("aws_region")
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
    """Fetch the last NOFACE ID stored in DynamoDB."""
    response = table.scan(FilterExpression="begins_with(FaceID, :prefix)", ExpressionAttributeValues={":prefix": "NOFACE"})
    
    if response['Items']:
        last_noface_ids = [item['FaceID'] for item in response['Items']]
        
        if last_noface_ids:
            return max(last_noface_ids)  # Return the maximum NOFACE ID found
    
    return "NOFACE_0"  # Default if none found


def increment_noface_id(last_noface_id):
    """Increment the last NOFACE ID."""
    if last_noface_id == "NOFACE_0":
        return "NOFACE_1"
    
    parts = last_noface_id.split("_")
    
    if len(parts) == 2 and parts[0] == "NOFACE":
        next_number = int(parts[1]) + 1
        return f"NOFACE_{next_number}"
    
    return "NOFACE_0"  # Fallback

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
            last_noface_id = get_last_noface_id()  # Function to get last NOFACE ID from DB
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



@app.route("/fetch_category_wise_data/<category_name>", methods=["GET"])
def get_images_by_category(category_name):
    """
    Fetch images from DynamoDB based on CategoryName.
    
    :param table_name: Name of the DynamoDB table
    :param category_name: Category to filter (e.g., "bhabhi")
    :return: List of image URLs
    """
    
    # Query DynamoDB
    response = table.scan(
        FilterExpression="CategoryName = :category",
        ExpressionAttributeValues={":category": category_name}
    )
    
    # Extract image URLs
    items = response.get("Items", [])
    
    return jsonify(items)
    
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
            FaceMatchThreshold=90,
            MaxFaces=1000
        )

    except Exception as e:
        logging.error(f"❌ Error searching face in Rekognition: {str(e)}")
        return jsonify({"error": "Failed to search face"}), 500

    matched_faces = response.get("FaceMatches", [])
    if not matched_faces:
        return jsonify({"MatchFound": False, "Message": "No matching face found."}), 404

    all_matched_images = []
    
    # Retrieve all matching images from DynamoDB for each matched FaceID
    for match in matched_faces:
        face_id = match["Face"]["FaceId"]

        try:
            result = table.scan(
                FilterExpression="FaceID = :face_id",
                ExpressionAttributeValues={":face_id": face_id}
            )
            items = result.get("Items", [])

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



if __name__ == '__main__':
    socketio.run(app, host="0.0.0.0", port=5000, debug=True, allow_unsafe_werkzeug=True)
