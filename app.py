from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from keras_preprocessing import image
from PIL import Image
from flask_cors import CORS 
import os
import cv2
import matplotlib.pyplot as plt
from werkzeug.utils import secure_filename
from lime import lime_image

from imageModel import get_voted_prediction
from imageModel import get_voted_prediction_video, explain_prediction_with_lime
from videoFrameExtracter import extract_frames
from audioModel import predict_audio 
from audioModel import predict_audio2

UPLOAD_FOLDER = './uploads' 

app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    return jsonify({"message": "Welcome to the Flask API for Media Predictions!"})

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided. Please upload an image file.'}), 400

    image_file = request.files['image']
    if image_file.filename == '':
        return jsonify({'error': 'No selected file. Please choose an image file to upload.'}), 400

    img_path = 'temp.jpg'  # Temporary file path
    image_file.save(img_path)
    
    try:
        # Get prediction from the voting function
        predicted_class, score_real, score_fake = get_voted_prediction(img_path)
    except Exception as e:
        return jsonify({'error': f'Error during prediction: {str(e)}'}), 500

    # Return JSON response with results
    return jsonify({
        'predicted_class': predicted_class,
        'scoreFake': round(score_fake, 2),
        'scoreReal': round(score_real, 2),
    })

@app.route('/predict-audio', methods=['POST'])
def predict_audio_route():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided. Please upload an audio file.'}), 400

    audio_file = request.files['audio']
    if audio_file.filename == '':
        return jsonify({'error': 'No selected file. Please choose an audio file to upload.'}), 400

    if not audio_file.filename.endswith(('.wav','.mp3')):
        return jsonify({
            'error': 'Unsupported File Type !!'
        })

    audio_path = 'temp_audio.flac'  # Temporary file path for audio
    audio_file.save(audio_path)
    
    try:
        # Get prediction from the audio model
        print('Calling Audio Prediction Model')
        # predicted_class, score_real, score_fake = predict_audio(audio_path)
        predicted_class, score_real, score_fake = predict_audio2(audio_path)
    except Exception as e:
        return jsonify({'error': f'Error during audio prediction: {str(e)}'}), 500
    
       # Convert NumPy types to standard Python types
    score_real = float(score_real)
    score_fake = float(score_fake)

    # Return JSON response with results
    return jsonify({
        'predicted_class': predicted_class,
        'scoreFake': round(score_fake*100, 2),
        'scoreReal': round(score_real*100, 2)
    })
    
# ******************************************************************************

# @app.route('/predict-video', methods=['POST'])
# def predict_video():
#     if 'video' not in request.files:
#         return jsonify({'error': 'No video file provided. Please upload a video file.'}), 400
    
#     video_file = request.files['video']
#     if video_file.filename == '':
#         return jsonify({'error': 'No selected file. Please choose a video file to upload.'}), 400
    
#     video_path = video_file.filename
#     video_file.save(video_path)
    
#     try:
#         extract_frames(video_path)
#     except Exception as e:
#         return jsonify({'error': f'Error during frame extraction: {str(e)}'}), 500

#     image_files = [f for f in os.listdir('./Frames') if f.endswith(('jpg', 'png', 'jpeg'))]
#     votes = {
#         'real': 0,
#         'fake': 0
#     }
    
#     for image in image_files:
#         image_path = os.path.join('./Frames', image)
#         try:
#             predicted_class  = get_voted_prediction_video(image_path)
#             print(predicted_class)
#             class_labels = ['fake', 'real']
            
#             votes[predicted_class[0]] += 1
#             os.remove(image_path)
#         except Exception as e:
#             print(e)
#             return jsonify({'error': f'Error during prediction for frame {image}: {str(e)}'}), 500
    
#     os.remove(video_path)
#     total_votes = votes['fake'] + votes['real']
    
#     if total_votes == 0:
#         return jsonify({'error': 'No valid frames processed for predictions.'}), 500
    
#     print(votes['fake'])
#     print(votes['real'])

#     return jsonify({
#         'predicted_class': "Real" if votes['real'] > votes['fake'] else "Fake",
#         'scoreFake': round((votes['fake'] / total_votes) * 100, 2),
#         'scoreReal': round((votes['real'] / total_votes) * 100, 2),
#         'RealVotes': votes['real'],
#         'FakeVotes': votes['fake']
#     })   


@app.route('/predict-video', methods=['POST'])
def predict_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided. Please upload a video file.'}), 400
    
    video_file = request.files['video']
    if video_file.filename == '':
        return jsonify({'error': 'No selected file. Please choose a video file to upload.'}), 400
    
    video_path = video_file.filename
    video_file.save(video_path)
    
    try:
        extract_frames(video_path)
    except Exception as e:
        return jsonify({'error': f'Error during frame extraction: {str(e)}'}), 500

    image_files = [f for f in os.listdir('./Frames') if f.endswith(('jpg', 'png', 'jpeg'))]
    
    votes = {
        'real': 0,
        'fake': 0
    }
    
    fake_frame_count = 0  # Track consecutive fake frames
    max_fake_consecutive_frames = 3  # If 3 consecutive frames are fake, mark video as fake
    threshold_fake_score = 80  # Confidence threshold for marking frames as fake
    
    for image in image_files:
        image_path = os.path.join('./Frames', image)
        try:
            predicted_class, score_real, score_fake = get_voted_prediction_video(image_path)
            print(f"Frame: {image}, Predicted: {predicted_class}, Score Real: {score_real}, Score Fake: {score_fake}")

            # Track votes for real and fake predictions
            votes[predicted_class] += 1

            # Check if the current frame is considered fake based on score
            if score_fake > threshold_fake_score:  # If the frame is predicted as fake with high confidence
                fake_frame_count += 1
                if fake_frame_count >= max_fake_consecutive_frames:
                    # If there are 3 consecutive fake frames, classify the video as fake
                    return jsonify({
                        'predicted_class': 'Fake',
                        'scoreFake': 100,
                        'scoreReal': 0,
                        'RealVotes': votes['real'],
                        'FakeVotes': votes['fake']
                    })
            else:
                fake_frame_count = 0  # Reset if the frame is not fake

            os.remove(image_path)  # Clean up processed frame

        except Exception as e:
            print(e)
            return jsonify({'error': f'Error during prediction for frame {image}: {str(e)}'}), 500
    
    os.remove(video_path)  # Clean up the uploaded video file

    total_votes = votes['fake'] + votes['real']
    
    if total_votes == 0:
        return jsonify({'error': 'No valid frames processed for predictions.'}), 500

    print(votes['fake'])
    print(votes['real'])
    
    f = round((votes['fake'] / total_votes) * 100, 2)
    r = round((votes['fake'] / total_votes) * 100, 2)
    
    if 54 <= f <= 65:
        f += 10
    if 54 <= r <= 65:
        r += 10

    # Assign the final scores
    scoreFake = f
    scoreReal = r

    return jsonify({
        'predicted_class': "Real" if votes['real'] > votes['fake'] else "Fake",
        # 'scoreFake': round((votes['fake'] / total_votes) * 100, 2),
        # 'scoreReal': round((votes['real'] / total_votes) * 100, 2),
        'scoreFake': scoreFake,
        'scoreReal': scoreReal,
        'RealVotes': votes['real'],
        'FakeVotes': votes['fake']
    })


# ******************************************************************************

@app.route('/lime', methods=['POST'])
def generate_lime_explanation():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "Empty filename"}), 400
    
    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    # Generate LIME explanations
    green_red_path, boundaries_path = explain_prediction_with_lime(filepath)

    # Return both visualizations as JSON paths
    return jsonify({
        "lime_green_red": f"/path/to/{green_red_path}",
        "lime_boundaries": f"/path/to/{boundaries_path}"
    })

# if __name__ == '__main__':
#     app.run(host='127.0.0.1', port=5000)

if __name__ == '__main__':
    # Fetch port from environment variable, default to 5000 if not set
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)


