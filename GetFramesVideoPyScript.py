import cv2
import os
import numpy as np
from pathlib import Path
import configparser
from datetime import datetime


def load_config(config_path):
    """
    Load configuration from a text file.
    Carga configuracion de un archivo de texto, debe indicarse
    path del video, path de la carpeta a donde se guardaran los archivos con las imagenes

    Args:
        config_path (str): Path to configuration file

    Returns:
        dict: Configuration parameters
    """
    config = configparser.ConfigParser()

    # Check if config file exists
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"No se hallo el archivo de configuracion, el .txt  {config_path}")

    config.read(config_path)

    # Extract parameters with default values
    params = {
        'video_path': config.get('PATHS', 'video_path', fallback=''),
        'output_path': config.get('PATHS', 'output_path', fallback='./extracted_frames'),
        'threshold': config.getint('PARAMETERS', 'threshold', fallback=30),
        'min_contour_area': config.getint('PARAMETERS', 'min_contour_area', fallback=500),
        'frame_skip': config.getint('PARAMETERS', 'frame_skip', fallback=5)
    }

    return params


def add_timestamp_to_path(base_path):
    """
    Add timestamp to base path to create unique folder name

    Args:
        base_path (str): Original base path

    Returns:
        str: Path with timestamp appended
    """
    # Get current timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Convert to Path object for easier manipulation
    path_obj = Path(base_path)

    # Create new path with timestamp appended to directory name
    if path_obj.is_dir() or base_path.endswith(os.path.sep):
        # If it's already a directory path, append timestamp
        new_path = str(path_obj) + "_" + timestamp
    else:
        # If it's a path that might include a file, append timestamp to the directory part
        parent = path_obj.parent
        name = path_obj.name
        new_path = str(parent / f"{name}_{timestamp}")

    return new_path


def extract_changing_frames(video_path, output_path, threshold=30, min_contour_area=500, frame_skip=5):
    """
    Extract frames from video when significant changes are detected.

    Args:
        video_path (str): Path to input video file
        output_path (str): Path to output directory for extracted frames
        threshold (int): Threshold for change detection (higher = less sensitive)
        min_contour_area (int): Minimum area for a contour to be considered significant
        frame_skip (int): Process every nth frame to improve performance
    """

    # Add timestamp to output path to create unique folder
    output_path_with_timestamp = add_timestamp_to_path(output_path)

    # Create output directory if it doesn't exist
    os.makedirs(output_path_with_timestamp, exist_ok=True)

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

    print(f"Video: {video_path}")
    print(f"Duracion: {duration:.2f} seconds")
    print(f"Total de frames recorridos: {total_frames}")
    print(f"FPS: {fps}")
    print(f"Directorio de frames obtenidos: {output_path_with_timestamp}")
    print(f"Parametros - Threshold: {threshold}, Min Contour Area: {min_contour_area}, Frame Skip: {frame_skip}")

    # Initialize variables
    prev_frame = None
    frame_count = 0
    saved_count = 0
    prev_gray = None

    print("\nProcesando frames...")

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # Skip frames for performance
        if frame_count % frame_skip != 0:
            frame_count += 1
            continue

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if prev_gray is None:
            prev_gray = gray
            frame_count += 1
            continue

        # Compute absolute difference between current and previous frame
        frame_diff = cv2.absdiff(prev_gray, gray)
        thresh = cv2.threshold(frame_diff, threshold, 255, cv2.THRESH_BINARY)[1]

        # Dilate the threshold image to fill in holes
        thresh = cv2.dilate(thresh, None, iterations=2)

        # Find contours on threshold image
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Check if any contour is large enough to indicate significant change
        significant_change = False
        for contour in contours:
            if cv2.contourArea(contour) > min_contour_area:
                significant_change = True
                break

        # If significant change detected, save the frame
        if significant_change:
            # Generate filename with timestamp and frame number
            timestamp = frame_count / fps
            filename = f"frame_{saved_count:06d}_time_{timestamp:.2f}s.jpg"
            filepath = os.path.join(output_path_with_timestamp, filename)

            # Save the frame
            cv2.imwrite(filepath, frame)
            saved_count += 1

            print(f"Saved frame {saved_count}: {filename}")

        # Update previous frame
        prev_gray = gray
        frame_count += 1

        # Progress indicator
        if frame_count % 100 == 0:
            progress = (frame_count / total_frames) * 100
            print(f"Progress: {progress:.1f}% - Saved {saved_count} frames so far")

    # Release video capture
    cap.release()

    print(f"\nExtraccion completa!")
    print(f"Total de frames procesados: {frame_count}")
    print(f"Numero de Frames guardados: {saved_count}")
    print(f"Directorio: {output_path_with_timestamp}")


def create_default_config(config_path):
    """
    Create a default configuration file if it doesn't exist.

    Args:
        config_path (str): Path where to create the config file
    """
    config = configparser.ConfigParser()

    config['PATHS'] = {
        'video_path': r'C:\Users\mitno\Videos\your_video.mp4',
        'output_path': r'C:\Users\mitno\Videos\extracted_frames'
    }

    config['PARAMETERS'] = {
        'threshold': '30',
        'min_contour_area': '500',
        'frame_skip': '5'
    }

    with open(config_path, 'w') as configfile:
        config.write(configfile)

    print(f"Archivo de Configuracion creado por defecto: {config_path}")
    print("Edita el archivo de configuracion, verifica el nombre del video, y el nombre de la carpeta donde se extraeran"
          "los frames,si no existe se creara con el nombre y ruta que tenga el archivo de configuracion, si existe usara"
          "ese nombre y ruta existente.")


def main():
    # Configuration file path
    config_path = r"C:\Users\mitno\Videos\frame_extractor_config.txt"

    # Create default config if it doesn't exist
    if not os.path.exists(config_path):
        create_default_config(config_path)
        return

    try:
        # Load configuration from file
        params = load_config(config_path)

        # Extract parameters
        video_path = params['video_path']
        output_path = params['output_path']
        threshold = params['threshold']
        min_contour_area = params['min_contour_area']
        frame_skip = params['frame_skip']

        # Validate video path
        if not video_path or not os.path.exists(video_path):
            print(f"Error: Video file not found at {video_path}")
            print("Please update the video_path in the configuration file.")
            return

        print("Configuration loaded successfully:")
        print(f"Video path: {video_path}")
        print(f"Output base path: {output_path}")
        print(f"Threshold: {threshold}")
        print(f"Min Contour Area: {min_contour_area}")
        print(f"Frame Skip: {frame_skip}")
        print("\nStarting frame extraction...")

        extract_changing_frames(video_path, output_path, threshold, min_contour_area, frame_skip)

    except Exception as e:
        print(f"Error: {e}")
        print("Please check your configuration file format.")


if __name__ == "__main__":
    main()