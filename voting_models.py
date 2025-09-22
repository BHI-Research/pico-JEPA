import re
import csv
import os
from collections import defaultdict

def parse_inference_file(input_file_path):

    video_results = defaultdict(list)
    
    
    video_pattern = re.compile(r'Video: (.+\.mp4)')
    index_pattern = re.compile(r'Predicted Class: .+?\(Index: (\d+)\)')
    
    with open(input_file_path, 'r') as f:
        current_video = None
        for line in f:
            
            video_match = video_pattern.search(line)
            if video_match:
                current_video = os.path.basename(video_match.group(1))
                continue
                
            
            index_match = index_pattern.search(line)
            if index_match and current_video:
                video_results[current_video].append(index_match.group(1))
    
    return video_results

def write_ensamble_csv(video_results, output_csv_path):

    with open(output_csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        writer.writerow(['index modelo 1', 'index modelo 2', 'index modelo 3', 'index modelo 4'])
        
        
        for video, indices in video_results.items():
            if len(indices) == 4:
                writer.writerow(indices)
            else:
                print(f"Warning: Video {video} has {len(indices)} predictions instead of 4. It will be skipped.")

if __name__ == "__main__":

    input_classes_file = "videos/classInd.txt"
    input_results_dir = "."
    output_csv_dir = "vote_results"

    if not os.path.exists(output_csv_dir):
        os.makedirs(output_csv_dir)

    try:
        with open(input_classes_file, 'r') as f:
            for line in f:
                
                class_name = line.strip().split(' ', 1)[-1]
                if not class_name:
                    continue

                input_file_name = f"{class_name}.txt"
                input_file_path = os.path.join(input_results_dir, input_file_name)
                output_csv_path = os.path.join(output_csv_dir, f"vote_{class_name}.csv")

                print(f"--- Processing results file: {input_file_path} ---")

                if os.path.exists(input_file_path):
                    # Parsear el archivo de resultados
                    video_results = parse_inference_file(input_file_path)
                    
                    
                    write_ensamble_csv(video_results, output_csv_path)
                    print(f"Results saved in: {output_csv_path}")
                else:
                    print(f"Warning: The file {input_file_path} does not exist. This class will be skipped.")
    except FileNotFoundError:
        print(f"Error: Class file not found '{input_classes_file}'.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

