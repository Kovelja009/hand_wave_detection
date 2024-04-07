import csv

def save_video_tracking_data(tracks, file_name):
    """Save tracking data to a CSV file."""
    fieldnames = tracks.keys()

    # Write the dictionary to the CSV file in append mode
    with open(file_name, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        # Check if the file is empty (i.e., if it needs a header row)
        # If the file is empty, write the header row
        if csvfile.tell() == 0:
            writer.writeheader()
        
        # Write each row of the dictionary to the CSV file
        for i in range(len(tracks['frame'])):
            row = {field: tracks[field][i] for field in fieldnames}
            writer.writerow(row)