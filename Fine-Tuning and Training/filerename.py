import os

def rename_files_in_directory(directory):
    """
    Rename files in the given directory by adding 1000 to their numeric names.
    For example, "0001.png" becomes "1001.png".
    """
    print(f"Scanning directory: {directory}")
    files_to_rename = [f for f in os.listdir(directory) if f.endswith('.png') and f[:-4].isdigit()]
    
    if not files_to_rename:
        print("No matching files found to rename (e.g., '0001.png').")
        return

    for filename in files_to_rename:
        try:
            old_number = int(filename[:-4])
            new_number = old_number + 1000
            # Ensure the new number is formatted with leading zeros if needed
            new_filename = f"{new_number:04}.png"
            
            old_filepath = os.path.join(directory, filename)
            new_filepath = os.path.join(directory, new_filename)

            # Avoid overwriting existing files
            if os.path.exists(new_filepath):
                print(f"Skipping rename: {new_filename} already exists.")
                continue

            os.rename(old_filepath, new_filepath)
            print(f"Renamed: {filename} -> {new_filename}")
        except ValueError:
            print(f"Skipping non-numeric filename: {filename}")
        except Exception as e:
            print(f"An error occurred with {filename}: {e}")

if __name__ == "__main__":
    # --- CHANGE THIS ---
    # Get the directory where the script itself is located. This is more reliable.
    script_directory = os.path.dirname(os.path.abspath(__file__))
    rename_files_in_directory(script_directory)