from librairies import *

def download_file(url, path):  
    """  
    Download a file from a given URL to a specified local path.  
      
    Args:  
        url (str): URL of the file to download.  
        path (str): Local path where the file will be saved.  
    """
    # Use subprocess to run 'curl' command for downloading files
    subprocess.run(["curl", "-L", url, "-o", path], check=True)  
  
def download_dataset(config):
    """  
    Download the necessary datasets if they don't already exist in the local storage.  
      
    Args:  
        config (dict): Configuration dictionary containing paths for datasets.  
    """
    gtFine_path = config["gtFine_path"]
    leftImg8bit_path = config["leftImg8bit_path"]
    # Create the 'data' directory if it doesn't exist  
    if not os.path.exists('data'):  
        os.makedirs('data')  
  
    # Check and download the gtFine zip if it doesn't exist  
    if not os.path.isfile(gtFine_path) and not os.path.isdir('./data/gtFine'):  
        download_file("https://s3-eu-west-1.amazonaws.com/static.oc-static.com/prod/courses/files/AI+Engineer/Project+8+-+Participez+%C3%A0+la+conception+d'une+voiture+autonome/P8_Cityscapes_gtFine_trainvaltest.zip", gtFine_path)  
        print(f"Downloaded: {gtFine_path}")  
    else:  
        print(f"Dataset already exist, skip downloading: {gtFine_path}")  
  
    # Check and download the leftImg8bit zip if it doesn't exist  
    if not os.path.isfile(leftImg8bit_path) and not os.path.isdir('./data/leftImg8bit'):  
        download_file("https://s3-eu-west-1.amazonaws.com/static.oc-static.com/prod/courses/files/AI+Engineer/Project+8+-+Participez+%C3%A0+la+conception+d'une+voiture+autonome/P8_Cityscapes_leftImg8bit_trainvaltest.zip", leftImg8bit_path)  
        print(f"Downloaded: {leftImg8bit_path}")  
    else:  
        print(f"Dataset already exist, skip downloading: {leftImg8bit_path}")  
  
def unzip_dataset(config):
    """  
    Unzip the downloaded datasets into the designated directory.  
      
    Args:  
        config (dict): Configuration dictionary containing paths for datasets.  
    """
    gtFine_path = config["gtFine_path"]
    leftImg8bit_path = config["leftImg8bit_path"]
    
    # Unzip gtFine dataset if the zip file exists  
    if os.path.isfile(gtFine_path):  
        subprocess.run(["unzip", "-q", "-n", gtFine_path, "-d", "data/"], check=True)  
        subprocess.run(["rm", gtFine_path], check=False)
  
    # Unzip leftImg8bit dataset if the zip file exists 
    if os.path.isfile(leftImg8bit_path):  
        subprocess.run(["unzip", "-q", "-n", leftImg8bit_path, "-d", "data/"], check=True)  
        subprocess.run(["rm", leftImg8bit_path], check=False)
          
def move_val_test(config):
    """  
    Organize dataset directories by moving validation data to test directories as necessary.  
      
    Args:  
        config (dict): Configuration dictionary containing paths for datasets.  
    """
    
    data_path = config["data_path"]
    
    # Define directories to delete under gtFine and leftImg8bit 
    data_paths_to_delete_gtFine = [  
        os.path.join(data_path, "test/berlin"),  
        os.path.join(data_path, "test/bielefeld"),  
        os.path.join(data_path, "test/bonn"),  
        os.path.join(data_path, "test/leverkusen"),  
        os.path.join(data_path, "test/mainz"),  
        os.path.join(data_path, "test/munich")  
    ]
    
    data_paths_to_delete_leftImg8bit = [  
        os.path.join(data_path, "..", "leftImg8bit", "test/berlin"),  
        os.path.join(data_path, "..", "leftImg8bit", "test/bielefeld"),  
        os.path.join(data_path, "..", "leftImg8bit", "test/bonn"),  
        os.path.join(data_path, "..", "leftImg8bit", "test/leverkusen"),  
        os.path.join(data_path, "..", "leftImg8bit", "test/mainz"),  
        os.path.join(data_path, "..", "leftImg8bit", "test/munich")  
    ]  
    
    # Define directories to move from 'val' to 'test' 
    data_paths_to_move_gtFine = [  
        os.path.join(data_path, "val/munster"),  
    ]  
      
    data_paths_to_move_leftImg8bit = [  
        os.path.join(data_path, "..", "leftImg8bit", "val/munster"),  
    ]  
    
    # Delete the specified directories  
    for directory in data_paths_to_delete_gtFine + data_paths_to_delete_leftImg8bit:  
        if os.path.exists(directory):  
            shutil.rmtree(directory)
            print(f"- Deleted directory: {directory}")
    
     # Move the specified directories  
    for src_dir in data_paths_to_move_gtFine + data_paths_to_move_leftImg8bit:  
        destination = src_dir.replace("val", "test")
        if os.path.exists(src_dir):  
            shutil.move(src_dir, destination)  
            print(f"Moved {src_dir} to {destination}")  