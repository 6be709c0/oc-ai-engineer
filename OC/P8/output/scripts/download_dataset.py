from librairies import *

def download_file(url, path):  
    subprocess.run(["curl", "-L", url, "-o", path], check=True)  
  
def download_dataset(config):  
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
    gtFine_path = config["gtFine_path"]
    leftImg8bit_path = config["leftImg8bit_path"]
    
    # Unzip gtFine  
    if os.path.isfile(gtFine_path):  
        subprocess.run(["unzip", "-q", "-n", gtFine_path, "-d", "data/"], check=True)  
        subprocess.run(["rm", gtFine_path], check=False)  # check=False because deletion is not critical  
  
    # Unzip leftImg8bit  
    if os.path.isfile(leftImg8bit_path):  
        subprocess.run(["unzip", "-q", "-n", leftImg8bit_path, "-d", "data/"], check=True)  
        subprocess.run(["rm", leftImg8bit_path], check=False)  # check=False because deletion is not critical  