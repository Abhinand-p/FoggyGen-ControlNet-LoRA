from datasets import load_dataset

# example 1: local folder
dataset = load_dataset("imagefolder", data_dir="foggy_shift_images", split='train')

print(dataset[0]['text'], dataset[0])