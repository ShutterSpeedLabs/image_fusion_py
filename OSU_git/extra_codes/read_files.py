import glob

path = "/media/kisna/data_1/image_fusion/image_fusion_dataset/OSU/1a/*"

for path in glob.glob(path, recursive=True):
    print(path)

