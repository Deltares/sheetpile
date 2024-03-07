import pickle

# open the file for reading
fileObject = open('D:\sheetpile/ai_model/inputs_geometry/data_600.p','rb')
# unpickle the object
data = pickle.load(fileObject)
feature_names = data['feature_names']
all_data = data['all_data']

# plot all inputs and save to file
import matplotlib.pyplot as plt
import numpy as np

counter = 0
# create gif from images
import imageio
images = []
for data_set in all_data[:200]:
    # plot as image
    if len(data_set.shape) > 1:
        counter += 1
        images.append(imageio.imread(f'D:\sheetpile\output_geometry/input_data{counter}.png'))
        fig, ax = plt.subplots()
        ax.imshow(data_set[2, :, :], cmap='gray')
        # remove ticks
        ax.set_xticks([])
        ax.set_yticks([])
        plt.tight_layout()
        plt.savefig(f'D:\sheetpile\output_geometry/input_data{counter}.png')
        plt.close()
imageio.mimsave(f'D:\sheetpile\output_geometry/inputs.gif', images[:-2], duration=1)

   



    #fig, axes = plt.subplots(subplots, 1, figsize=(10, 10))
    #for i, ax in enumerate(axes):
    #    # plot as image
    #    ax.imshow(data_set[i], cmap='gray')
    #    # write label
    #    ax.set_title(feature_names[i])
    #    # remove ticks
    #    ax.set_xticks([])
    #    ax.set_yticks([])
    #plt.tight_layout()
    #plt.savefig(f'D:\sheetpile\output_geometry/input_data{counter}.png')
    #plt.close()
    #counter += 1