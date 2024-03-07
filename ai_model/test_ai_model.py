# open the v2 model and test it
from pix2pix_2d_RBG import set_up_data_as_input, Generator, Discriminator
import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import savgol_filter


def calculateLowerUpper(column):
    mean = np.mean(column.to_numpy())
    std = np.std(column.to_numpy())
    lower = (mean - 1.96 * std) 
    upper = (mean + 1.96 * std) 
    return lower,upper


def plot_tensorboard_training(discriminator_csv, total_loss_csv):
    # open the two csv files
    discriminator_df = pd.read_csv(discriminator_csv, delimiter=',')
    total_loss_df = pd.read_csv(total_loss_csv, delimiter=',')
    # smooth the losses
    yhat_discriminator = savgol_filter(discriminator_df['Value'], 200, 3)
    yhat_total_loss = savgol_filter(total_loss_df['Value'], 200, 3)
    lower_discriminator, upper_discriminator = calculateLowerUpper(discriminator_df['Value'])
    lower_total_loss, upper_total_loss = calculateLowerUpper(total_loss_df['Value'])
    # plot the losses
    fig, ax = plt.subplots(1, 2, figsize=(15, 5), sharex=True)
    ax[0].plot(discriminator_df['Step'], yhat_discriminator, color='blue')
    ax[0].set_title('Discriminator loss')
    ax[0].set_ylabel('Loss')
    ax[1].plot(total_loss_df['Step'], yhat_total_loss, color='blue')
    ax[1].set_title('Total loss')
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Loss')
    fig.savefig("output/loss.png")
    plt.close()


if __name__=="__main__":
    #plot_tensorboard_training("logs/fit_train/keep20231124-094939/keep20231124-094939_disc_loss.csv", 
    #                          "logs/fit_train/keep20231124-094939/keep20231124-094939_gen_total_loss.csv")
    directory_inputs = "D:/sheetpile/ai_model/inputs_geometry_re"
    BATCH_SIZE = 1
    train_dataset, test_dataset = set_up_data_as_input(directory_inputs)
    OUTPUT_CHANNELS = 1
    generator_optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0.5)
    discriminator_optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0.5)
    generator=Generator()
    discriminator=Discriminator()
    checkpoint_dir = "D:/sheetpile/ai_model/training_checkpoints_2d_geometry_refined"
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                     discriminator_optimizer=discriminator_optimizer,
                                     generator=generator,
                                     discriminator=discriminator)

    ckpt_manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=11)
    if ckpt_manager.latest_checkpoint:
        checkpoint.restore(ckpt_manager.latest_checkpoint)
        # You can also access previous checkpoints like this: ckpt_manager.checkpoints[3]
        print ('Latest checkpoint restored!!')
        dataset = [value for counter, value in enumerate(test_dataset)]
        predictions = np.array([generator(data[0], training=True) for data in dataset])
        target = np.array([data[1] for data in dataset]).flatten()
        for index in range(len(dataset)):
            # plot the result
            fig, ax = plt.subplots(2, 4, figsize=(30, 15))
            # plot the inputs 
            ax[0, 0].imshow(dataset[index][0][0, :, :, 0].numpy(), cmap='gray')
            ax[0, 1].imshow(dataset[index][0][0, :, :, 1].numpy(), cmap='gray' )
            ax[0, 2].imshow(dataset[index][0][0, :, :, 2].numpy(), cmap='gray')
            ax[0, 3].imshow(dataset[index][0][0, :, :, 3].numpy(), cmap='gray')
            # plot the target
            ax[1, 1].imshow(dataset[index][1][0, :, :, 0].numpy(), cmap='gray')
            # plot the prediction
            ax[1, 2].imshow(predictions[index][0, :, :, 0], cmap='gray')
            # plot the error
            ax[1, 3].imshow(dataset[index][1][0, :, :, 0].numpy() - predictions[index][0, :, :, 0], cmap='gray')
            # add titles
            ax[0, 0].set_title('Youngs Modulus')
            ax[0, 1].set_title('Geometry')
            ax[0, 2].set_title('Water Pressure')
            ax[0, 3].set_title('Soil Density')
            ax[1, 1].set_title('Target displacement')
            ax[1, 2].set_title('Predicted displacement')
            absolute_error = np.abs(dataset[index][1][0, :, :, 0].numpy() - predictions[index][0, :, :, 0]).max()
            ax[1, 3].set_title(f"Maximal absolute error: {absolute_error:.2f}")
            # remove all the axis
            for i in range(2):
                for j in range(4):
                    ax[i, j].axis('off')
            # make distance between subplots smaller
            plt.subplots_adjust(wspace=0.1, hspace=0.1)
            plt.tight_layout()
            # save the figure
            plt.savefig(f"output_geometry_re/testing/test_{index}.png")
            plt.close()
