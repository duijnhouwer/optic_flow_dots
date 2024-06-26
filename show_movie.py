import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button
import torch
from egomotnet import load_checkpoint, init_from_checkpoint
import os
from optic_flow_dots_dataset import OpticFlowDotsDataset
from torch.utils.data import DataLoader


upon_animation_close = "quit"

def animate_movie(data_dict):
    """Animate a movie stored in a PyTorch tensor."""

    stim = data_dict['stimulus'] # shape: Frames x Height x Width
    targ = data_dict['target_response'] # shape: vector
    resp = data_dict['actual_response'] # shape: vector
    loss = data_dict['loss']

    def on_button_clicked(event):
        global upon_animation_close
        upon_animation_close = "show_next_random_example"
        plt.close(fig)


    def update(frame_number):
        ax.clear()
        ax.imshow(stim[0, frame_number], cmap='gray')
        #ax.set_axis_off()
        ax.set_title('mLOSS = {}'.format(round(loss*1000000)))
        ax.set_ylabel('mRESP = {}'.format([str(round(i*1000000)) for i in resp.tolist()]))
        ax.set_xlabel('mTARG = {}'.format([str(round(i*1000000)) for i in targ.tolist()]))


    fig, ax = plt.subplots(figsize=(10, 8))
    plt.subplots_adjust(bottom=0.2)  # Make room for the button
    button_ax = fig.add_axes([0.4, 0.05, 0.2, 0.075])  # Rectangle [left, bottom, width, height]
    button = Button(button_ax, 'Next')
    button.on_clicked(on_button_clicked)

    animation = FuncAnimation(fig, update, frames = range(stim.size(1)), interval=250)
    plt.show()


def main():
    global upon_animation_close

    # Instantiate data loader
    data_folder_path = os.path.dirname(__file__)+'_data'
    dataset = OpticFlowDotsDataset(data_folder_path)
    show_loader = DataLoader(dataset, batch_size=1, shuffle=True)

    # Load the model
    checkpoint = load_checkpoint('MOST_RECENT_IN_DEFAULT_FOLDER')
    if checkpoint==None:
        return

    model = init_from_checkpoint(checkpoint)[0]
    model.to('cpu')
    model.eval()

    while True:

        #for one_batch_data, one_batch_targets in show_loader:
        #    break

        with torch.no_grad():
            for X, y in show_loader:
                X, y = X.to('cpu'), y.to('cpu')
                yHat = model(X)
                loss = checkpoint['hyperparms']['loss_fnc'](yHat, y)
                break

            data_dict = {'stimulus': X.squeeze(0),
                         'target_response': y.squeeze(0),
                         'actual_response': yHat.squeeze(0),
                         'loss': loss.item()}

        upon_animation_close="quit"
        animate_movie(data_dict) # May change upon_animation_close
        if upon_animation_close=="quit":
            break


if __name__ == "__main__":
    main()
