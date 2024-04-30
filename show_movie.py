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

    animation = FuncAnimation(fig, update, frames=range(stim.size(1)), interval=40)
    plt.show()


def main():
    global upon_animation_close
    # Load the model
    checkpoint = load_checkpoint('MOST_RECENT_IN_DEFAULT_FOLDER')
    if checkpoint==None:
        return
    
    model = init_from_checkpoint(checkpoint)
    model.to('cpu')
    
    data_folder_path = os.path.dirname(__file__)+'_data'
    dataset = OpticFlowDotsDataset(data_folder_path)
    
    while True:
        show_loader = DataLoader(dataset, batch_size=1, shuffle=True)
        for one_batch_data, one_batch_targets in show_loader:
            break
    
        with torch.no_grad():
            for batch_data, batch_targets in show_loader:
                batch_data, batch_targets = batch_data.to('cpu'), batch_targets.to('cpu')  # Move data to the device
                model_response = model(batch_data)
                loss = checkpoint['hyperparms']['loss_fnc'](model_response, batch_targets)
                break
            
            data_dict = {'stimulus': batch_data.squeeze(0), 
                         'target_response': batch_targets.squeeze(0),
                         'actual_response': model_response.squeeze(0),
                         'loss': loss.item()}
    
        upon_animation_close="quit"
        animate_movie(data_dict) # May change upon_animation_close
        if upon_animation_close=="quit":
            break


if __name__ == "__main__":
    main()