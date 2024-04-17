import matplotlib.pyplot as plt
import matplotlib.animation as animation
import inspect
import the_luggage as lgg

def animate_movie(data_dict):
    """Animate a movie stored in a PyTorch tensor."""
    stim = data_dict['stimulus']
    resp = data_dict['target_response']
    
    fig, ax = plt.subplots()
    
    def update(frame_number):
        ax.clear()
        ax.imshow(stim[0, frame_number], cmap='gray')
        ax.set_axis_off()
        ax.set_title([str(round(i*1000000)) for i in resp.tolist()])

    ani = animation.FuncAnimation(fig, update, frames=range(stim.size(1)), interval=50)
    plt.show()

def main():
    K = lgg.load_stimulus_and_target_response()
    if K['file_path'] == "No file selected":
        print('[{}.main] {}'.format(inspect.currentframe().f_code.co_name, "No file selected"))
        return
    
    # Load the model
    model = lgg.load_pytorch_model()
    if model==None:
        return
    animate_movie(K)

if __name__ == "__main__":
    main()