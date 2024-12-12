import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class TrainingVisualizer:
    def __init__(self):
        self.losses = [0]  
        self.learning_rates = [0]  

        self.fig, self.axs = plt.subplots(1, 2, figsize=(12, 4))
        self.loss_line, = self.axs[0].plot([0], [0], label='Training Loss')  
        self.lr_line, = self.axs[1].plot([0], [0], label='Learning Rate', color='r')  

        self.axs[0].set_xlabel('Iterations')
        self.axs[0].set_ylabel('Loss')
        self.axs[1].set_xlabel('Iterations')
        self.axs[1].set_ylabel('Learning Rate')

        self.axs[0].legend()
        self.axs[1].legend()

    def update_plot(self, frame):
        if frame < len(self.losses):
            self.loss_line.set_data(range(frame + 1), self.losses[:frame + 1])
            self.lr_line.set_data(range(frame + 1), self.learning_rates[:frame + 1])
        return self.loss_line, self.lr_line

    def visualize(self):
        animation = FuncAnimation(self.fig, self.update_plot, frames=len(self.losses), blit=True)
        plt.show()