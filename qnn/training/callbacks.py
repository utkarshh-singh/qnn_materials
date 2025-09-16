
import matplotlib.pyplot as plt
from IPython.display import clear_output
class LiveObjectivePlot:
    def __init__(self, title='Objective vs Iteration'):
        self.values = []; self.title = title
    def __call__(self, weights, obj_value):
        self.values.append(float(obj_value))
        clear_output(wait=True)
        plt.title(self.title); plt.xlabel('Iteration'); plt.ylabel('Objective')
        plt.plot(range(len(self.values)), self.values); plt.show()
