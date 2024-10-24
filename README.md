# Gradient Descent Visualization

This project is a Streamlit application that visualizes the gradient descent algorithm applied to different mathematical functions. Users can choose from a variety of functions, adjust learning rates, and see how the algorithm converges towards a minimum in real-time.

## Features

- **Function Selection**: Choose from five different functions:
  - Quadratic Function: \( f(x) = (x - 1)^2 \)
  - Cubic Function: \( f(x) = (x - 2)^3 \)
  - Sinusoidal Function: \( f(x) = \sin(x) \)
  - Exponential Function: \( f(x) = e^x \)
  - Custom Function: Enter any function to visualize gradient descent.
  
- **Real-time Updates**: Watch as the current value of \( x \), \( f(x) \), and the gradient are updated in real-time.
  
- **Dynamic Controls**: Use sliders to adjust the learning rate and the number of iterations.

- **Interactive Visualization**: Visualize the path taken by the gradient descent algorithm on a plot.

## Installation

To run this application, ensure you have Python 3.x installed along with Streamlit. Follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/gradient-descent-visualization.git
2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
3. **Run the application**
   ```bash
   streamlit run descent.py

# Usage
Choose a Function: Select a function from the dropdown menu.
Adjust Learning Rate: Use the slider to set the learning rate for gradient descent.
Set Number of Iterations: Choose the number of iterations to run the algorithm.
Watch the Results: The current values will be displayed on the left sidebar, and you can see the path taken by the algorithm on the graph.
