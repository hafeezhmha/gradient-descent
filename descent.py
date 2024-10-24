# import numpy as np
# import matplotlib.pyplot as plt
# import streamlit as st

# # Define the functions and their derivatives
# def quadratic(x):
#     return (x - 1)**2

# def quadratic_derivative(x):
#     return 2 * (x - 1)

# def cubic(x):
#     return (x - 2)**3

# def cubic_derivative(x):
#     return 3 * (x - 2)**2

# def sinusoidal(x):
#     return np.sin(x)

# def sinusoidal_derivative(x):
#     return np.cos(x)

# def exponential(x):
#     return np.exp(x)

# def exponential_derivative(x):
#     return np.exp(x)

# def plot_gradient_descent(f, df, starting_point, learning_rate, num_iterations):
#     x = starting_point
#     x_history = [x]
#     y_history = [f(x)]

#     for _ in range(num_iterations):
#         try:
#             x_new = x - learning_rate * df(x)
#             # Limit the value of x to prevent overflow
#             if np.abs(x_new) > 1e5:  # Adjust this threshold as needed
#                 st.warning("Value of x is too large, stopping gradient descent.")
#                 break
#             x = x_new
#             x_history.append(x)
#             y_history.append(f(x))  # Save the function value at the new x
#         except OverflowError as e:
#             st.warning(f"Overflow error occurred: {e}. Stopping gradient descent.")
#             break

#     return x_history, y_history

# # Streamlit app layout
# st.title("Gradient Descent Visualization")

# function_choice = st.selectbox("Choose a function", (
#     "Quadratic Function",
#     "Cubic Function",
#     "Sinusoidal Function",
#     "Exponential Function",
# ))

# learning_rate = st.slider("Learning Rate", 0.001, 0.1, 0.01)
# num_iterations = st.slider("Number of Iterations", 1, 500, 50)

# # Create a mapping from function names to actual functions
# function_mapping = {
#     "Quadratic Function": (quadratic, quadratic_derivative, 0.0),
#     "Cubic Function": (cubic, cubic_derivative, 1.0),
#     "Sinusoidal Function": (sinusoidal, sinusoidal_derivative, 1.0),
#     "Exponential Function": (exponential, exponential_derivative, 0.0),
# }

# f, df, starting_point = function_mapping[function_choice]
# x_history, y_history = plot_gradient_descent(
#     f,
#     df,
#     starting_point,
#     learning_rate,
#     num_iterations
# )

# # Generate x values for plotting the function
# if function_choice == "Sinusoidal Function":
#     x = np.linspace(-2 * np.pi, 2 * np.pi, 100)
# else:
#     x = np.linspace(-2, 4, 100)

# # Plotting the selected function and the gradient descent path
# fig, ax = plt.subplots()
# ax.plot(x, f(x), label='Function', color='blue')
# ax.scatter(x_history, y_history, color='red', s=50, label='Gradient Descent Steps')
# ax.set_title(f'Gradient Descent on {function_choice}')
# ax.set_xlabel('x-axis')
# ax.set_ylabel('f(x)')
# ax.legend()

# # Display the figure
# st.pyplot(fig)
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# Define the functions and their derivatives
def quadratic(x, a=1):
    return a * (x - 1)**2

def quadratic_derivative(a=1):
    return lambda x: 2 * a * (x - 1)

def cubic(x, a=1, b=0, c=0):
    return a * (x - 2)**3 + b * (x - 2) + c

def cubic_derivative(a=1, b=0):
    return lambda x: 3 * a * (x - 2)**2 + b

def sinusoidal(x, amplitude=1):
    return amplitude * np.sin(x)

def sinusoidal_derivative(amplitude=1):
    return lambda x: amplitude * np.cos(x)

def exponential(x, a=1):
    return a * np.exp(x)

def exponential_derivative(a=1):
    return lambda x: a * np.exp(x)

def plot_gradient_descent(f, df, starting_point, learning_rate, num_iterations):
    x = starting_point
    x_history = [x]
    y_history = [f(x)]
    gradient_history = []

    for _ in range(num_iterations):
        try:
            x_new = x - learning_rate * df(x)
            # Limit the value of x to prevent overflow
            if np.abs(x_new) > 1e5:
                st.warning("Value of x is too large, stopping gradient descent.")
                break
            x = x_new
            x_history.append(x)
            y_history.append(f(x))
            gradient_history.append(df(x))
        except OverflowError as e:
            st.warning(f"Overflow error occurred: {e}. Stopping gradient descent.")
            break

    return x_history, y_history, gradient_history

# Streamlit app layout
st.title("Gradient Descent Visualization")

function_choice = st.selectbox("Choose a function", (
    "Quadratic Function",
    "Cubic Function",
    "Sinusoidal Function",
    "Exponential Function",
))

# Parameters for the selected function
if function_choice == "Quadratic Function":
    a = st.slider("Select coefficient a", 0.1, 3.0, 1.0)
    f = lambda x: quadratic(x, a)
    df = quadratic_derivative(a)

elif function_choice == "Cubic Function":
    a = st.slider("Select coefficient a", -3.0, 3.0, 1.0)
    b = st.slider("Select coefficient b", -3.0, 3.0, 0.0)
    c = st.slider("Select coefficient c", -3.0, 3.0, 0.0)
    f = lambda x: cubic(x, a, b, c)
    df = cubic_derivative(a, b)

elif function_choice == "Sinusoidal Function":
    amplitude = st.slider("Select Amplitude", 0.1, 3.0, 1.0)
    f = lambda x: sinusoidal(x, amplitude)
    df = sinusoidal_derivative(amplitude)

elif function_choice == "Exponential Function":
    a = st.slider("Select coefficient a", 0.1, 3.0, 1.0)
    f = lambda x: exponential(x, a)
    df = exponential_derivative(a)

learning_rate = st.slider("Learning Rate", 0.001, 0.1, 0.01)
num_iterations = st.slider("Number of Iterations", 1, 500, 50)
starting_point = 0.0  # Fixed starting point for simplicity

x_history, y_history, gradient_history = plot_gradient_descent(
    f,
    df,
    starting_point,
    learning_rate,
    num_iterations
)

# Generate x values for plotting the function
if function_choice == "Sinusoidal Function":
    x = np.linspace(-2 * np.pi, 2 * np.pi, 100)
else:
    x = np.linspace(-3, 3, 100)

# Plotting the selected function and the gradient descent path
fig, ax = plt.subplots()
ax.plot(x, f(x), label='Function', color='blue')

# Add arrows to visualize the gradient direction
for i in range(len(x_history) - 1):
    ax.arrow(x_history[i], y_history[i], x_history[i + 1] - x_history[i],
             y_history[i + 1] - y_history[i],
             head_width=0.1, head_length=0.2, fc='red', ec='red')

ax.scatter(x_history, y_history, color='red', s=50, label='Gradient Descent Steps')
ax.set_title(f'Gradient Descent on {function_choice}')
ax.set_xlabel('x-axis')
ax.set_ylabel('f(x)')
ax.legend()

# Display the figure
st.pyplot(fig)

# Sidebar for displaying current values
st.sidebar.header("Current Values")
st.sidebar.markdown(f"**Current x:** {x_history[-1]:.4f}")
st.sidebar.markdown(f"**Current f(x):** {y_history[-1]:.4f}")
st.sidebar.markdown(f"**Current gradient:** {gradient_history[-1]:.4f}")
