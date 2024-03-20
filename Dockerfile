# Use an official Python runtime as a parent image
FROM python:3.9

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed dependencies specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 8501 available to the world outside this container
EXPOSE 8501

# Define environment variable
ENV STREAMLIT_SERVER_PORT=8501

# Set PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION environment variable
ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

# Run the Streamlit app when the container launc
CMD ["streamlit", "run", "app.py"]

