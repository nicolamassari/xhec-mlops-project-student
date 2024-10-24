# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Install Prefect for server management
RUN pip install prefect

# Copy the rest of the application code into the container
COPY . .

# Copy the run_services.sh script into the container
COPY bin/run_services.sh /app/bin/run_services.sh

# Ensure the run_services.sh script is executable
RUN chmod +x /app/bin/run_services.sh

# Expose the ports for Prefect server and Uvicorn
EXPOSE 4201
EXPOSE 8001

# Use run_services.sh as the entry point
ENTRYPOINT ["/app/bin/run_services.sh"]
