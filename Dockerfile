FROM python:3.10

# Set working directory inside the container
WORKDIR /app

# Copy everything first (including setup.py)
COPY . /app

# Install Python dependencies (includes -e .)
RUN pip install --upgrade pip \
 && pip install  -r requirements.txt

# Expose Streamlit's default port
EXPOSE 8501

# Start Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]