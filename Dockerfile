# use official python image
FROM python:3.10-slim

# set working directory
WORKDIR /app

# copy reqiuirements file first (better for caching)
COPY requirements.txt .

# install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# copy project files
COPY . .

# exspose FastAPI port
EXPOSE 8000

# Run the FastAPI server
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]