FROM python:3.11-slim

# dossier de travail
WORKDIR /app

# copier les requirements
COPY requirements.txt .

# installer les dependances
RUN pip install --no-cache-dir -r requirements.txt

# copier le projet
COPY . .

# exposer le port de jupyter
EXPOSE 8888

# lancer jupyter notebook
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]