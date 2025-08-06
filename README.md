# Running the Insurance Coverage API Locally
1. Clone the repo: 'git clone https://github.com/premklp16/coverage-checker-llm.git'
2. Install dependencies: 'pip install -r requirements.txt'
3. Place the dataset in the root directory.
4. Run the server: 'python insurance_api.py'
5. Test the endpoint: 'curl -X POST "http://localhost:8000/predict" -F "file=@dataset.pdf" -F "scenario=Emergency treatment for a car accident" -H "Content-Type: multipart/form-data"
