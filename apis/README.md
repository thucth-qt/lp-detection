# Features
- Vehicle detection([api/detector/vehicle](api/detector/vehicle))
- License plate detection([api/detector/lp](api/detector/lp))
- License plate in vehicle detection([api/detector/lp-in-vehicle](api/detector/lp-in-vehicle))

# Setup environment
Install virtual environment using pip as command below:
```bash
# Create virtual environment
virtualvenv -p python3 venv
# Activate virtual environment
source venv/bin/activate
# Install nessesary packages
pip install -r requirements.txt
```

# Run backend
After setting up completely, run backend API using command below:
```bash
python apis/backend/manage.py runserver
```