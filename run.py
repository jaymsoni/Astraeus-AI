"""
Entry point for the Astraeus application.
This file ensures proper module imports by adding the project root to PYTHONPATH.
"""
import os
import sys

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Now we can import our app
from backend.app import app

if __name__ == '__main__':
    app.run(debug=True) 