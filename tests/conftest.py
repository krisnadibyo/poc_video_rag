import pytest
import os
from dotenv import load_dotenv

@pytest.fixture(autouse=True)
def load_env():
    """Load environment variables before each test"""
    load_dotenv()

@pytest.fixture(autouse=True)
def mock_directories():
    """Create and clean up test directories"""
    # Create test directories
    os.makedirs("audio", exist_ok=True)
    os.makedirs("transcripts", exist_ok=True)
    
    yield
    
    # Clean up test files after tests
    for file in os.listdir("audio"):
        os.remove(os.path.join("audio", file))
    for file in os.listdir("transcripts"):
        os.remove(os.path.join("transcripts", file)) 