"""
Smart Trash AI Classification Service

Run with: python run.py
"""

import os
import uvicorn
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8081"))
    reload = os.getenv("ENV", "development") == "development"

    print(f"\nStarting server on http://{host}:{port}")
    print(f"API Docs: http://{host}:{port}/docs")
    print(f"Reload: {reload}\n")

    uvicorn.run(
        "src.presentation.api.main:app",
        host=host,
        port=port,
        reload=reload
    )
