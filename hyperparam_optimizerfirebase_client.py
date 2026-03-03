"""
Firebase client for state management and real-time data streaming.
Provides persistence for optimization results and intermediate states.
"""
import os
import json
import logging
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError

try:
    import firebase_admin
    from firebase_admin import credentials, firestore, exceptions
    FIREBASE_AVAILABLE = True
except ImportError:
    FIREBASE_AVAILABLE = False
    logging.warning("firebase-admin not available. Using mock client.")

from .config import config

logger = logging.getLogger(__name__)


class FirebaseClient:
    """Firebase Firestore client for optimization state management"""
    
    def __init__(self):
        self.client = None
        self.collection = None
        self.initialized = False
        
        if FIREBASE_AVAILABLE and os.getenv("USE_FIREBASE", "false").lower() == "true":
            self._initialize_firebase()
        else:
            logger.info("Firebase client running in mock mode")
    
    def _initialize_firebase(self) -> None:
        """Initialize Firebase connection"""
        try:
            if not firebase_admin._apps:
                creds_path = config.firebase.credentials_path
                if os.path.exists(creds_path):
                    cred = credentials.Certificate(creds_path)
                    firebase_admin.initialize_app(cred)
                    logger.info("Firebase initialized with service account")
                else:
                    # Try environment variable or default credentials
                    firebase_admin.initialize_app()
                    logger.info("Firebase initialized with default credentials")
            
            self.client = firestore.client()
            self.collection = self.client.collection(config.firebase.collection_name)
            self.initialized