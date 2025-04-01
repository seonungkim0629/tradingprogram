"""
Recovery and safe shutdown system for Bitcoin Trading Bot

This module provides functionality for system state persistence, recovery from
unexpected shutdowns, and safe termination of trading activities.
"""

import os
import json
import time
import signal
import threading
import atexit
from datetime import datetime, timedelta
from typing import Callable, Dict, Any, Optional

from utils.logging import get_logger

# Initialize logger
logger = get_logger(__name__)

# Constants
CHECKPOINT_DIR = "checkpoints"
CHECKPOINT_INTERVAL = 300  # seconds (5 minutes)
MAX_CHECKPOINT_FILES = 10

# Ensure checkpoint directory exists
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Global variables
_current_state = {}
_last_checkpoint_time = 0
_checkpoint_thread = None
_stop_flag = threading.Event()
_shutdown_handlers = []
_shutdown_in_progress = False
_checkpoint_data_callback = None
_state_data_callback = None

def register_state_provider(key, provider_func):
    """
    Register a function that provides state data for a particular component
    
    Args:
        key (str): The key to identify this component's state
        provider_func (callable): A function that returns the current state dict for the component
    """
    if not callable(provider_func):
        logger.error(f"Provider for {key} is not callable")
        return
    
    logger.debug(f"Registered state provider for {key}")
    _current_state[key] = {'provider': provider_func, 'data': None}

def update_state(key, data):
    """
    Update the state data for a component
    
    Args:
        key (str): The component key
        data: The state data to store
    """
    if key not in _current_state:
        _current_state[key] = {'provider': None, 'data': None}
    
    _current_state[key]['data'] = data
    logger.debug(f"Updated state for {key}")

def get_state(key=None):
    """
    Get the current state
    
    Args:
        key (str, optional): If provided, return only this component's state
        
    Returns:
        The current state dict or component state
    """
    if key is not None:
        if key in _current_state:
            # If a provider function exists, call it to get the latest state
            if _current_state[key]['provider']:
                try:
                    _current_state[key]['data'] = _current_state[key]['provider']()
                except Exception as e:
                    logger.error(f"Error getting state from provider for {key}: {str(e)}")
            
            return _current_state[key]['data']
        else:
            logger.warning(f"No state found for key: {key}")
            return None
    
    # Return all state data
    result = {}
    for k, v in _current_state.items():
        # Get latest state from providers
        if v['provider']:
            try:
                v['data'] = v['provider']()
            except Exception as e:
                logger.error(f"Error getting state from provider for {k}: {str(e)}")
        
        result[k] = v['data']
    
    return result

def save_checkpoint(force=False):
    """
    Save the current state to a checkpoint file
    
    Args:
        force (bool): If True, save regardless of time since last checkpoint
        
    Returns:
        bool: True if checkpoint was saved, False otherwise
    """
    global _last_checkpoint_time
    
    current_time = time.time()
    if not force and current_time - _last_checkpoint_time < CHECKPOINT_INTERVAL:
        logger.debug("Skipping checkpoint, too soon since last checkpoint")
        return False
    
    # Get latest state
    state = get_state()
    
    # Use checkpoint data callback if available
    if _checkpoint_data_callback:
        try:
            checkpoint_data = _checkpoint_data_callback()
            if checkpoint_data:
                state = {**state, **checkpoint_data}
        except Exception as e:
            logger.error(f"Error in checkpoint data callback: {str(e)}")
    
    # Skip if empty state
    if not state:
        logger.warning("No state to checkpoint")
        return False
    
    # Generate timestamp for filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_file = os.path.join(CHECKPOINT_DIR, f"checkpoint_{timestamp}.json")
    
    try:
        with open(checkpoint_file, 'w') as f:
            # Include metadata
            checkpoint_data = {
                'timestamp': timestamp,
                'version': '1.0',
                'state': state
            }
            json.dump(checkpoint_data, f, indent=2, default=str)
        
        _last_checkpoint_time = current_time
        logger.info(f"Checkpoint saved to {checkpoint_file}")
        
        # Cleanup old checkpoint files
        cleanup_old_checkpoints()
        
        return True
    
    except Exception as e:
        logger.error(f"Error saving checkpoint: {str(e)}")
        return False

def load_latest_checkpoint():
    """
    Load the latest checkpoint file
    
    Returns:
        dict: The loaded state, or None if no checkpoint found
    """
    checkpoint_files = []
    
    try:
        # Get list of checkpoint files
        for file in os.listdir(CHECKPOINT_DIR):
            if file.startswith("checkpoint_") and file.endswith(".json"):
                checkpoint_files.append(os.path.join(CHECKPOINT_DIR, file))
        
        if not checkpoint_files:
            logger.info("No checkpoint files found")
            return None
        
        # Sort by modification time (newest first)
        checkpoint_files.sort(key=os.path.getmtime, reverse=True)
        latest_checkpoint = checkpoint_files[0]
        
        # Load the checkpoint
        with open(latest_checkpoint, 'r') as f:
            checkpoint_data = json.load(f)
        
        logger.info(f"Loaded checkpoint from {latest_checkpoint}")
        return checkpoint_data['state']
    
    except Exception as e:
        logger.error(f"Error loading checkpoint: {str(e)}")
        return None

def cleanup_old_checkpoints():
    """Delete old checkpoint files, keeping only the most recent ones"""
    try:
        checkpoint_files = []
        
        for file in os.listdir(CHECKPOINT_DIR):
            if file.startswith("checkpoint_") and file.endswith(".json"):
                checkpoint_files.append(os.path.join(CHECKPOINT_DIR, file))
        
        # Sort by modification time (oldest first)
        checkpoint_files.sort(key=os.path.getmtime)
        
        # If we have more than the max, delete the oldest ones
        while len(checkpoint_files) > MAX_CHECKPOINT_FILES:
            oldest = checkpoint_files.pop(0)
            os.remove(oldest)
            logger.debug(f"Deleted old checkpoint {oldest}")
    
    except Exception as e:
        logger.error(f"Error cleaning up checkpoints: {str(e)}")

def _checkpoint_loop():
    """Background thread for periodic checkpoints"""
    logger.info("Checkpoint thread started")
    
    while not _stop_flag.is_set():
            save_checkpoint()
        
            # Wait for next checkpoint time or until stopped
            _stop_flag.wait(CHECKPOINT_INTERVAL)
        

def was_abnormal_shutdown():
    """
    Check if the last shutdown was abnormal
    
    Returns:
        bool: True if abnormal shutdown detected, False otherwise
    """
    # Load the latest checkpoint
    latest_state = load_latest_checkpoint()
    
    if not latest_state:
        return False
    
    # Check for clean shutdown flag
    if 'clean_shutdown' in latest_state and latest_state['clean_shutdown'] is True:
        return False
    
    # Check timestamp - if it's recent (within last 24h), it might be abnormal
    if 'timestamp' in latest_state:
        try:
            checkpoint_time = datetime.fromisoformat(latest_state['timestamp'])
            now = datetime.now()
            
            # If checkpoint is older than 24 hours, probably not an abnormal shutdown
            if now - checkpoint_time > timedelta(hours=24):
                return False
            
            # Recent checkpoint without clean shutdown flag suggests abnormal shutdown
            return True
        except:
            pass
    
    # If we can't determine, assume it was normal
    return False

def recover_from_checkpoint():
    """
    Attempt to recover from the latest checkpoint
    
    Returns:
        dict: The recovered state, or None if recovery failed
    """
    if not was_abnormal_shutdown():
        logger.info("No abnormal shutdown detected, no recovery needed")
        return None
    
    logger.info("Abnormal shutdown detected, attempting recovery")
    
    # Load the latest state
    state = load_latest_checkpoint()
    
    if not state:
        logger.warning("No checkpoint found for recovery")
        return None
    
    # Restore state to components
    for key, data in state.items():
        update_state(key, data)
    
    logger.info("System state recovered from checkpoint")
    return state

def register_shutdown_handler(handler_func):
    """
    Register a function to be called during shutdown
    
    Args:
        handler_func (callable): Function to call during shutdown
    """
    if not callable(handler_func):
        logger.error("Shutdown handler is not callable")
        return
    
    _shutdown_handlers.append(handler_func)
    logger.debug("Registered shutdown handler")

def _handle_shutdown(signal_num=None, frame=None):
    """Handle shutdowns gracefully"""
    global _shutdown_in_progress
    
    if _shutdown_in_progress:
        logger.warning("Shutdown already in progress, forcing exit")
        os._exit(1)
        return
    
    _shutdown_in_progress = True
    logger.info("Initiating safe shutdown sequence")
    
    # Stop the checkpoint thread
    stop_checkpoint_thread()
    
    # Call all registered shutdown handlers
    for handler in _shutdown_handlers:
        try:
            handler()
        except Exception as e:
            logger.error(f"Error in shutdown handler: {str(e)}")
    
    # Save a final checkpoint with clean shutdown flag
    state = get_state()
    if state is None:
        state = {}
    
    state['clean_shutdown'] = True
    
    # Use state data callback if available
    if _state_data_callback:
        try:
            additional_state = _state_data_callback()
            if additional_state:
                state.update(additional_state)
        except Exception as e:
            logger.error(f"Error in state data callback: {str(e)}")
    
    # Save the final state
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_file = os.path.join(CHECKPOINT_DIR, f"checkpoint_{timestamp}.json")
    
    try:
        with open(checkpoint_file, 'w') as f:
            checkpoint_data = {
                'timestamp': timestamp,
                'version': '1.0',
                'clean_shutdown': True,
                'state': state
            }
            json.dump(checkpoint_data, f, indent=2, default=str)
        logger.info(f"Checkpoint saved to {checkpoint_file}")
    except Exception as e:
        logger.error(f"Error saving final checkpoint: {str(e)}")
    
    logger.info("Shutdown completed successfully")

def start_checkpoint_thread():
    """Start the background checkpoint thread"""
    global _checkpoint_thread, _stop_flag
    
    if _checkpoint_thread and _checkpoint_thread.is_alive():
        logger.warning("Checkpoint thread already running")
        return
    
    _stop_flag.clear()
    _checkpoint_thread = threading.Thread(target=_checkpoint_loop, daemon=True)
    _checkpoint_thread.start()

def stop_checkpoint_thread():
    """Stop the background checkpoint thread"""
    global _checkpoint_thread, _stop_flag
    
    if not _checkpoint_thread or not _checkpoint_thread.is_alive():
        logger.warning("No checkpoint thread running")
        return
    
    logger.info("Stopping checkpoint thread")
    _stop_flag.set()
    _checkpoint_thread.join(timeout=5.0)  # Wait up to 5 seconds for the thread to end

def setup_recovery_system(checkpoint_data_callback: Optional[Callable[[], Dict[str, Any]]] = None,
                         state_data_callback: Optional[Callable[[], Dict[str, Any]]] = None) -> None:
    """
    Setup the recovery system with optional callbacks for state data
    
    Args:
        checkpoint_data_callback (Optional[Callable[[], Dict[str, Any]]]): Function that returns data to include in checkpoints
        state_data_callback (Optional[Callable[[], Dict[str, Any]]]): Function that returns data to include in shutdown state
    """
    global _checkpoint_data_callback, _state_data_callback
    
    # Store callbacks
    _checkpoint_data_callback = checkpoint_data_callback
    _state_data_callback = state_data_callback
    
    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, _handle_shutdown)
    signal.signal(signal.SIGTERM, _handle_shutdown)
        
    # Register exit handler
    atexit.register(_handle_shutdown)

    # Check for recovery from previous abnormal shutdown
    if was_abnormal_shutdown():
        recover_from_checkpoint()
    
    # Start the checkpoint thread
    start_checkpoint_thread()
    
    logger.info("Recovery system initialized")

def create_checkpoint(state_data: Optional[Dict[str, Any]] = None) -> bool:
    """
    Create a checkpoint with the current state
    
    Args:
        state_data (Optional[Dict[str, Any]]): Additional state data to include
        
    Returns:
        bool: True if checkpoint was created successfully
    """
    # Update state with provided data
    if state_data:
        for key, value in state_data.items():
            update_state(key, value)
    
    # Force create a checkpoint
    return save_checkpoint(force=True)

def save_state(key: str, data: Any) -> None:
    """
    Save state data for a component (alias for update_state)
    
    Args:
        key (str): The component key
        data: The state data to store
    """
    update_state(key, data) 