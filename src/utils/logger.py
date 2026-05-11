import logging
import sys

# Central logger configuration - Console only
# We use the root logger so that all modules (including TF/MediaPipe) follow the same format
def setup_logging():
    root = logging.getLogger()
    
    # Remove any existing handlers to avoid duplicates
    for handler in root.handlers[:]:
        root.removeHandler(handler)
        
    root.setLevel(logging.INFO)
    
    # Create console handler pointing to stdout
    # We explicitly set the stream to handle potential encoding issues by using 'backslashreplace' if needed,
    # but standard stdout is usually fine if we avoid emojis.
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    
    # Professional format
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S')
    handler.setFormatter(formatter)
    
    root.addHandler(handler)
    
    # Also get our specific project logger
    logger = logging.getLogger("ISL_Pipeline")
    logger.setLevel(logging.INFO)
    # Ensure it doesn't add its own handlers and propagate to root
    logger.propagate = True 
    
    return logger

# Initialize once
logger = setup_logging()
