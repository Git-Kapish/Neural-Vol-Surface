"""Save today's option-chain snapshot. Schedule daily to build real history."""
from nvs.config import setup_logging
from nvs.data.fetch import save_snapshot

if __name__ == "__main__":
    setup_logging()
    save_snapshot()
