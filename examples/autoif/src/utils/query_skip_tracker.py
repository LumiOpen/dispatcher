class QuerySkipTracker:
    """Tracks statistics about skipped queries during loading."""
    def __init__(self):
        self.total = 0
        self.reasons = {
            'length': 0,  # Query exceeds max length
            'key_not_in_item': 0,  # Required key missing from item
            'not_enough_turns': 0,  # Not enough conversation turns
            'json_decode_error': 0,  # JSON parsing error
            'invalid_format': 0  # General format issues
        }
    
    def skip(self, reason: str):
        """Record a skipped query with the given reason."""
        self.total += 1
        if reason in self.reasons:
            self.reasons[reason] += 1
    
    def print_summary(self):
        """Print summary of skipped queries."""
        print("\nSkipped Queries Summary:")
        print(f"Total skipped: {self.total}")
        for reason, count in self.reasons.items():
            if count > 0:
                print(f"  - {reason}: {count}")
