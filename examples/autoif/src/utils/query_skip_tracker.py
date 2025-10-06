class QuerySkipTracker:
    """Tracks statistics about skipped queries during loading."""
    def __init__(self):
        self.total = 0
        self.reasons = {}
    
    def skip(self, reason: str):
        """Record a skipped query with the given reason."""
        self.total += 1
        self.reasons[reason] = self.reasons.get(reason, 0) + 1
    
    def print_summary(self):
        """Print summary of skipped queries."""
        print("\nSkipped Queries Summary:")
        print(f"Total skipped: {self.total}")
        for reason, count in self.reasons.items():
            if count > 0:
                print(f"  - {reason}: {count}")
