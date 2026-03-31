import argparse
import random
import json
import os
from typing import List, Dict, Tuple, Optional

from src.utils.query_skip_tracker import QuerySkipTracker


class InstructionSelector:
    """Handles instruction selection with uniform usage distribution."""

    def __init__(self, verifiers_list: List[Dict]):
        self.verifiers = {v['instruction_id']: v for v in verifiers_list}
        self.instruction_usage_count = {v['instruction_id']: 0 for v in verifiers_list}

    def select_instructions(self, n: int, exclude: Optional[set] = None) -> List[Dict]:
        """Select n instructions maintaining uniform usage distribution.

        Prefers instructions with the lowest usage count, breaking ties randomly.
        Instructions in ``exclude`` are skipped unless there aren't enough
        alternatives.
        """
        exclude = exclude or set()
        available_ids = [iid for iid in self.verifiers if iid not in exclude]
        if len(available_ids) < n:
            available_ids = list(self.verifiers.keys())

        min_usage = min(self.instruction_usage_count[iid] for iid in available_ids)
        selected = []
        current_usage = min_usage

        while len(selected) < n:
            candidates = [iid for iid in available_ids
                         if self.instruction_usage_count[iid] == current_usage
                         and iid not in selected]
            if not candidates:
                current_usage += 1
                continue
            needed = min(n - len(selected), len(candidates))
            selected.extend(random.sample(candidates, needed))

        for iid in selected:
            self.instruction_usage_count[iid] += 1
        return [self.verifiers[iid] for iid in selected]

    def select_instructions_multi_turn(
        self,
        instructions_per_turn: "int | List[float]",
        turns: int,
    ) -> List[List[Dict]]:
        """Select instructions for a multi-turn conversation with accumulation.

        Every sample gets exactly ``turns`` turns. Instructions accumulate
        across turns so that turn N contains all instructions from turns 0..N.

        Args:
            instructions_per_turn: Either a fixed int (every turn gets this
                many new instructions) or a list of weights where weight[i]
                is the relative probability of adding i+1 new instructions
                on each turn (sampled independently per turn).
            turns: Number of conversation turns.

        Returns:
            List of ``turns`` instruction lists, each containing all
            accumulated instructions up to and including that turn.
        """
        used = set()
        accumulated = []
        result = []

        for _ in range(turns):
            if isinstance(instructions_per_turn, list):
                choices = list(range(1, len(instructions_per_turn) + 1))
                n = random.choices(choices, weights=instructions_per_turn, k=1)[0]
            else:
                n = instructions_per_turn

            new_verifiers = self.select_instructions(n, exclude=used)
            for v in new_verifiers:
                used.add(v['instruction_id'])
            accumulated.extend(new_verifiers)
            result.append(accumulated.copy())

        return result

    def get_usage_stats(self) -> Dict:
        """Get current usage statistics."""
        return {'instruction_usage': dict(self.instruction_usage_count)}


def load_verifiers(path: str) -> List[Dict]:
    """Load validated verifiers, keeping only passing eval functions.

    Filters to entries with success=True and non-empty passing_functions.
    Test cases and other validation metadata are dropped.
    """
    verifiers = []
    skipped = 0
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)

            if not entry.get('success', False):
                skipped += 1
                continue

            passing_funcs = entry.get('passing_functions', [])
            if not passing_funcs:
                skipped += 1
                continue

            verifiers.append({
                'instruction_id': entry['instruction_id'],
                'instruction': entry['instruction'],
                'instruction_category': entry.get('instruction_category', 'default'),
                'placeholders': entry.get('placeholders', {}),
                'eval_funcs': passing_funcs,
            })

    print(f"Loaded {len(verifiers)} validated verifiers (skipped {skipped} failed/empty)")
    return verifiers


def load_queries(
    path: str, query_max_len: Optional[int] = None
) -> Tuple[List[Dict], QuerySkipTracker]:
    """Load queries from a JSONL file or directory of JSONL files.

    Expected format per line: {"messages": [{"role": "user", "content": "..."}], ...}
    """
    queries = []
    tracker = QuerySkipTracker()

    files = []
    if os.path.isdir(path):
        for fname in sorted(os.listdir(path)):
            if fname.endswith(('.jsonl', '.json')):
                files.append(os.path.join(path, fname))
        if not files:
            print(f"Warning: no JSONL files found in directory {path}")
    else:
        files.append(path)

    for fpath in files:
        with open(fpath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    tracker.skip('json_decode_error')
                    continue

                messages = item.get('messages')
                if not messages or not isinstance(messages, list):
                    tracker.skip('no_messages')
                    continue

                user_msgs = [m for m in messages if m.get('role') == 'user']
                if not user_msgs:
                    tracker.skip('no_user_message')
                    continue

                if query_max_len and any(
                    len(m.get('content', '')) > query_max_len for m in user_msgs
                ):
                    tracker.skip('exceeds_max_len')
                    continue

                metadata = {k: v for k, v in item.items() if k != 'messages'}
                queries.append({
                    'messages': messages,
                    'metadata': metadata,
                })

    return queries, tracker


def create_output_entry(
    query: Dict, verifiers_by_turn: List[List[Dict]], source: str
) -> Dict:
    """Create output entry combining query messages with per-turn constraint selections.

    Constraints are stored once in a top-level dict keyed by constraint_id.
    Each turn entry only references constraint_ids to avoid duplicating data
    across accumulated turns.
    """
    constraints = {}
    turns = []
    for turn_verifiers in verifiers_by_turn:
        for v in turn_verifiers:
            cid = v['instruction_id']
            if cid not in constraints:
                constraints[cid] = {
                    'constraint': v['instruction'],
                    'category': v.get('instruction_category', 'default'),
                    'placeholders': v.get('placeholders', {}),
                    'eval_funcs': v['eval_funcs'],
                }
        turns.append({
            'constraint_ids': [v['instruction_id'] for v in turn_verifiers],
        })

    return {
        'messages': query['messages'],
        'query_metadata': query['metadata'],
        'constraints': constraints,
        'turns': turns,
        'source': source,
    }


def concat_queries(
    verifiers_path: str,
    queries_path: str,
    output_file: str,
    num_output_lines: Optional[int] = None,
    instructions_per_turn: "int | List[float]" = 1,
    turns: int = 3,
    query_max_len: Optional[int] = None,
) -> int:
    """Concatenate queries with validated instruction verifiers for response generation.

    Each output sample combines a query (in messages format) with a fixed
    number of turns. Instructions accumulate across turns so that the
    response job can dynamically build prompts from this data.
    """
    verifiers_list = load_verifiers(verifiers_path)
    if not verifiers_list:
        print("Error: no valid verifiers found")
        return 0

    queries, tracker = load_queries(queries_path, query_max_len=query_max_len)
    tracker.print_summary()
    print(f"Total queries loaded: {len(queries)}")

    if not queries:
        print("Error: no queries loaded")
        return 0

    selector = InstructionSelector(verifiers_list)

    num_iterations = num_output_lines if num_output_lines is not None else len(queries)
    instruction_count_distribution: Dict[int, int] = {}

    count = 0
    with open(output_file, 'w') as f:
        for i in range(num_iterations):
            query = queries[i % len(queries)]

            verifiers_by_turn = selector.select_instructions_multi_turn(
                instructions_per_turn, turns
            )

            total_instructions = len(verifiers_by_turn[-1])
            instruction_count_distribution[total_instructions] = (
                instruction_count_distribution.get(total_instructions, 0) + 1
            )

            output = create_output_entry(query, verifiers_by_turn, source=queries_path)
            f.write(json.dumps(output, ensure_ascii=False) + '\n')
            count += 1

    print(f"\nGenerated {count} samples to {output_file}")
    print(f"Turns: {turns}, instructions per turn: {instructions_per_turn}")
    print(f"Total instruction count distribution: "
          f"{dict(sorted(instruction_count_distribution.items()))}")
    if num_output_lines is not None and count > len(queries):
        print(f"Queries reused: {count - len(queries)} times")

    usage = selector.get_usage_stats()
    vals = list(usage['instruction_usage'].values())
    print(f"Instruction usage: min={min(vals)}, max={max(vals)}, "
          f"mean={sum(vals)/len(vals):.1f}")

    return count


def main():
    parser = argparse.ArgumentParser(
        description='Concatenate validated instructions with query data for response generation'
    )
    parser.add_argument('--verifiers-file', type=str, required=True,
                        help='Validated verifiers JSONL file (from cross-validation)')
    parser.add_argument('--queries-path', type=str, required=True,
                        help='JSONL file or directory of JSONL files with messages-format queries')
    parser.add_argument('--output-file', type=str, required=True,
                        help='Output JSONL file path')
    parser.add_argument('--num-output-lines', type=int, default=None,
                        help='Number of output samples (reuses queries cyclically). '
                             'Default: one sample per query.')
    parser.add_argument('--instructions-per-turn', type=str, default='1',
                        help='Either a single integer for a fixed count of new instructions '
                             'per turn, or a comma-separated list of weights where the i-th '
                             'weight is the relative probability of adding i new instructions '
                             'on each turn (e.g., "1,0.5,0.25" means P(1)∝1, P(2)∝0.5, '
                             'P(3)∝0.25). Default: "1"')
    parser.add_argument('--turns', type=int, default=3,
                        help='Number of conversation turns (default: 3)')
    parser.add_argument('--query-max-len', type=int, default=None,
                        help='Maximum user message length in characters (skip longer queries)')

    args = parser.parse_args()

    if ',' in args.instructions_per_turn:
        instructions_per_turn = [float(x) for x in args.instructions_per_turn.split(',')]
    else:
        instructions_per_turn = int(args.instructions_per_turn)

    concat_queries(
        verifiers_path=args.verifiers_file,
        queries_path=args.queries_path,
        output_file=args.output_file,
        num_output_lines=args.num_output_lines,
        instructions_per_turn=instructions_per_turn,
        turns=args.turns,
        query_max_len=args.query_max_len,
    )


if __name__ == "__main__":
    main()
