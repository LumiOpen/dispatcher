import json
import sys
import os
import numpy as np
import pandas as pd
from argparse import ArgumentParser
import matplotlib.pyplot as plt


metrics = [
    'prompt_level_strict_acc',
    'prompt_level_loose_acc',
    'inst_level_strict_acc',
    'inst_level_loose_acc'
]

instruct_types = [
    'change_case:capital_word_frequency', 
    'change_case:english_capital', 
    'change_case:english_lowercase', 
    'combination:repeat_prompt', 
    'combination:two_responses', 
    'detectable_content:number_placeholders', 
    'detectable_content:postscript', 
    'detectable_format:constrained_response', 
    'detectable_format:json_format', 
    'detectable_format:multiple_sections', 
    'detectable_format:number_bullet_lists', 
    'detectable_format:number_highlighted_sections', 
    'detectable_format:title', 
    'keywords:existence', 
    'keywords:forbidden_words', 
    'keywords:frequency', 
    'keywords:letter_frequency', 
    'language:response_language', 
    'length_constraints:nth_paragraph_first_word', 
    'length_constraints:number_paragraphs', 
    'length_constraints:number_sentences', 
    'length_constraints:number_words', 
    'punctuation:no_comma', 
    'startend:end_checker', 
    'startend:quotation'
]

instruct_categories = [
    'change_case', 
    'combination', 
    'detectable_content', 
    'detectable_format', 
    'keywords', 
    'language', 
    'length_constraints', 
    'punctuation', 
    'startend'
]

response_lang_instruction = 'language:response_language'
                        
def argparser():
    ap = ArgumentParser()
    ap.add_argument(
        "--ifeval_jsonl",
        type=str,
        nargs="+",
        default=None,
        help="list of ifeval jsonl samples to analyse",
    )
    ap.add_argument(
        "--display_names",
        type=str,
        nargs="+",
        default=None,
        help="display names for the jsonl samples (used only in plot legend)",
    )
    ap.add_argument('--lang', default=None, type=str)
    return ap


def analyse_ifeval_results(results_path):
    # print(f"Analysing IFEval results from: {results_path}")
    results = [json.loads(line) for line in open(results_path)]
    accuracy_all = {metric: [] for metric in metrics}
    accuracy_instruct_type = {instruct: [] for instruct in instruct_types}
    accuracy_instruct_category = {instruct: [] for instruct in instruct_categories}
    for i, res in enumerate(results):
        for metric in metrics:
            if isinstance(res[metric], list):
                accuracy_all[metric].extend(res[metric])
            else:
                accuracy_all[metric].append(res[metric])
            if 'prompt_level' not in metric:
                if metric == 'inst_level_strict_acc':
                    for inst_index, instruct in enumerate(res['doc']['instruction_id_list']):
                        inst_category = instruct.split(":")[0]
                        accuracy_instruct_type[instruct].append(res[metric][inst_index])
                        accuracy_instruct_category[inst_category].append(res[metric][inst_index])
    print("==== Overall accuracy ====")
    for metric in accuracy_all:
        print(f"{metric}: {sum(accuracy_all[metric])/len(accuracy_all[metric])}")
        print("correct:", sum(accuracy_all[metric]))
        print("total:", len(accuracy_all[metric]))
        print("--"*20)
    return accuracy_instruct_category

def plot_accuracy_by_instruction_category(accuracy_by_inst_category, lang=None):
    # Extract instruction types and their accuracies
    display_names = list(accuracy_by_inst_category.keys())
    print("Plotting accuracy by instruction group")
    accuracy_df = {
                    'name': [],
                    'instruct_category': [],
                    'accuracy': []
                }
    for display_name in display_names:
        for instruct_cat in accuracy_by_inst_category[display_names[0]]:
            accuracy_df['name'].append(display_name)
            accuracy_df['instruct_category'].append(instruct_cat)
            accuracy_df['accuracy'].append(sum(accuracy_by_inst_category[display_name][instruct_cat]) / 
                                            len(accuracy_by_inst_category[display_name][instruct_cat]))
    dataframe = pd.DataFrame(accuracy_df)
    print("dataframe:", dataframe)
    # Create a bar plot
    # Set the width of the bars
    bar_width = 0.35
    # Set the positions of the bars on the x-axis
    instruction_categories = dataframe['instruct_category'].unique()
    r = np.arange(len(instruction_categories))
    display_names = dataframe['name'].unique()
    for i, display_name in enumerate(display_names):
        subset = dataframe[dataframe['name'] == display_name]
        accuracies = [subset[subset['instruct_category'] == category]['accuracy'].values[0] if category in subset['instruct_category'].values else 0 for category in instruction_categories]
        plt.bar(r + i * bar_width, accuracies, width=bar_width, label=display_name, alpha=0.7)
    plt.ylim(0, 1)  # Set y-axis limits from 0 to 1
    plt.xticks(r + bar_width / 2, instruction_categories, rotation=45, ha='right')
    plt.legend(title='IFEval')
    plt.tight_layout()
    output_plot = 'ifeval_acc_by_instruction_category.png'
    plt.savefig(output_plot)
    print(f"Saved plot to {output_plot}")


def main(argv):
    args = argparser().parse_args(argv[1:])
    json_files = args.ifeval_jsonl
    display_names = args.display_names
    analysis_results = {}
    for file_path, display_name in zip(json_files, display_names):
        print(f"\nAnalyzing IFEval results for {display_name} (jsonl file: {file_path})")
        accuracy_by_category = analyse_ifeval_results(file_path)
        analysis_results[display_name] = accuracy_by_category
    plot_accuracy_by_instruction_category(analysis_results, lang=args.lang)


if __name__ == '__main__':
    sys.exit(main(sys.argv))