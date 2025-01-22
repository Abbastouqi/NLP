import os
import yaml

def create_project_structure():
    # Define the project structure
    project_structure = {
        'data/': ['raw_data/', 'processed_data/'],
        'models/': ['bert_embeddings/', 'gpt2_lora/'],
        'src/': ['data_preprocessing.py', 'bert_encoder.py', 'gpt2_generator.py', 'training.py'],
        'utils/': ['helpers.py', 'evaluation.py'],
        'config/': ['config.yaml'],
        'notebooks/': ['exploration.ipynb', 'evaluation.ipynb']
    }
    
    # Create directories and files
    for directory, contents in project_structure.items():
        # Create the main directory
        os.makedirs(directory.rstrip('/'), exist_ok=True)
        
        # Create subdirectories and files
        for item in contents:
            full_path = os.path.join(directory, item)
            
            # If item ends with '/', it's a directory
            if item.endswith('/'):
                os.makedirs(full_path.rstrip('/'), exist_ok=True)
            else:
                # Create empty file
                with open(full_path, 'w') as f:
                    if item == 'config.yaml':
                        # Add some basic configuration
                        config = {
                            'data': {
                                'raw_data_path': '../data/raw_data',
                                'processed_data_path': '../data/processed_data'
                            },
                            'model': {
                                'bert_model': 'bert-base-uncased',
                                'gpt2_model': 'gpt2'
                            },
                            'training': {
                                'batch_size': 32,
                                'learning_rate': 2e-5,
                                'num_epochs': 3
                            }
                        }
                        yaml.dump(config, f, default_flow_style=False)
                    elif item.endswith('.py'):
                        # Add python file template
                        f.write('"""\nDescription: [Add description here]\n"""\n\ndef main():\n    pass\n\nif __name__ == "__main__":\n    main()')
                    elif item.endswith('.ipynb'):
                        # Add Jupyter notebook template
                        notebook_template = {
                            "cells": [],
                            "metadata": {},
                            "nbformat": 4,
                            "nbformat_minor": 4
                        }
                        import json
                        json.dump(notebook_template, f)

if __name__ == "__main__":
    create_project_structure()